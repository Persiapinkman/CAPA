#!/usr/bin/env python3
"""
视频帧提取工具

支持从视频文件中提取帧，自动检测并使用GPU加速（如果可用），
支持多线程并发处理，支持多种输入方式（单个文件、目录、txt文件列表）。

用法:
    python extract_frames.py <input> [options]

输入类型:
    - 单个视频文件路径
    - 视频文件所在目录
    - 包含视频路径列表的txt文件（每行一个路径）

选项:
    --output-dir: 输出目录（默认：与输入视频同目录下的frames文件夹）
    --fps: 提取帧率（默认：1，即每秒1帧）
    --format: 输出图片格式（默认：jpg，可选：png, jpg）
    --threads: 并发线程数（默认：自动检测CPU核心数）
    --gpu: 强制使用GPU（默认：自动检测）
    --no-gpu: 强制使用CPU
    --quality: JPEG质量（1-100，默认：95）
"""

import argparse
import hashlib
import math
import os
import sys
import subprocess
import time
import json
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from logging_config import get_logger, configure_logging


# ============================================================================
# 常量定义
# ============================================================================

# 支持的视频格式
VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', 
    '.m4v', '.3gp', '.ts', '.mts'
}

# 默认配置
DEFAULT_FPS = 1.0
DEFAULT_IMAGE_FORMAT = 'jpg'
DEFAULT_QUALITY = 80
DEFAULT_PROGRESS_BAR_WIDTH = 28
DEFAULT_PROGRESS_UPDATE_INTERVAL = 1.0  # 秒

# 线程配置
DEFAULT_THREAD_RATIO = 0.5  # CPU核心数的一半
MAX_PROBE_WORKERS = 16  # 预估帧数时的最大并发数

# FFmpeg 配置
FFMPEG_JPEG_QUALITY_MAP = {
    95: 2,  # 高质量
    90: 3,
    80: 4,
    0: 5   # 默认/中等质量
}


# ============================================================================
# 枚举和数据类型
# ============================================================================

class ImageFormat(Enum):
    """支持的图片格式"""
    JPG = 'jpg'
    PNG = 'png'


class GPUAcceleration(Enum):
    """GPU加速模式"""
    AUTO = 'auto'
    FORCE_GPU = 'force_gpu'
    FORCE_CPU = 'force_cpu'


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class ExtractionConfig:
    """视频帧提取配置"""
    input_path: Path
    output_dir: Optional[Path] = None
    fps: float = DEFAULT_FPS
    image_format: str = DEFAULT_IMAGE_FORMAT
    quality: int = DEFAULT_QUALITY
    threads: Optional[int] = None
    gpu_mode: GPUAcceleration = GPUAcceleration.AUTO
    show_progress: bool = True
    verbose: bool = False
    log_dir: Optional[Path] = None
    
    def __post_init__(self):
        """验证和规范化配置"""
        if self.fps <= 0:
            raise ValueError(f"fps 必须大于 0，当前值: {self.fps}")
        if self.image_format not in [fmt.value for fmt in ImageFormat]:
            raise ValueError(f"不支持的图片格式: {self.image_format}")
        if not (1 <= self.quality <= 100):
            raise ValueError(f"quality 必须在 1-100 之间，当前值: {self.quality}")


@dataclass
class ProcessingStats:
    """处理统计信息"""
    total_videos: int = 0
    success_count: int = 0
    failed_count: int = 0
    done_videos: int = 0
    done_expected_frames: int = 0
    total_expected_frames: int = 0
    failed_videos: List[Path] = None
    failed_errors: Dict[Path, str] = None
    
    def __post_init__(self):
        if self.failed_videos is None:
            self.failed_videos = []
        if self.failed_errors is None:
            self.failed_errors = {}


# ============================================================================
# 工具函数
# ============================================================================

def check_ffmpeg() -> bool:
    """检查ffmpeg是否已安装"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_nvidia_gpu() -> bool:
    """检查是否有NVIDIA GPU"""
    try:
        subprocess.run(
            ['nvidia-smi'], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_intel_qsv() -> bool:
    """检查是否有Intel QSV支持"""
    try:
        if not Path('/dev/dri/renderD128').exists():
            return False
        
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return 'h264_qsv' in result.stdout or 'hevc_qsv' in result.stdout
    except Exception:
        return False


def check_gpu_available() -> Tuple[bool, Optional[str]]:
    """
    检查是否有可用的GPU加速
    
    Returns:
        (是否可用, GPU类型描述)
    """
    if check_nvidia_gpu():
        return True, "NVIDIA CUDA"
    if check_intel_qsv():
        return True, "Intel QSV"
    return False, None


def get_video_duration_seconds(video_path: Path) -> Optional[float]:
    """用 ffprobe 获取视频时长（秒）。失败则返回 None。"""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path),
        ]
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout or "{}")
        duration = data.get("format", {}).get("duration")
        if duration is None:
            return None
        return float(duration)
    except Exception:
        return None


def calculate_expected_frames(duration: float, fps: float) -> int:
    """根据视频时长和fps计算预期帧数"""
    if duration <= 0 or fps <= 0:
        return 0
    return max(1, int(math.ceil(duration * fps)))


def get_jpeg_quality_value(quality: int) -> int:
    """将用户输入的quality (1-100) 转换为ffmpeg的-q:v值"""
    if quality >= 95:
        return FFMPEG_JPEG_QUALITY_MAP[95]
    elif quality >= 90:
        return FFMPEG_JPEG_QUALITY_MAP[90]
    elif quality >= 80:
        return FFMPEG_JPEG_QUALITY_MAP[80]
    else:
        return FFMPEG_JPEG_QUALITY_MAP[0]


def _short_hash(path: Path, length: int = 8) -> str:
    """为路径生成短hash，避免输出目录名冲突"""
    h = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()
    return h[:length]


def compute_output_dir(
    video_path: Path,
    base_output_dir: Optional[Path],
    input_root: Optional[Path],
) -> Path:
    """
    计算单个视频的输出目录

    - 未指定 --output-dir：输出到 <video_parent>/frames/<stem>
    - 指定 --output-dir：保留相对路径结构，避免同名视频冲突
    """
    if not base_output_dir:
        return video_path.parent / "frames" / video_path.stem

    if input_root:
        try:
            rel = video_path.resolve().relative_to(input_root.resolve())
            rel_no_suffix = rel.with_suffix("")
            return base_output_dir / rel_no_suffix
        except ValueError:
            pass

    return base_output_dir / f"{video_path.stem}__{_short_hash(video_path)}"


# ============================================================================
# 视频文件收集
# ============================================================================

def collect_video_files(input_path: Path, verbose: bool = False) -> List[Path]:
    """收集所有视频文件"""
    videos = []
    logger = get_logger(__name__)
    
    if input_path.is_file():
        if input_path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(input_path)
        elif input_path.suffix.lower() == '.txt':
            videos.extend(_collect_from_txt(input_path, verbose))
        else:
            logger.error("不支持的文件类型: %s", input_path.suffix)
    elif input_path.is_dir():
        videos.extend(_collect_from_directory(input_path))
    else:
        logger.error("输入路径不存在: %s", input_path)
    
    return sorted(set(videos))


def _collect_from_txt(txt_path: Path, verbose: bool) -> List[Path]:
    """从txt文件读取视频路径列表"""
    videos = []
    logger = get_logger(__name__)
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    video_path = Path(line)
                    if video_path.exists() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                        videos.append(video_path)
                    elif verbose:
                        logger.warning("跳过无效路径: %s", line)
    except Exception as e:
        logger.error("无法读取 txt 文件 %s: %s", txt_path, e)
    
    return videos


def _collect_from_directory(directory: Path) -> List[Path]:
    """从目录递归查找所有视频文件"""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        for p in directory.rglob(f'*{ext}'):
            if p.is_file():
                videos.append(p)
        for p in directory.rglob(f'*{ext.upper()}'):
            if p.is_file():
                videos.append(p)
    return videos


# ============================================================================
# 帧提取核心逻辑
# ============================================================================

class FFmpegCommandBuilder:
    """FFmpeg命令构建器"""
    
    def __init__(
        self,
        video_path: Path,
        output_pattern: Path,
        fps: float,
        image_format: str,
        quality: int,
    ):
        self.video_path = video_path
        self.output_pattern = output_pattern
        self.fps = fps
        self.image_format = image_format
        self.quality = quality
    
    def build(self, use_gpu: bool) -> List[str]:
        """构建ffmpeg命令"""
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-y',
            '-i', str(self.video_path),
        ]
        
        if use_gpu:
            cmd.extend(self._get_gpu_options())
        
        cmd.extend(['-vf', f'fps={self.fps}'])
        
        if self.image_format.lower() == ImageFormat.JPG.value:
            qv = get_jpeg_quality_value(self.quality)
            cmd.extend(['-q:v', str(qv)])
        
        cmd.append(str(self.output_pattern))
        return cmd
    
    def _get_gpu_options(self) -> List[str]:
        """获取GPU加速选项"""
        if check_nvidia_gpu():
            return ['-hwaccel', 'cuda']
        elif check_intel_qsv():
            return ['-hwaccel', 'qsv']
        return []


def extract_frames_single(
    video_path: Path,
    output_dir: Path,
    config: ExtractionConfig,
) -> Tuple[bool, int, Optional[str]]:
    """
    从单个视频提取帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        config: 提取配置
    
    Returns:
        (是否成功, 实际帧数, 错误信息)
    """
    logger = get_logger(__name__)
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = video_path.stem
        output_pattern = output_dir / f"{video_name}_frame_%06d.{config.image_format}"
        
        builder = FFmpegCommandBuilder(
            video_path, output_pattern, config.fps, 
            config.image_format, config.quality
        )
        
        # 根据配置决定是否使用GPU
        use_gpu = False
        if config.gpu_mode == GPUAcceleration.FORCE_GPU:
            use_gpu = True
        elif config.gpu_mode == GPUAcceleration.AUTO:
            use_gpu, _ = check_gpu_available()
        
        # 尝试执行
        cmd = builder.build(use_gpu)
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        # GPU失败时回退到CPU
        if result.returncode != 0 and use_gpu:
            if config.verbose:
                logger.warning("GPU 模式抽帧失败，视频 %s，自动回退到 CPU 模式", video_path.name)
            cmd = builder.build(False)
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
        
        if result.returncode == 0:
            frame_count = len(list(output_dir.glob(f"{video_name}_frame_*.{config.image_format}")))
            if config.verbose:
                logger.info("视频 %s: 成功提取 %d 帧 -> %s", video_path.name, frame_count, output_dir)
            return True, frame_count, None
        else:
            err = (result.stderr or "").strip()
            err_tail = err[-500:] if len(err) > 500 else err
            if config.verbose:
                logger.error("视频 %s: 抽帧失败，错误: %s", video_path.name, err_tail)
            return False, 0, (err_tail or "ffmpeg 返回非0退出码")
            
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if config.verbose:
            logger.exception("视频 %s: 处理出错 - %s", video_path.name, msg)
        return False, 0, msg


def process_video(
    video_path: Path,
    config: ExtractionConfig,
    base_output_dir: Optional[Path],
    input_root: Optional[Path],
) -> Tuple[Path, bool, int, Optional[str]]:
    """处理单个视频（用于多线程）"""
    output_dir = compute_output_dir(video_path, base_output_dir, input_root)
    success, frame_count, err = extract_frames_single(video_path, output_dir, config)
    return video_path, success, frame_count, err


# ============================================================================
# 进度显示
# ============================================================================

class ProgressMonitor:
    """进度监控器"""
    
    def __init__(
        self,
        total_videos: int,
        total_expected_frames: int,
        stats: ProcessingStats,
        lock: threading.Lock,
        show_progress: bool = True,
    ):
        self.total_videos = total_videos
        self.total_expected_frames = total_expected_frames
        self.stats = stats
        self.lock = lock
        self.show_progress = show_progress
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
    
    def start(self):
        """启动进度监控线程"""
        if self.show_progress:
            self.thread = threading.Thread(target=self._progress_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        """停止进度监控"""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.show_progress:
            print()  # 换行，避免进度行覆盖后续输出
    
    def _progress_loop(self):
        """进度更新循环"""
        while not self.stop_event.is_set():
            # 在锁内读取所有统计信息，确保数据一致性
            # 注意：必须使用同一个锁，确保与更新操作的互斥
            with self.lock:
                dv = self.stats.done_videos
                se = self.stats.success_count
                fe = self.stats.failed_count
                df = self.stats.done_expected_frames
                te = self.total_expected_frames
            
            # 在锁外进行格式化输出，避免长时间持有锁
            # 这样可以减少锁竞争，提高性能
            if te > 0 and df >= 0:
                ratio = min(1.0, max(0.0, df / te))  # 确保 ratio 在 [0, 1] 范围内
                bar = self._format_progress_bar(ratio)
                msg = f"\r⏳ 进度 {bar} {ratio*100:6.2f}%  视频 {dv}/{self.total_videos}  成功 {se}  失败 {fe}"
            else:
                # 如果没有预期帧数，只显示视频进度
                msg = f"\r⏳ 进度  视频 {dv}/{self.total_videos}  成功 {se}  失败 {fe}"
            
            sys.stdout.write(msg)
            sys.stdout.flush()
            time.sleep(DEFAULT_PROGRESS_UPDATE_INTERVAL)
    
    @staticmethod
    def _format_progress_bar(ratio: float, width: int = DEFAULT_PROGRESS_BAR_WIDTH) -> str:
        """格式化进度条"""
        ratio = max(0.0, min(1.0, ratio))
        filled = int(round(ratio * width))
        return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


# ============================================================================
# 预估和批量处理
# ============================================================================

def estimate_expected_frames(
    videos: List[Path],
    fps: float,
    max_workers: Optional[int] = None,
) -> Tuple[Dict[Path, Optional[int]], int, int]:
    """
    预估所有视频的预期帧数
    
    Returns:
        (视频到预期帧数的映射, 总预期帧数, 无法获取时长的视频数)
    """
    logger = get_logger(__name__)
    expected_frames_map: Dict[Path, Optional[int]] = {}
    total_expected_frames = 0
    unknown_count = 0
    
    def _probe_expected(video: Path) -> Tuple[Path, Optional[int]]:
        dur = get_video_duration_seconds(video)
        if dur is None or dur <= 0:
            return video, None
        exp = calculate_expected_frames(dur, fps)
        return video, exp
    
    probe_workers = min(
        max_workers or MAX_PROBE_WORKERS,
        os.cpu_count() or 4,
        len(videos)
    )
    
    with ThreadPoolExecutor(max_workers=probe_workers) as executor:
        futures = {executor.submit(_probe_expected, v): v for v in videos}
        for future in as_completed(futures):
            v, exp = future.result()
            expected_frames_map[v] = exp
            if exp is None:
                unknown_count += 1
            else:
                total_expected_frames += exp
    
    return expected_frames_map, total_expected_frames, unknown_count


def process_videos_batch(
    videos: List[Path],
    config: ExtractionConfig,
    base_output_dir: Optional[Path],
    input_root: Optional[Path],
    expected_frames_map: Dict[Path, Optional[int]],
    stats: ProcessingStats,
) -> None:
    """批量处理视频"""
    logger = get_logger(__name__)
    
    max_workers = min(
        config.threads or int(os.cpu_count() * DEFAULT_THREAD_RATIO),
        len(videos)
    )
    logger.info("使用 %d 个线程并发处理视频", max_workers)
    
    lock = threading.Lock()
    progress_monitor = ProgressMonitor(
        len(videos),
        stats.total_expected_frames,
        stats,
        lock,
        config.show_progress,
    )
    progress_monitor.start()
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_video,
                    video, config, base_output_dir, input_root
                ): video
                for video in videos
            }
            
            for future in as_completed(futures):
                video_path, success, _frame_count, err = future.result()
                with lock:
                    stats.done_videos += 1
                    if success:
                        stats.success_count += 1
                        if config.show_progress:
                            exp = expected_frames_map.get(video_path)
                            if isinstance(exp, int) and exp > 0:
                                stats.done_expected_frames += exp
                    else:
                        stats.failed_count += 1
                        stats.failed_videos.append(video_path)
                        if err:
                            stats.failed_errors[video_path] = err
    finally:
        progress_monitor.stop()


# ============================================================================
# 命令行接口
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='从视频文件中提取帧',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('input', type=str, help='输入：视频文件、目录或txt文件路径')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录（默认：与输入视频同目录下的frames文件夹）')
    parser.add_argument('--fps', type=float, default=DEFAULT_FPS,
                       help=f'提取帧率，每秒提取多少帧（默认：{DEFAULT_FPS}）')
    parser.add_argument('--format', type=str, default=DEFAULT_IMAGE_FORMAT, 
                       choices=[fmt.value for fmt in ImageFormat],
                       help=f'输出图片格式（默认：{DEFAULT_IMAGE_FORMAT}）')
    parser.add_argument('--threads', type=int, default=None,
                       help='并发线程数（默认：自动检测CPU核心数）')
    parser.add_argument('--gpu', action='store_true',
                       help='强制使用GPU加速')
    parser.add_argument('--no-gpu', action='store_true',
                       help='强制使用CPU（不使用GPU）')
    parser.add_argument('--quality', type=int, default=DEFAULT_QUALITY, 
                       choices=range(1, 101), metavar='[1-100]',
                       help=f'JPEG质量（默认：{DEFAULT_QUALITY}，仅对jpg格式有效）')
    parser.add_argument('--show-progress', type=int, choices=[0, 1], default=1,
                       help='是否显示整体进度：1=显示(默认)，0=关闭')
    parser.add_argument('--verbose', action='store_true',
                       help='输出逐视频日志（默认不输出，避免刷屏）')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='日志文件目录（默认：skill根目录下的logs/）')
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> ExtractionConfig:
    """从命令行参数创建配置对象"""
    input_path = Path(args.input).resolve()
    
    # 确定GPU模式
    if args.no_gpu:
        gpu_mode = GPUAcceleration.FORCE_CPU
    elif args.gpu:
        gpu_mode = GPUAcceleration.FORCE_GPU
    else:
        gpu_mode = GPUAcceleration.AUTO
    
    # 确定输出目录
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    
    # 确定日志目录
    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else None
    
    return ExtractionConfig(
        input_path=input_path,
        output_dir=output_dir,
        fps=args.fps,
        image_format=args.format,
        quality=args.quality,
        threads=args.threads,
        gpu_mode=gpu_mode,
        show_progress=(args.show_progress == 1),
        verbose=args.verbose,
        log_dir=log_dir,
    )


def print_summary(stats: ProcessingStats, verbose: bool):
    """打印处理总结"""
    print(f"\n{'='*60}")
    print(f"✅ 成功: {stats.success_count}/{stats.total_videos}")
    if stats.failed_videos:
        print(f"❌ 失败: {stats.failed_count}")
        print("\n失败的视频:")
        for video in stats.failed_videos:
            msg = stats.failed_errors.get(video)
            if msg and verbose:
                print(f"  - {video}\n    - {msg}")
            else:
                print(f"  - {video}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    args = parse_arguments()
    
    # 配置日志系统
    config = create_config_from_args(args)
    if config.log_dir:
        configure_logging(log_dir=config.log_dir)
    logger = get_logger(__name__)
    
    # 检查ffmpeg
    if not check_ffmpeg():
        logger.error("未找到 ffmpeg，请先安装 ffmpeg（https://ffmpeg.org/download.html）")
        sys.exit(1)
    
    # 确定GPU使用
    use_gpu, gpu_type = check_gpu_available()
    if config.gpu_mode == GPUAcceleration.FORCE_GPU:
        if not use_gpu:
            logger.warning("请求使用 GPU 但未检测到可用 GPU，将使用 CPU")
            use_gpu = False
        else:
            logger.info("使用 %s GPU 加速进行视频抽帧", gpu_type)
    elif config.gpu_mode == GPUAcceleration.FORCE_CPU:
        use_gpu = False
        logger.info("使用 CPU 进行视频抽帧")
    else:  # AUTO
        if use_gpu:
            logger.info("使用 %s GPU 加速进行视频抽帧", gpu_type)
        else:
            logger.info("使用 CPU 进行视频抽帧")
    
    # 收集视频文件
    videos = collect_video_files(config.input_path, config.verbose)
    if not videos:
        logger.error("未找到任何视频文件，输入路径: %s", config.input_path)
        sys.exit(1)
    
    logger.info("找到 %d 个视频文件，开始批量抽帧", len(videos))
    
    # 确定输出目录和输入根目录
    base_output_dir = config.output_dir
    input_root = config.input_path if config.input_path.is_dir() else None
    
    # 预估总帧数（用于进度显示）
    expected_frames_map: Dict[Path, Optional[int]] = {}
    total_expected_frames = 0
    unknown_duration_count = 0
    
    if config.show_progress:
        logger.info("正在用 ffprobe 预估总帧数（用于进度显示）...")
        expected_frames_map, total_expected_frames, unknown_duration_count = estimate_expected_frames(
            videos, config.fps
        )
        if unknown_duration_count > 0:
            logger.warning("有 %d 个视频无法获取时长，进度帧数会略偏保守", unknown_duration_count)
        logger.info("预计总帧数（近似）：%d", total_expected_frames)
    
    # 初始化统计信息
    stats = ProcessingStats(
        total_videos=len(videos),
        total_expected_frames=total_expected_frames,
    )
    
    # 批量处理
    process_videos_batch(
        videos, config, base_output_dir, input_root,
        expected_frames_map, stats
    )
    
    # 打印总结
    print_summary(stats, config.verbose)
    
    sys.exit(0 if stats.success_count == stats.total_videos else 1)


if __name__ == '__main__':
    main()
