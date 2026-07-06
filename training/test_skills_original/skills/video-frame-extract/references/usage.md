# 使用指南

本文档提供 video-frame-extract 技能的详细使用说明、参数参考和示例。

## 目录

- [基本用法](#基本用法)
- [高级选项](#高级选项)
- [输入方式](#输入方式)
- [输出结构](#输出结构)
- [参数参考](#参数参考)
- [txt文件列表格式](#txt文件列表格式)
- [性能优化建议](#性能优化建议)

## 基本用法

### 提取单个视频

```bash
# 提取单个视频，每秒1帧，输出为jpg格式
python scripts/extract_frames.py video.mp4

# 指定输出目录
python scripts/extract_frames.py video.mp4 --output-dir ./output_frames
```

### 提取目录中所有视频

```bash
# 提取目录中所有视频
python scripts/extract_frames.py /path/to/videos/ --output-dir ./all_frames
```

### 从txt文件读取视频列表

```bash
# 从txt文件读取视频列表并提取
python scripts/extract_frames.py video_list.txt
```

## 高级选项

### 帧率控制

```bash
# 指定帧率（每秒提取2帧）
python scripts/extract_frames.py video.mp4 --fps 2.0

# 提取所有帧（fps设为视频原始帧率，需要根据视频调整）
python scripts/extract_frames.py video.mp4 --fps 30.0
```

### 输出格式

```bash
# 输出PNG格式（无损）
python scripts/extract_frames.py video.mp4 --format png

# 输出JPG格式（默认，有损压缩但文件更小）
python scripts/extract_frames.py video.mp4 --format jpg
```

### 质量设置

```bash
# 指定JPEG质量（1-100，默认80）
# 质量越高，文件越大
python scripts/extract_frames.py video.mp4 --quality 95  # 高质量
python scripts/extract_frames.py video.mp4 --quality 80  # 默认质量
python scripts/extract_frames.py video.mp4 --quality 60  # 较低质量，文件更小
```

### 并发控制

```bash
# 使用4个线程并发处理
python scripts/extract_frames.py /path/to/videos/ --threads 4

# 使用所有CPU核心（默认是核心数的一半）
python scripts/extract_frames.py /path/to/videos/ --threads 8
```

### GPU加速

```bash
# 强制使用GPU（如果可用）
python scripts/extract_frames.py video.mp4 --gpu

# 强制使用CPU（不使用GPU）
python scripts/extract_frames.py video.mp4 --no-gpu

# 自动检测（默认行为）
python scripts/extract_frames.py video.mp4
```

### 进度和日志

```bash
# 关闭整体进度显示
python scripts/extract_frames.py /path/to/videos/ --show-progress 0

# 显示逐视频详细日志
python scripts/extract_frames.py /path/to/videos/ --verbose

# 指定日志文件目录
python scripts/extract_frames.py /path/to/videos/ --log-dir /path/to/logs
```

### 组合使用

```bash
# 完整示例：提取目录中所有视频，使用GPU，5fps，PNG格式，4线程，详细日志
python scripts/extract_frames.py /path/to/videos/ \
    --output-dir ./output_frames \
    --fps 5 \
    --format png \
    --threads 4 \
    --gpu \
    --verbose \
    --log-dir ./logs
```

## 输入方式

### 1. 单个视频文件

直接提供视频文件路径：

```bash
python scripts/extract_frames.py /path/to/video.mp4
```

### 2. 视频目录

自动递归查找目录中的所有视频文件：

```bash
python scripts/extract_frames.py /path/to/videos/
```

支持的视频格式：mp4, avi, mov, mkv, flv, wmv, webm, m4v, 3gp, ts, mts

### 3. txt文件列表

txt文件中每行一个视频路径，支持注释行（以#开头）：

创建 `video_list.txt` 文件：

```
# 这是注释行
/path/to/video1.mp4
/path/to/video2.avi
/path/to/video3.mov
# 可以包含注释
/path/to/video4.mkv
```

然后运行：

```bash
python scripts/extract_frames.py video_list.txt
```

**注意事项**：
- 空行会被忽略
- 以 `#` 开头的行被视为注释
- 不存在的路径会被跳过（verbose模式下会显示警告）

## 输出结构

### 默认输出位置

未指定 `--output-dir` 时，提取的帧保存在与视频文件同目录下的 `frames/<视频名>/` 文件夹中。

**示例**：
- 输入：`data/video.mp4`
- 输出：`data/frames/video/video_frame_000001.jpg`, `video_frame_000002.jpg`, ...

### 指定输出目录

使用 `--output-dir` 时，输出结构取决于输入类型：

**输入是目录时**：保留相对路径结构，避免同名视频冲突
- 输入：`/videos/subdir/video.mp4`
- 输出：`<output_dir>/subdir/video/video_frame_000001.jpg`

**输入是txt文件或无法确定相对关系时**：使用 `<视频名>__<hash>/` 确保唯一
- 输入：`/videos/video.mp4`（从txt读取）
- 输出：`<output_dir>/video__abc12345/video_frame_000001.jpg`

### 文件命名格式

帧文件命名格式：`<视频名>_frame_%06d.<格式>`

- `%06d` 表示6位数字，从000001开始递增
- 格式为 `jpg` 或 `png`（取决于 `--format` 参数）

## 参数参考

`scripts/extract_frames.py` 支持以下参数：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `input` | 字符串 | 输入：视频文件、目录或txt文件路径 | 必需 |
| `--output-dir` | 字符串 | 输出目录 | 与输入视频同目录下的frames文件夹 |
| `--fps` | 浮点数 | 提取帧率（每秒提取多少帧） | 1.0 |
| `--format` | 字符串 | 输出图片格式（jpg/png） | jpg |
| `--threads` | 整数 | 并发线程数 | 自动检测CPU核心数的一半 |
| `--gpu` | 标志 | 强制使用GPU加速 | 自动检测 |
| `--no-gpu` | 标志 | 强制使用CPU | 自动检测 |
| `--quality` | 整数 | JPEG质量（1-100，仅对jpg有效） | 80 |
| `--show-progress` | 整数 | 是否显示整体进度：1=显示(默认)，0=关闭 | 1 |
| `--verbose` | 标志 | 输出逐视频日志（默认不输出） | 关闭 |
| `--log-dir` | 字符串 | 日志文件目录 | skill根目录下的logs/ |

### 参数详细说明

#### `--fps` (提取帧率)

- **类型**: 浮点数
- **范围**: > 0
- **说明**: 控制每秒提取多少帧。值越大，提取的帧越多，处理时间越长，输出文件越多。
- **示例**: 
  - `--fps 1.0`: 每秒1帧（适合快速预览）
  - `--fps 5.0`: 每秒5帧（适合一般分析）
  - `--fps 30.0`: 每秒30帧（接近原始帧率）

#### `--format` (输出格式)

- **类型**: 字符串
- **可选值**: `jpg`, `png`
- **说明**: 
  - `jpg`: 有损压缩，文件小，适合大多数场景
  - `png`: 无损压缩，文件大，适合需要精确像素的场景

#### `--quality` (JPEG质量)

- **类型**: 整数
- **范围**: 1-100
- **说明**: 仅对jpg格式有效。值越大，质量越高，文件越大。
- **推荐值**:
  - 95-100: 高质量，文件较大
  - 80-90: 默认质量，平衡质量和大小
  - 60-80: 较低质量，文件较小

#### `--threads` (并发线程数)

- **类型**: 整数
- **范围**: 1 - CPU核心数
- **说明**: 控制同时处理的视频数量。默认是CPU核心数的一半。
- **建议**:
  - 大量小视频：可以增加线程数
  - 少量大视频：减少线程数避免内存压力

#### `--show-progress` (进度显示)

- **类型**: 整数
- **可选值**: `0` (关闭), `1` (显示)
- **说明**: 控制是否显示实时进度条。进度条显示：
  - 已完成帧数百分比
  - 已完成视频数
  - 成功/失败数量

#### `--verbose` (详细日志)

- **类型**: 标志（无需值）
- **说明**: 启用后，每个视频处理完成后会输出详细日志，包括：
  - 成功提取的帧数
  - 输出目录路径
  - 失败时的错误信息

#### `--log-dir` (日志目录)

- **类型**: 字符串
- **说明**: 指定日志文件的保存目录。日志文件名为 `video_frame_extract.log`，支持滚动（10MB/文件，保留5个历史文件）。

## txt文件列表格式

txt文件列表支持以下特性：

1. **每行一个路径**：可以是绝对路径或相对路径
2. **注释行**：以 `#` 开头的行会被忽略
3. **空行**：空行会被忽略
4. **路径验证**：不存在的路径会被跳过（verbose模式下会显示警告）

**示例文件** (`video_list.txt`):

```
# 这是注释行
# 可以包含说明信息

/path/to/video1.mp4
/path/to/video2.avi

# 另一个注释
/path/to/video3.mov
/path/to/video4.mkv
```

**注意事项**：
- 路径中的空格需要用引号包裹，或在脚本中正确处理
- 相对路径是相对于运行脚本时的当前工作目录
- 建议使用绝对路径以避免路径解析问题

## 性能优化建议

### 1. GPU加速

如果有NVIDIA GPU，使用 `--gpu` 可以显著提升处理速度：

```bash
python scripts/extract_frames.py /path/to/videos/ --gpu
```

**要求**：
- NVIDIA GPU: 需要安装NVIDIA驱动和CUDA
- Intel GPU: 需要支持QSV的Intel显卡

### 2. 线程数调优

- **大量小视频**：可以增加线程数（如 `--threads 8`）
- **少量大视频**：减少线程数（如 `--threads 2`）避免内存压力
- **默认值**：CPU核心数的一半，通常是最佳平衡

### 3. 帧率选择

根据需求选择合适的fps，避免提取过多不必要的帧：

- **快速预览**: `--fps 0.5` (每2秒1帧)
- **一般分析**: `--fps 1.0` (每秒1帧，默认)
- **详细分析**: `--fps 5.0` (每秒5帧)
- **完整提取**: `--fps 30.0` (接近原始帧率)

### 4. 格式选择

- **JPG**: 文件小，处理快，适合大多数场景
- **PNG**: 质量高，文件大，适合需要精确像素的场景

### 5. 输出目录规划

- 使用 `--output-dir` 统一管理输出，避免分散在各处
- 对于大量视频，建议使用SSD存储以提高I/O性能

### 6. 批量处理建议

- 对于超大量视频（>1000个），考虑分批处理
- 使用 `--show-progress 0` 和 `--verbose` 组合，将日志重定向到文件进行分析

## 常见问题

### Q: 为什么进度条一直显示0%？

A: 可能的原因：
1. 所有视频都无法获取时长（进度基于预期帧数计算）
2. 视频处理速度很快，进度更新不及时
3. 使用 `--show-progress 0` 关闭了进度显示

### Q: GPU加速不工作怎么办？

A: 
1. 检查GPU驱动是否正确安装：`nvidia-smi` (NVIDIA) 或检查 `/dev/dri/renderD128` (Intel)
2. 使用 `--no-gpu` 强制使用CPU
3. 脚本会自动回退到CPU，检查日志确认

### Q: 如何处理大量视频？

A:
1. 使用 `--threads` 调整并发数
2. 使用 `--output-dir` 统一输出目录
3. 使用 `--show-progress 0` 减少输出
4. 考虑分批处理

### Q: 输出文件太多怎么办？

A:
1. 降低 `--fps` 值，减少提取的帧数
2. 使用 `--format jpg` 和较低的 `--quality` 值，减小文件大小
3. 考虑只处理关键片段
