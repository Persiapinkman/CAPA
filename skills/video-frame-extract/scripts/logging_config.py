#!/usr/bin/env python3
"""
视频帧提取工具日志配置

集中管理 logging 配置，做到：
- 统一格式：时间戳 / 等级 / 模块 / 消息
- 控制台 + 滚动文件双通道输出
- 支持日志等级控制，便于线上排查

使用方式（在其他模块中）：

    from logging_config import get_logger
    logger = get_logger(__name__)

    logger.info("message")
    logger.error("error message")
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

_CONFIGURED = False
_CUSTOM_LOG_DIR: Optional[Path] = None


def configure_logging(log_dir: Optional[Path] = None) -> None:
    """
    配置日志系统（应在程序启动时尽早调用）。
    
    Args:
        log_dir: 自定义日志目录，None 时使用默认目录
    """
    global _CONFIGURED, _CUSTOM_LOG_DIR
    if _CONFIGURED:
        # 如果已经配置过，需要重置才能应用新的 log_dir
        # 清除现有 handler 并重置状态
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        _CONFIGURED = False
    
    if log_dir is not None:
        _CUSTOM_LOG_DIR = Path(log_dir).expanduser().resolve()
    else:
        _CUSTOM_LOG_DIR = None


def _get_default_log_dir() -> Path:
    """
    默认日志目录：
    - 优先使用 configure_logging() 设置的 _CUSTOM_LOG_DIR
    - 其次使用环境变量 VIDEO_FRAME_EXTRACT_LOG_DIR
    - 否则使用当前文件所在 skill 根目录下的 logs/
    """
    if _CUSTOM_LOG_DIR is not None:
        return _CUSTOM_LOG_DIR
    
    env_dir = os.getenv("VIDEO_FRAME_EXTRACT_LOG_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    # 当前文件: .../scripts/logging_config.py
    scripts_dir = Path(__file__).resolve().parent
    skill_root = scripts_dir.parent  # .../video-frame-extract/
    return (skill_root / "logs").resolve()


def _configure_root_logger(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
) -> None:
    """配置根 logger，仅在第一次调用时执行。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir = log_dir or _get_default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "video_frame_extract.log"

    # 日志格式：时间 等级 进程/线程 模块 行号 消息
    fmt = (
        "%(asctime)s [%(levelname)s] "
        "%(process)d:%(threadName)s "
        "%(name)s:%(lineno)d - %(message)s"
    )
    datefmt = "%Y-%m-%d %H:%M:%S"

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # 控制台输出（一般只看 INFO+）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # 滚动文件输出（DEBUG 级别，保留若干历史日志）
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 根 logger 接收所有级别，具体由 handler 控制
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取带有统一配置的 logger。

    - name 为空时返回 root logger
    - 多次调用会复用同一套 handler，不会重复添加
    """
    _configure_root_logger()
    return logging.getLogger(name)

