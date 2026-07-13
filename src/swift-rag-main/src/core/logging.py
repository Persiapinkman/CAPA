import logging
import os
import sys
from src.core.config import get_settings

settings = get_settings()


def get_logger(name: str) -> logging.Logger:
    """获取配置好的日志记录器，包含代码位置信息"""
    logger = logging.getLogger(name)

    # 设置日志级别
    log_level = getattr(logging, settings.LOG_LEVEL)
    logger.setLevel(log_level)

    # 如果没有处理器，添加处理器
    if not logger.handlers:
        # 添加控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(settings.LOG_FORMAT)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 如果配置了日志文件，添加文件处理器
        if settings.LOG_TO_FILE and settings.LOG_FILE_PATH:
            try:
                log_dir = os.path.dirname(settings.LOG_FILE_PATH)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

                file_handler = logging.FileHandler(
                    settings.LOG_FILE_PATH, encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except OSError as exc:
                logger.warning(
                    "Failed to initialize file logging at %s: %s. Falling back to console only.",
                    settings.LOG_FILE_PATH,
                    exc,
                )

    return logger
