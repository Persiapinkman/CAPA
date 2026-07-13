from fastapi import FastAPI
from src.api.routes import router
from src.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION
)

# 注册路由
app.include_router(router, prefix=settings.API_V1_STR)

if __name__ == "__main__":
    import uvicorn
    import os
    from src.core.logging import get_logger

    logger = get_logger(__name__)
    logger.info(f"Starting {settings.PROJECT_NAME} service")

    # 如果配置了日志文件，确保目录存在
    if settings.LOG_TO_FILE and settings.LOG_FILE_PATH:
        try:
            log_dir = os.path.dirname(settings.LOG_FILE_PATH)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                logger.info(f"Created log directory: {log_dir}")
        except OSError as exc:
            logger.warning(
                "Failed to create log directory %s: %s",
                settings.LOG_FILE_PATH,
                exc,
            )

    # 运行服务器
    logger.info(f"Server configuration: host={settings.HOST}, port={settings.PORT}, reload={settings.RELOAD}")

    # 配置日志格式
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = '%(asctime)s - %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_config=log_config
    )
