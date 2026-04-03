"""
Allow the app to be executed as a module: python -m app
"""
from app.main import app
from app.core.settings import get_settings
import uvicorn
import logging

if __name__ == "__main__":
    settings = get_settings()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Credit Risk Engine microservice...")
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )
