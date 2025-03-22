import os
from dotenv import load_dotenv

load_dotenv()

# 어플리케이션 설정
APP_NAME = 'MediBudget API'
API_PREFIX = '/api'
DEBUG = os.environ.get("DEBUG", "False").lower() == 'true'

# CORS 설정
CORS_ORIGINS = [
    "http://localhost:3000",  # React 개발 서버
    "http://localhost:8000"  # FastAPI 문서
]


