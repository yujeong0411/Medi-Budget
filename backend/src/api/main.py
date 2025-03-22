from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from configs.setttings import APP_NAME, API_PREFIX, CORS_ORIGINS
from .routes import router

# FastAPI 애플리케이션 생성
app = FastAPI(title=APP_NAME)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# API 경로에 라우터 등록
app.include_router(router, prefix=API_PREFIX)

@app.get("/")
def read_root():
    return {"message": f"Welcome to {APP_NAME}"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
