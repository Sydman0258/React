from fastapi import FastAPI
from app.routes import process

app = FastAPI()

app.include_router(process.router)

@app.get("/")
def root():
    return {"message": "Image Service Running 🚀"}