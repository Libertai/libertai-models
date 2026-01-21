from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.api_keys import router as libertai_router
from src.image_routes import router as image_router
from src.proxy import router as proxy_router

app = FastAPI(title="LibertAI backend service")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(libertai_router)
app.include_router(image_router)
app.include_router(proxy_router)
