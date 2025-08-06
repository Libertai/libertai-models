import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.api_keys import KeysManager
from src.api_keys import router as libertai_router
from src.proxy import router as proxy_router

keys_manager = KeysManager()


async def run_jobs():
    while True:
        await keys_manager.refresh_keys()
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("Starting server...")
    asyncio.create_task(run_jobs())
    yield


app = FastAPI(title="LibertAI backend service", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(libertai_router)
app.include_router(proxy_router)
