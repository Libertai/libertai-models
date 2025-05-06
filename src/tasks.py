import aiohttp

from src.config import config
from src.interfaces.usage import UsageFullData


async def report_usage_event_task(usage: UsageFullData):
    print(f"Collecting usage {usage}")
    try:
        async with aiohttp.ClientSession() as session:
            session.headers["x-admin-token"] = config.BACKEND_SECRET_TOKEN
            path = "api-keys/admin/usage"
            async with session.post(f"{config.BACKEND_API_URL}/{path}", json=usage.model_dump_json()) as response:
                if response.status != 200:
                    print(f"Error reporting usage: {response.status}")

    except Exception as e:
        print(f"Exception occurred during usage report {str(e)}")
