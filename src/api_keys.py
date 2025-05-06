import aiohttp

from src.config import config


async def get_active_keys() -> set:
    keys = set()

    try:
        async with aiohttp.ClientSession() as session:
            session.headers["x-admin-token"] = config.BACKEND_SECRET_TOKEN
            path = "api-keys/admin/list"
            async with session.get(f"{config.BACKEND_API_URL}/{path}") as response:
                if response.status == 200:
                    data = await response.json()
                    keys.update(data.get("keys"))
                else:
                    print(f"Error fetching accounts: {response.status}")
                    return keys

    except Exception as e:
        print(f"Exception fetching accounts {str(e)}")
        return keys

    return keys


class KeysManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(KeysManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):  # Check if already initialized
            self.keys = set()

    def add_keys(self, keys):
        self.keys.update(keys)

    def key_exists(self, key):
        return key in self.keys

    async def refresh_keys(self):
        self.keys = await get_active_keys()
