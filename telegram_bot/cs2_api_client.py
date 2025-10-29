import aiohttp


class CS2ApiClient:
    def __init__(self, base_url: str, token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.token = token

    async def _get_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def get_token(self, username: str, password: str) -> str | None:
        url = f"{self.base_url}/token/"
        payload = {"username": username, "password": password}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    print(f"⚠️ TOKEN ERROR {resp.status} for {url}")
                    return None
                data = await resp.json()
                self.token = data.get("access")
                return self.token

    async def search(self, category: str, query: str) -> list[dict]:
        url = f"{self.base_url}/{category}/"
        params = {"name": query}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, params=params, headers=await self._get_headers()
            ) as resp:
                if resp.status != 200:
                    print(f"⚠️ API SEARCH ERROR {resp.status} for {url}")
                    return []
                return await resp.json()

    async def forecast(self, payload: dict) -> dict | None:
        url = f"{self.base_url}/forecast/"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=await self._get_headers()
            ) as resp:
                if resp.status != 200:
                    print(f"⚠️ FORECAST ERROR {resp.status}")
                    return None
                return await resp.json()
