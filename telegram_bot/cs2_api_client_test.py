# import pytest
# from aiohttp import web

# from telegram_bot.cs2_api_client import CS2ApiClient  # adjust import path


# # -------------------------------
# # Helper to start a test server
# # -------------------------------
# async def start_test_server():
#     async def token_handler(request):
#         data = await request.json()
#         if data.get("username") == "user" and data.get("password") == "pass":
#             return web.json_response({"access": "testtoken"})
#         return web.Response(status=401)

#     async def search_handler(request):
#         name = request.query.get("name")
#         return web.json_response([{"id": 1, "name": name}])

#     async def forecast_handler(request):
#         data = await request.json()
#         return web.json_response({"forecast": "sunny", "payload": data})

#     app = web.Application()
#     app.router.add_post("/token/", token_handler)
#     app.router.add_get("/items/", search_handler)
#     app.router.add_post("/forecast/", forecast_handler)

#     runner = web.AppRunner(app)
#     await runner.setup()
#     site = web.TCPSite(runner, "localhost", 0)
#     await site.start()

#     port = site._server.sockets[0].getsockname()[1]
#     url = f"http://localhost:{port}"
#     return runner, url  # return runner so we can clean up


# # -------------------------------
# # Tests
# # -------------------------------


# @pytest.mark.asyncio
# async def test_get_token():
#     runner, url = await start_test_server()
#     client = CS2ApiClient(url)

#     token = await client.get_token("user", "pass")
#     assert token == "testtoken"
#     assert client.token == "testtoken"

#     token = await client.get_token("user", "wrong")
#     assert token is None

#     await runner.cleanup()


# @pytest.mark.asyncio
# async def test_search():
#     runner, url = await start_test_server()
#     client = CS2ApiClient(url, token="testtoken")

#     results = await client.search("items", "TestItem")
#     assert results == [{"id": 1, "name": "TestItem"}]

#     await runner.cleanup()


# @pytest.mark.asyncio
# async def test_forecast():
#     runner, url = await start_test_server()
#     client = CS2ApiClient(url, token="testtoken")

#     payload = {"temp": 25}
#     result = await client.forecast(payload)
#     assert result["forecast"] == "sunny"
#     assert result["payload"] == payload

#     await runner.cleanup()
