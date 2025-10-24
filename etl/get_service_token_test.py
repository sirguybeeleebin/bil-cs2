import httpx
import respx

from etl.get_service_token import get_service_token

AUTH_URL = "http://auth-service/service/token"
SERVICE_ID = "test_id"
SERVICE_SECRET = "test_secret"


@respx.mock
def test_get_service_token_success():
    token_value = "abc123"
    route = respx.post(AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": token_value})
    )

    with httpx.Client() as client:
        token = get_service_token(client, SERVICE_ID, SERVICE_SECRET, AUTH_URL)

    assert token == token_value
    assert route.called


@respx.mock
def test_get_service_token_no_token():
    route = respx.post(AUTH_URL).mock(
        return_value=httpx.Response(200, json={})  # no access_token
    )

    with httpx.Client() as client:
        token = get_service_token(client, SERVICE_ID, SERVICE_SECRET, AUTH_URL)

    assert token is None
    assert route.called


@respx.mock
def test_get_service_token_http_error():
    route = respx.post(AUTH_URL).mock(
        return_value=httpx.Response(401)  # unauthorized
    )

    with httpx.Client() as client:
        token = get_service_token(client, SERVICE_ID, SERVICE_SECRET, AUTH_URL)

    assert token is None
    assert route.called
