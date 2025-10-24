import logging

import httpx

log = logging.getLogger("etl_worker")


def get_service_token(
    client: httpx.Client,
    service_id: str,
    service_secret: str,
    auth_url: str,
) -> str | None:
    try:
        resp = client.post(
            auth_url,
            json={"client_id": service_id, "client_secret": service_secret},
            timeout=10,
        )
        resp.raise_for_status()
        token = resp.json().get("access_token")
        if not token:
            log.error("Failed to get access_token from auth service")
        return token
    except httpx.HTTPError as e:
        log.error(f"Error getting service token: {e}")
        return None
