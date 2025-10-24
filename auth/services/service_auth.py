from datetime import datetime, timedelta, timezone
from uuid import UUID

import bcrypt
import jwt

from auth.repositories.service import ServiceRepository


class ServiceAlreadyExistsError(Exception):
    pass


class ServiceNotFoundError(Exception):
    pass


class InvalidServiceSecretError(Exception):
    pass


class ServiceAuthService:
    def __init__(
        self,
        service_repository: ServiceRepository,
        jwt_secret: str,
        jwt_algorithm: str,
        token_expire_minutes: int,
    ):
        self.service_repository = service_repository
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.token_expire_minutes = token_expire_minutes

    async def register_service(self, client_id: str, client_secret: str) -> dict:
        existing = await self.service_repository.get_service_by_client_id(client_id)
        if existing:
            raise ServiceAlreadyExistsError
        hashed_secret = bcrypt.hashpw(client_secret.encode(), bcrypt.gensalt()).decode()
        return await self.service_repository.upsert_service(client_id, hashed_secret)

    async def authenticate_service(self, client_id: str, client_secret: str) -> str:
        service = await self.service_repository.get_service_by_client_id(client_id)
        if not service:
            raise ServiceNotFoundError
        if not bcrypt.checkpw(
            client_secret.encode(), service["client_secret_hash"].encode()
        ):
            raise InvalidServiceSecretError
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=self.token_expire_minutes
        )
        payload = {"service_id": str(service["service_id"]), "exp": expire}
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    async def get_me(self, service_id: UUID) -> dict:
        service = await self.service_repository.get_service_by_id(service_id)
        if not service:
            raise ServiceNotFoundError
        return {"service_id": service["service_id"]}


def make_service_auth_service(
    service_repository: ServiceRepository,
    jwt_secret: str,
    jwt_algorithm: str,
    token_expire_minutes: int,
) -> ServiceAuthService:
    return ServiceAuthService(
        service_repository=service_repository,
        jwt_secret=jwt_secret,
        jwt_algorithm=jwt_algorithm,
        token_expire_minutes=token_expire_minutes,
    )
