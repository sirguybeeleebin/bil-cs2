from datetime import datetime, timedelta, timezone
from uuid import UUID

import bcrypt
import jwt

from auth.repositories.service import ServiceRepository
from auth.repositories.user import UserRepository


class AuthService:
    def __init__(
        self,
        user_repository: UserRepository,
        service_repository: ServiceRepository,
        jwt_secret: str,
        jwt_algorithm: str,
        token_expire_minutes: int,
    ):
        self.user_repository: UserRepository = user_repository
        self.service_repository: ServiceRepository = service_repository
        self.jwt_secret: str = jwt_secret
        self.jwt_algorithm: str = jwt_algorithm
        self.token_expire_minutes: int = token_expire_minutes

    async def register_user(self, username: str, password: str) -> dict | None:
        existing = await self.user_repository.get_user_by_username(username)
        if existing:
            return None
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        return await self.user_repository.upsert_user(username, hashed_password)

    async def authenticate_user(self, username: str, password: str) -> str | None:
        user = await self.user_repository.get_user_by_username(username)
        if not user:
            return None
        if not bcrypt.checkpw(password.encode(), user["password"].encode()):
            return None
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=self.token_expire_minutes
        )
        payload = {"user_id": str(user["user_id"]), "exp": expire}
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        return token

    async def get_me(self, user_id: UUID) -> dict | None:
        user = await self.user_repository.get_user_by_id(user_id)
        if not user:
            return None
        return {"user_id": user["user_id"]}

    async def register_service(self, client_id: str, client_secret: str) -> dict | None:
        existing = await self.service_repository.get_service_by_client_id(client_id)
        if existing:
            return None
        hashed_secret = bcrypt.hashpw(client_secret.encode(), bcrypt.gensalt()).decode()
        return await self.service_repository.upsert_service(client_id, hashed_secret)

    async def authenticate_service(
        self, client_id: str, client_secret: str
    ) -> str | None:
        service = await self.service_repository.get_service_by_client_id(client_id)
        if not service:
            return None
        if not bcrypt.checkpw(
            client_secret.encode(), service["client_secret"].encode()
        ):
            return None
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=self.token_expire_minutes
        )
        payload = {"service_id": str(service["service_id"]), "exp": expire}
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        return token


def make_auth_service(
    user_repository: UserRepository,
    service_repository: ServiceRepository,
    jwt_secret: str,
    jwt_algorithm: str,
    token_expire_minutes: int,
) -> AuthService:
    return AuthService(
        user_repository,
        service_repository,
        jwt_secret,
        jwt_algorithm,
        token_expire_minutes,
    )
