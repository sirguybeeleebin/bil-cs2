from datetime import datetime, timedelta, timezone
from uuid import UUID

import bcrypt
import jwt

from auth.repositories.user import UserRepository


class UserAlreadyExistsError(Exception):
    pass


class UserNotFoundError(Exception):
    pass


class InvalidUserPasswordError(Exception):
    pass


class AuthService:
    def __init__(
        self,
        user_repository: UserRepository,
        jwt_secret: str,
        jwt_algorithm: str,
        token_expire_minutes: int,
    ):
        self.user_repository = user_repository
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.token_expire_minutes = token_expire_minutes

    async def register_user(self, username: str, password: str) -> dict:
        existing = await self.user_repository.get_user_by_username(username)
        if existing:
            raise UserAlreadyExistsError
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        return await self.user_repository.upsert_user(username, hashed_password)

    async def authenticate_user(self, username: str, password: str) -> str:
        user = await self.user_repository.get_user_by_username(username)
        if not user:
            raise UserNotFoundError
        if not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
            raise InvalidUserPasswordError
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=self.token_expire_minutes
        )
        payload = {"user_id": str(user["user_id"]), "exp": expire}
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    async def get_me(self, user_id: UUID) -> dict:
        user = await self.user_repository.get_user_by_id(user_id)
        if not user:
            raise UserNotFoundError
        return {"user_id": user["user_id"]}


def make_auth_service(
    user_repository: UserRepository,
    jwt_secret: str,
    jwt_algorithm: str,
    token_expire_minutes: int,
) -> AuthService:
    return AuthService(
        user_repository=user_repository,
        jwt_secret=jwt_secret,
        jwt_algorithm=jwt_algorithm,
        token_expire_minutes=token_expire_minutes,
    )
