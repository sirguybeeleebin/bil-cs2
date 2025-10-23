# services/auth.py
import os
from datetime import datetime, timedelta
from typing import Optional

import jwt
import bcrypt

from repositories.user import UserRepository


class AuthService:
    def __init__(
        self,
        user_repo: UserRepository,
        jwt_secret: str = None,
        jwt_algorithm: str = "HS256",
        access_token_expire_minutes: int = 60 * 24,
    ):
        self.user_repo = user_repo
        self.jwt_secret = jwt_secret or os.getenv("JWT_SECRET", "supersecret")
        self.jwt_algorithm = jwt_algorithm
        self.access_token_expire_minutes = access_token_expire_minutes

    async def register(self, username: str, password: str) -> str:
        existing_user = await self.user_repo.get_by_username(username)
        if existing_user:
            raise ValueError("Username already exists")

        hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        user = await self.user_repo.upsert(
            username=username, password_hash=hashed_password.decode("utf-8")
        )

        # Create JWT token after registration
        token = self._create_access_token(user["user_id"])
        return token

    async def login(self, username: str, password: str) -> Optional[str]:
        user = await self.user_repo.get_by_username(username)
        if not user:
            return None

        if not bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
            return None

        return self._create_access_token(user["user_id"])

    def _create_access_token(self, user_id: int) -> str:
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        payload = {"user_id": user_id, "exp": expire}
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        return token
    
    
def make_auth_service(
    user_repository: UserRepository,
    jwt_secret: str = "supersecret",
    jwt_algorithm: str = "HS256",
    access_token_expire_minutes: int = 60 * 24
) -> AuthService:    
    return AuthService(
        user_repo=user_repository,
        jwt_secret=jwt_secret,
        jwt_algorithm=jwt_algorithm,
        access_token_expire_minutes=access_token_expire_minutes,
    )