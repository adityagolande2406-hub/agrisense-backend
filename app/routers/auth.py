"""Auth router — register/login/logout."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from jose import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import User
from app.schemas import LoginRequest, RegisterRequest, TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["Auth"])
logger = logging.getLogger(__name__)
password_hasher = PasswordHasher()


def _hash_password(password: str) -> str:
    return password_hasher.hash(password)


def _verify_password(plain: str, hashed: str) -> bool:
    try:
        return password_hasher.verify(hashed, plain)
    except VerifyMismatchError:
        return False

def _create_token(user_id: int) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": str(user_id), "exp": expire},
        settings.SECRET_KEY,
        algorithm="HS256",
    )


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # Check duplicate email
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered.")
    user = User(
        full_name=body.full_name,
        email=body.email,
        hashed_password=_hash_password(body.password),
        mobile_number=body.mobile_number,
        preferred_language=body.preferred_language,
    )
    db.add(user)
    await db.flush()
    token = _create_token(user.id)
    return TokenResponse(access_token=token, user=UserResponse.model_validate(user))


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user or not _verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = _create_token(user.id)
    return TokenResponse(access_token=token, user=UserResponse.model_validate(user))


@router.post("/logout", status_code=204)
async def logout():
    # Stateless JWT — client discards token
    return None
