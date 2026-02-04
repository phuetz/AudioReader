"""Gestion d'erreurs structurée pour l'API v2."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorCode(str, Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    TTS_ENGINE_UNAVAILABLE = "TTS_ENGINE_UNAVAILABLE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_ALREADY_CANCELLED = "JOB_ALREADY_CANCELLED"
    GENERATION_FAILED = "GENERATION_FAILED"
    CLONE_FAILED = "CLONE_FAILED"
    PODCAST_ALREADY_RUNNING = "PODCAST_ALREADY_RUNNING"
    PODCAST_NOT_RUNNING = "PODCAST_NOT_RUNNING"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorDetail(BaseModel):
    field: Optional[str] = None
    message: str


class ErrorResponse(BaseModel):
    error: ErrorCode
    message: str
    details: Optional[List[ErrorDetail]] = None


class APIError(HTTPException):
    """Exception HTTP structurée."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int = 400,
        details: Optional[List[ErrorDetail]] = None,
    ):
        body = ErrorResponse(error=code, message=message, details=details)
        super().__init__(status_code=status_code, detail=body.model_dump())
