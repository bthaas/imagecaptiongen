from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum
from datetime import datetime

class ErrorCode(str, Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    INVALID_IMAGE_FORMAT = "INVALID_IMAGE_FORMAT"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    MODEL_ERROR = "MODEL_ERROR"
    PROCESSING_ERROR = "PROCESSING_ERROR"

class ErrorResponse(BaseModel):
    error: ErrorCode
    message: str
    request_id: str
    details: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    system_info: Dict[str, Any]
    services: Dict[str, str]

class CaptionRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    max_length: int = Field(20, description="Maximum length of the generated caption")
    temperature: float = Field(1.0, description="Sampling temperature for generation")

class CaptionResponse(BaseModel):
    caption: str
    confidence: float
    processing_time: float
    image_id: str
