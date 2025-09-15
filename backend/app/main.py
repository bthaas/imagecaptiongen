"""
FastAPI main application entry point
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from datetime import datetime, timezone
import logging
import sys
import psutil
import os
import uuid
import base64
import binascii
import time
from .models.api_models import HealthResponse, ErrorResponse, ErrorCode, CaptionRequest, CaptionResponse
from .middleware.error_handler import ErrorHandlerMiddleware
from .services.caption_service import CaptionService, CaptionServiceError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Initialize caption service
caption_service = CaptionService()

app = FastAPI(
    title="AI Image Caption Generator",
    description="Generate captions for images using CNN + LSTM architecture",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-production-domain.com"],  # TODO: Update with your production frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add custom error handling middleware
app.add_middleware(ErrorHandlerMiddleware)

# Custom exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with our standard error format"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    error_response = ErrorResponse(
        error=ErrorCode.VALIDATION_ERROR if exc.status_code == 404 else ErrorCode.INTERNAL_ERROR,
        message=str(exc.detail),
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    error_response = ErrorResponse(
        error=ErrorCode.VALIDATION_ERROR,
        message=f"Validation error: {str(exc)}",
        request_id=request_id,
        details={"validation_errors": exc.errors()}
    )
    
    return JSONResponse(
        status_code=422,
        content=error_response.model_dump()
    )

@app.get("/")
async def root():
    """Root endpoint returning basic API information"""
    return {
        "message": "AI Image Caption Generator API",
        "version": "1.0.0",
        "docs": "/api/docs"
    }

@app.options("/api/v1/health")
async def health_options():
    """Handle CORS preflight requests for health endpoint"""
    return JSONResponse(content={}, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "*"
    })

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint with system status information
    Returns system health, memory usage, and service status
    """
    try:
        # Get system information
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check if TensorFlow can be imported (model availability check)
        model_status = "unknown"
        try:
            import tensorflow as tf
            model_status = "available"
        except ImportError:
            model_status = "unavailable"
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
            system_info={
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            },
            services={
                "api": "healthy",
                "model": model_status
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Health check failed"
        )


@app.post("/api/v1/generate-caption", response_model=CaptionResponse)
async def generate_caption(request: CaptionRequest, http_request: Request):
    """
    Generate caption for an uploaded image using CNN + LSTM architecture.
    
    This endpoint accepts a base64-encoded image and returns a generated caption
    with confidence score and processing metadata.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Add request ID to request state for logging
    http_request.state.request_id = request_id
    
    logger.info(f"Caption generation request started - ID: {request_id}")
    
    try:
        # Validate and decode image data
        image_data = _decode_image_data(request.image_data)
        
        # Log request details (without image data)
        logger.info(f"Request {request_id} - max_length: {request.max_length}, "
                   f"temperature: {request.temperature}, image_size: {len(image_data)} bytes")
        
        # Generate caption using the ML pipeline
        result = caption_service.generate_caption(
            image_data=image_data,
            temperature=request.temperature,
            use_beam_search=False,  # Can be made configurable later
            beam_width=3
        )
        
        if not result.success:
            # Handle ML pipeline errors
            error_code = _map_error_to_code(result.error_message)
            logger.error(f"Request {request_id} failed: {result.error_message}")
            
            error_response = ErrorResponse(
                error=error_code,
                message=result.error_message or "Caption generation failed",
                request_id=request_id
            )
            
            return JSONResponse(
                status_code=500,
                content=error_response.model_dump()
            )
        
        # Create successful response
        response = CaptionResponse(
            caption=result.caption,
            confidence=result.confidence,
            processing_time=result.processing_time,
            image_id=request_id
        )
        
        logger.info(f"Request {request_id} completed successfully in {result.processing_time:.3f}s - "
                   f"Caption: '{result.caption}' (confidence: {result.confidence:.3f})")
        
        return response
        
    except ValueError as e:
        # Handle image decoding errors
        error_msg = f"Invalid image data: {str(e)}"
        logger.error(f"Request {request_id} - {error_msg}")
        
        error_response = ErrorResponse(
            error=ErrorCode.INVALID_IMAGE_FORMAT,
            message=error_msg,
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=400,
            content=error_response.model_dump()
        )
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Request {request_id} - {error_msg}", exc_info=True)
        
        error_response = ErrorResponse(
            error=ErrorCode.INTERNAL_ERROR,
            message="Internal server error occurred",
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump()
        )


@app.get("/api/v1/model-info")
async def get_model_info():
    """
    Get information about the loaded ML models and service status.
    """
    try:
        status = caption_service.get_service_status()
        return {
            "service_status": status,
            "api_version": "1.0.0",
            "supported_formats": ["JPEG", "PNG", "WebP"],
            "max_image_size_mb": 10,
            "max_caption_length": 50
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve model information"
        )


def _decode_image_data(image_data_str: str) -> bytes:
    """
    Decode base64 image data from request.
    
    Args:
        image_data_str: Base64 encoded image string (with or without data URL prefix)
        
    Returns:
        bytes: Decoded image data
        
    Raises:
        ValueError: If image data is invalid or cannot be decoded
    """
    try:
        # Handle data URL format (data:image/jpeg;base64,...)
        if image_data_str.startswith('data:'):
            if ';base64,' not in image_data_str:
                raise ValueError("Invalid data URL format - missing base64 encoding")
            image_data_str = image_data_str.split(';base64,')[1]
        
        # Decode base64 data
        image_data = base64.b64decode(image_data_str)
        
        if len(image_data) == 0:
            raise ValueError("Empty image data")
            
        # Basic validation - check for common image file signatures
        if not _is_valid_image_signature(image_data):
            raise ValueError("Invalid image format - not a valid image file")
            
        return image_data
        
    except base64.binascii.Error as e:
        raise ValueError(f"Invalid base64 encoding: {str(e)}")
    except Exception as e:
        raise ValueError(f"Could not decode image data: {str(e)}")


def _is_valid_image_signature(data: bytes) -> bool:
    """
    Check if data starts with a valid image file signature.
    
    Args:
        data: Image data bytes
        
    Returns:
        bool: True if valid image signature found
    """
    if len(data) < 4:
        return False
    
    # Check for common image signatures
    signatures = [
        b'\xff\xd8\xff',  # JPEG
        b'\x89PNG\r\n\x1a\n',  # PNG
        b'RIFF',  # WebP (and other RIFF formats)
        b'GIF87a',  # GIF87a
        b'GIF89a',  # GIF89a
    ]
    
    for sig in signatures:
        if data.startswith(sig):
            return True
    
    # Check WebP more specifically
    if data.startswith(b'RIFF') and len(data) >= 12:
        if data[8:12] == b'WEBP':
            return True
    
    return False


def _map_error_to_code(error_message: str) -> ErrorCode:
    """
    Map error message to appropriate error code.
    
    Args:
        error_message: Error message from ML pipeline
        
    Returns:
        ErrorCode: Appropriate error code
    """
    if not error_message:
        return ErrorCode.INTERNAL_ERROR
    
    error_lower = error_message.lower()
    
    if any(keyword in error_lower for keyword in ['format', 'invalid image', 'decode', 'corrupt']):
        return ErrorCode.INVALID_IMAGE_FORMAT
    elif any(keyword in error_lower for keyword in ['size', 'too large', 'memory']):
        return ErrorCode.IMAGE_TOO_LARGE
    elif any(keyword in error_lower for keyword in ['model', 'tensorflow', 'prediction']):
        return ErrorCode.MODEL_ERROR
    elif any(keyword in error_lower for keyword in ['processing', 'pipeline']):
        return ErrorCode.PROCESSING_ERROR
    else:
        return ErrorCode.INTERNAL_ERROR