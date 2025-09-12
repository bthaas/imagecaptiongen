"""
Custom error handling middleware for FastAPI
"""
import logging
import uuid
from datetime import datetime
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from ..models.api_models import ErrorResponse, ErrorCode

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Custom middleware to handle exceptions and provide consistent error responses
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process the request and handle any exceptions that occur
        """
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        try:
            # Log incoming request
            logger.info(
                f"Request {request_id}: {request.method} {request.url.path}"
            )
            
            response = await call_next(request)
            
            # Check if response is an HTTP error and convert to our format
            if response.status_code >= 400:
                logger.warning(
                    f"Request {request_id}: HTTP error response - {response.status_code}"
                )
                
                # For 404 and other HTTP errors, create our standard error response
                if response.status_code == 404:
                    error_response = ErrorResponse(
                        error=ErrorCode.VALIDATION_ERROR,
                        message="Not Found",
                        request_id=request_id
                    )
                    
                    return JSONResponse(
                        status_code=404,
                        content=error_response.model_dump()
                    )
            
            # Log successful response
            logger.info(
                f"Request {request_id}: Completed with status {response.status_code}"
            )
            
            return response
            
        except (HTTPException, StarletteHTTPException) as exc:
            # Handle FastAPI and Starlette HTTP exceptions
            logger.warning(
                f"Request {request_id}: HTTP exception - {exc.status_code}: {exc.detail}"
            )
            
            error_response = ErrorResponse(
                error=self._map_http_status_to_error_code(exc.status_code),
                message=str(exc.detail),
                request_id=request_id
            )
            
            return JSONResponse(
                status_code=exc.status_code,
                content=error_response.model_dump()
            )
            
        except ValueError as exc:
            # Handle validation errors
            logger.error(
                f"Request {request_id}: Validation error - {str(exc)}"
            )
            
            error_response = ErrorResponse(
                error=ErrorCode.VALIDATION_ERROR,
                message=f"Validation error: {str(exc)}",
                request_id=request_id
            )
            
            return JSONResponse(
                status_code=400,
                content=error_response.model_dump()
            )
            
        except Exception as exc:
            # Handle unexpected errors
            logger.error(
                f"Request {request_id}: Unexpected error - {str(exc)}",
                exc_info=True
            )
            
            error_response = ErrorResponse(
                error=ErrorCode.INTERNAL_ERROR,
                message="An internal server error occurred",
                request_id=request_id,
                details={"error_type": type(exc).__name__} if logger.isEnabledFor(logging.DEBUG) else None
            )
            
            return JSONResponse(
                status_code=500,
                content=error_response.model_dump()
            )
    
    def _map_http_status_to_error_code(self, status_code: int) -> ErrorCode:
        """
        Map HTTP status codes to our custom error codes
        """
        mapping = {
            400: ErrorCode.VALIDATION_ERROR,
            404: ErrorCode.VALIDATION_ERROR,
            413: ErrorCode.IMAGE_TOO_LARGE,
            415: ErrorCode.INVALID_IMAGE_FORMAT,
            422: ErrorCode.VALIDATION_ERROR,
            500: ErrorCode.INTERNAL_ERROR,
        }
        
        return mapping.get(status_code, ErrorCode.INTERNAL_ERROR)