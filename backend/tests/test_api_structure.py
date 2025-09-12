"""
Unit tests for core API structure and basic endpoints
"""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json
from unittest.mock import patch, MagicMock

from app.main import app
from app.models.api_models import ErrorCode


@pytest.fixture
def client():
    """Create test client for FastAPI app"""
    return TestClient(app)


class TestRootEndpoint:
    """Test cases for root endpoint"""
    
    def test_root_endpoint_returns_basic_info(self, client):
        """Test that root endpoint returns basic API information"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "AI Image Caption Generator API"
        assert data["version"] == "1.0.0"
        assert data["docs"] == "/api/docs"
    
    def test_root_endpoint_content_type(self, client):
        """Test that root endpoint returns JSON content type"""
        response = client.get("/")
        
        assert response.headers["content-type"] == "application/json"


class TestHealthEndpoint:
    """Test cases for health check endpoint"""
    
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_health_check_success(self, mock_disk, mock_memory, client):
        """Test successful health check response"""
        # Mock system information
        mock_memory.return_value = MagicMock(
            percent=45.2,
            available=8589934592  # 8GB in bytes
        )
        mock_disk.return_value = MagicMock(
            percent=60.5,
            free=107374182400  # 100GB in bytes
        )
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data
        assert "system_info" in data
        assert "services" in data
        
        # Check system info
        system_info = data["system_info"]
        assert system_info["memory_usage_percent"] == 45.2
        assert system_info["memory_available_gb"] == 8.0
        assert system_info["disk_usage_percent"] == 60.5
        assert system_info["disk_free_gb"] == 100.0
        
        # Check services
        services = data["services"]
        assert services["api"] == "healthy"
        assert services["model"] in ["available", "unavailable"]
    
    def test_health_check_response_model(self, client):
        """Test that health check response matches expected model structure"""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate timestamp format
        timestamp = data["timestamp"]
        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        # Validate required fields exist and have correct types
        assert isinstance(data["status"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["system_info"], dict)
        assert isinstance(data["services"], dict)
    
    @patch('psutil.virtual_memory')
    def test_health_check_handles_system_error(self, mock_memory, client):
        """Test health check handles system information errors gracefully"""
        # Mock system error
        mock_memory.side_effect = Exception("System error")
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == ErrorCode.INTERNAL_ERROR
        assert "Health check failed" in data["message"]


class TestCORSConfiguration:
    """Test cases for CORS middleware configuration"""
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in responses"""
        response = client.options("/api/v1/health")
        
        # Check that CORS headers are present
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers
    
    def test_cors_allows_frontend_origin(self, client):
        """Test that CORS allows frontend origins"""
        headers = {"Origin": "http://localhost:3000"}
        response = client.get("/api/v1/health", headers=headers)
        
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


class TestErrorHandling:
    """Test cases for error handling middleware"""
    
    def test_404_error_handling(self, client):
        """Test that 404 errors are handled properly"""
        response = client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404
        data = response.json()
        
        # Check error response structure
        assert "error" in data
        assert "message" in data
        assert "timestamp" in data
        assert "request_id" in data
    
    def test_error_response_format(self, client):
        """Test that error responses follow the standard format"""
        response = client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404
        data = response.json()
        
        # Validate error response structure
        assert data["error"] in [code.value for code in ErrorCode]
        assert isinstance(data["message"], str)
        assert isinstance(data["request_id"], str)
        
        # Validate timestamp format
        timestamp = data["timestamp"]
        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    def test_request_id_generation(self, client):
        """Test that each request gets a unique request ID"""
        response1 = client.get("/nonexistent-endpoint-1")
        response2 = client.get("/nonexistent-endpoint-2")
        
        data1 = response1.json()
        data2 = response2.json()
        
        assert data1["request_id"] != data2["request_id"]


class TestAPIDocumentation:
    """Test cases for API documentation endpoints"""
    
    def test_openapi_docs_accessible(self, client):
        """Test that OpenAPI documentation is accessible"""
        response = client.get("/api/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_docs_accessible(self, client):
        """Test that ReDoc documentation is accessible"""
        response = client.get("/api/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_openapi_schema_accessible(self, client):
        """Test that OpenAPI schema is accessible"""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Validate it's valid JSON
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "AI Image Caption Generator"


class TestMiddlewareIntegration:
    """Test cases for middleware integration"""
    
    def test_request_logging(self, client, caplog):
        """Test that requests are logged properly"""
        with caplog.at_level("INFO"):
            response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        
        # Check that request was logged
        log_messages = [record.message for record in caplog.records]
        request_logs = [msg for msg in log_messages if "GET /api/v1/health" in msg]
        assert len(request_logs) >= 1
    
    def test_error_logging(self, client, caplog):
        """Test that errors are logged properly"""
        with caplog.at_level("WARNING"):
            response = client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404
        
        # Check that error was logged
        log_messages = [record.message for record in caplog.records]
        error_logs = [msg for msg in log_messages if "HTTP error response" in msg or "HTTP exception" in msg]
        assert len(error_logs) >= 1


class TestAPIConfiguration:
    """Test cases for API configuration"""
    
    def test_api_title_and_version(self, client):
        """Test that API has correct title and version in OpenAPI schema"""
        response = client.get("/openapi.json")
        schema = response.json()
        
        assert schema["info"]["title"] == "AI Image Caption Generator"
        assert schema["info"]["version"] == "1.0.0"
        assert "Generate captions for images" in schema["info"]["description"]
    
    def test_api_endpoints_documented(self, client):
        """Test that all endpoints are documented in OpenAPI schema"""
        response = client.get("/openapi.json")
        schema = response.json()
        
        paths = schema["paths"]
        
        # Check that main endpoints are documented
        assert "/" in paths
        assert "/api/v1/health" in paths
        
        # Check HTTP methods
        assert "get" in paths["/"]
        assert "get" in paths["/api/v1/health"]