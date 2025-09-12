/**
 * Integration tests for API client service
 */

import { APIClient, APIError, NetworkError, TimeoutError } from '../apiClient';

// Mock fetch for testing
global.fetch = jest.fn();
const mockFetch = fetch as jest.MockedFunction<typeof fetch>;

// Mock FileReader for base64 conversion
global.FileReader = jest.fn(() => ({
  readAsDataURL: jest.fn(),
  onload: null,
  onerror: null,
  result: null,
})) as any;

describe('APIClient', () => {
  let apiClient: APIClient;

  beforeEach(() => {
    apiClient = new APIClient('http://localhost:8000', 5000);
    mockFetch.mockClear();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Health Check', () => {
    it('should successfully check API health', async () => {
      const mockResponse = {
        status: 'healthy',
        timestamp: '2024-01-15T10:30:00Z',
        version: '1.0.0',
        system_info: { memory_usage_percent: 45.2 },
        services: { api: 'healthy', model: 'available' }
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      const result = await apiClient.checkHealth();

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/health',
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('should handle health check failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: async () => ({
          error: 'INTERNAL_ERROR',
          message: 'Service unavailable',
        }),
      } as Response);

      await expect(apiClient.checkHealth()).rejects.toThrow(APIError);
    });
  });

  describe('Model Info', () => {
    it('should successfully get model information', async () => {
      const mockResponse = {
        service_status: { model_loaded: true },
        api_version: '1.0.0',
        supported_formats: ['JPEG', 'PNG', 'WebP'],
        max_image_size_mb: 10,
        max_caption_length: 50
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      const result = await apiClient.getModelInfo();

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/model-info',
        expect.objectContaining({
          method: 'GET',
        })
      );
      expect(result).toEqual(mockResponse);
    });
  });

  describe('Caption Generation', () => {
    let mockFile: File;

    beforeEach(() => {
      // Create a mock file
      mockFile = new File(['test image data'], 'test.jpg', { type: 'image/jpeg' });

      // Mock FileReader
      const mockFileReader = {
        readAsDataURL: jest.fn(),
        onload: null,
        onerror: null,
        result: 'data:image/jpeg;base64,dGVzdCBpbWFnZSBkYXRh', // "test image data" in base64
      };

      (global.FileReader as jest.Mock).mockImplementation(() => mockFileReader);

      // Simulate successful file reading
      setTimeout(() => {
        if (mockFileReader.onload) {
          mockFileReader.onload({} as ProgressEvent<FileReader>);
        }
      }, 0);
    });

    it('should successfully generate caption', async () => {
      const mockResponse = {
        caption: 'A beautiful landscape with mountains',
        confidence: 0.85,
        processing_time: 2.34,
        image_id: 'img_123456789',
        timestamp: '2024-01-15T10:30:00Z'
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      const result = await apiClient.generateCaption(mockFile, {
        maxLength: 20,
        temperature: 1.0
      });

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/generate-caption',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
          body: JSON.stringify({
            image_data: 'data:image/jpeg;base64,dGVzdCBpbWFnZSBkYXRh',
            max_length: 20,
            temperature: 1.0,
          }),
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('should handle API errors with proper error mapping', async () => {
      const mockErrorResponse = {
        error: 'INVALID_IMAGE_FORMAT',
        message: 'Unsupported image format',
        request_id: 'req_123',
        timestamp: '2024-01-15T10:30:00Z'
      };

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => mockErrorResponse,
      } as Response);

      await expect(apiClient.generateCaption(mockFile)).rejects.toThrow(APIError);
    });

    it('should retry on server errors', async () => {
      // First call fails with 500
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: async () => ({
          error: 'INTERNAL_ERROR',
          message: 'Temporary server error',
        }),
      } as Response);

      // Second call succeeds
      const mockResponse = {
        caption: 'A beautiful landscape',
        confidence: 0.85,
        processing_time: 2.34,
        image_id: 'img_123456789'
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      const result = await apiClient.generateCaption(mockFile);

      expect(mockFetch).toHaveBeenCalledTimes(2);
      expect(result).toEqual(mockResponse);
    });

    it('should not retry on client errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({
          error: 'VALIDATION_ERROR',
          message: 'Invalid request data',
        }),
      } as Response);

      await expect(apiClient.generateCaption(mockFile)).rejects.toThrow(APIError);
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });
  });

  describe('Network Error Handling', () => {
    it('should handle network connection failures', async () => {
      mockFetch.mockRejectedValueOnce(new TypeError('Failed to fetch'));

      await expect(apiClient.checkHealth()).rejects.toThrow(NetworkError);
    });

    it('should handle timeout errors', async () => {
      // Mock a slow response that exceeds timeout
      mockFetch.mockImplementationOnce(() => 
        new Promise(resolve => setTimeout(resolve, 10000))
      );

      const shortTimeoutClient = new APIClient('http://localhost:8000', 100);
      await expect(shortTimeoutClient.checkHealth()).rejects.toThrow(TimeoutError);
    });
  });

  describe('Error Response Parsing', () => {
    it('should handle malformed error responses', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: async () => {
          throw new Error('Invalid JSON');
        },
      } as Response);

      await expect(apiClient.checkHealth()).rejects.toThrow(APIError);
    });

    it('should use status text when error response is not JSON', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
        json: async () => {
          throw new Error('Not JSON');
        },
      } as Response);

      try {
        await apiClient.checkHealth();
      } catch (error) {
        expect(error).toBeInstanceOf(APIError);
        expect((error as APIError).message).toBe('Not Found');
      }
    });
  });
});