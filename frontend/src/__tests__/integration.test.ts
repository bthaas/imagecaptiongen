/**
 * Integration tests for frontend-backend communication
 * Tests the API client logic and error handling without actual network calls
 */

import { APIClient, APIError, NetworkError, TimeoutError } from '../services/apiClient';

// Mock fetch for controlled testing
global.fetch = jest.fn();
const mockFetch = fetch as jest.MockedFunction<typeof fetch>;

// Mock FileReader for base64 conversion
const createMockFileReader = () => ({
  readAsDataURL: jest.fn(),
  onload: null as ((event: ProgressEvent<FileReader>) => void) | null,
  onerror: null as ((event: ProgressEvent<FileReader>) => void) | null,
  result: null as string | null,
});

global.FileReader = jest.fn(() => createMockFileReader()) as any;

describe('Frontend-Backend Integration Logic', () => {
  let apiClient: APIClient;

  // Increase timeout for integration tests
  jest.setTimeout(10000);

  // Helper function to mock FileReader with specific result
  const mockFileReaderWithResult = (result: string) => {
    const mockReader = createMockFileReader();
    mockReader.result = result;
    (global.FileReader as jest.Mock).mockImplementationOnce(() => {
      setTimeout(() => {
        if (mockReader.onload) {
          mockReader.onload({} as ProgressEvent<FileReader>);
        }
      }, 0);
      return mockReader;
    });
  };

  beforeEach(() => {
    apiClient = new APIClient('http://localhost:8000', 5000);
    mockFetch.mockClear();
    jest.clearAllMocks();
  });

  describe('API Client Integration Flow', () => {
    it('should handle complete image upload and caption generation flow', async () => {
      // Mock successful file reading
      const mockImageData = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A8A';
      
      // Mock FileReader behavior
      mockFileReaderWithResult(mockImageData);

      // Mock successful API response
      const mockResponse = {
        caption: 'A red square on a white background',
        confidence: 0.95,
        processing_time: 1.23,
        image_id: 'test-image-123',
        timestamp: '2024-01-15T10:30:00Z'
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      // Create test file
      const testFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });

      // Test the complete flow
      const result = await apiClient.generateCaption(testFile, {
        maxLength: 20,
        temperature: 1.0
      });

      expect(result).toEqual(mockResponse);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/generate-caption',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
          body: JSON.stringify({
            image_data: mockImageData,
            max_length: 20,
            temperature: 1.0,
          }),
        })
      );
    });

    it('should validate file size before upload', async () => {
      // Create oversized file (15MB)
      const largeData = new Array(15 * 1024 * 1024).fill('x').join('');
      const largeFile = new File([largeData], 'large.jpg', { type: 'image/jpeg' });

      await expect(apiClient.generateCaption(largeFile)).rejects.toThrow(APIError);
      expect(mockFetch).not.toHaveBeenCalled();
    });

    it('should validate file type before upload', async () => {
      const invalidFile = new File(['test'], 'test.txt', { type: 'text/plain' });

      await expect(apiClient.generateCaption(invalidFile)).rejects.toThrow(APIError);
      expect(mockFetch).not.toHaveBeenCalled();
    });

    it('should handle empty files', async () => {
      const emptyFile = new File([], 'empty.jpg', { type: 'image/jpeg' });

      await expect(apiClient.generateCaption(emptyFile)).rejects.toThrow(APIError);
      expect(mockFetch).not.toHaveBeenCalled();
    });
  });

  describe('Error Response Integration', () => {
    it('should properly parse and handle API error responses', async () => {
      const mockErrorResponse = {
        error: 'INVALID_IMAGE_FORMAT',
        message: 'Unsupported image format',
        request_id: 'req-123',
        timestamp: '2024-01-15T10:30:00Z'
      };

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => mockErrorResponse,
      } as Response);

      // Mock file reading
      mockFileReaderWithResult('data:image/jpeg;base64,invalid');

      const testFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });

      try {
        await apiClient.generateCaption(testFile);
        fail('Expected an error to be thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(APIError);
        const apiError = error as APIError;
        expect(apiError.message).toBe('Unsupported image format');
        expect(apiError.statusCode).toBe(400);
        expect(apiError.errorCode).toBe('INVALID_IMAGE_FORMAT');
        expect(apiError.requestId).toBe('req-123');
      }
    });

    it('should handle malformed error responses', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: async () => {
          throw new Error('Invalid JSON');
        },
      } as Response);

      // Mock file reading
      mockFileReaderWithResult('data:image/jpeg;base64,test');

      const testFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });

      try {
        await apiClient.generateCaption(testFile);
        fail('Expected an error to be thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(APIError);
        expect((error as APIError).message).toBe('Internal Server Error');
      }
    });
  });

  describe('Retry Logic Integration', () => {
    it('should retry on server errors (5xx)', async () => {
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
        caption: 'A test image',
        confidence: 0.85,
        processing_time: 2.34,
        image_id: 'img-123'
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      // Mock file reading
      mockFileReaderWithResult('data:image/jpeg;base64,test');

      const testFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
      const result = await apiClient.generateCaption(testFile);

      expect(mockFetch).toHaveBeenCalledTimes(2);
      expect(result).toEqual(mockResponse);
    });

    it('should not retry on client errors (4xx)', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({
          error: 'VALIDATION_ERROR',
          message: 'Invalid request data',
        }),
      } as Response);

      // Mock file reading
      mockFileReaderWithResult('data:image/jpeg;base64,test');

      const testFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });

      await expect(apiClient.generateCaption(testFile)).rejects.toThrow(APIError);
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('should retry on rate limit errors (429)', async () => {
      // First call fails with 429
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 429,
        json: async () => ({
          error: 'RATE_LIMIT_EXCEEDED',
          message: 'Too many requests',
        }),
      } as Response);

      // Second call succeeds
      const mockResponse = {
        caption: 'A test image',
        confidence: 0.85,
        processing_time: 2.34,
        image_id: 'img-123'
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      // Mock file reading
      mockFileReaderWithResult('data:image/jpeg;base64,test');

      const testFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
      const result = await apiClient.generateCaption(testFile);

      expect(mockFetch).toHaveBeenCalledTimes(2);
      expect(result).toEqual(mockResponse);
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

  describe('Health Check Integration', () => {
    it('should successfully parse health check response', async () => {
      const mockHealthResponse = {
        status: 'healthy',
        timestamp: '2024-01-15T10:30:00Z',
        version: '1.0.0',
        system_info: {
          memory_usage_percent: 45.2,
          memory_available_gb: 5.87,
          disk_usage_percent: 1.8,
          disk_free_gb: 567.44
        },
        services: {
          api: 'healthy',
          model: 'available'
        }
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockHealthResponse,
      } as Response);

      const result = await apiClient.checkHealth();

      expect(result).toEqual(mockHealthResponse);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/health',
        expect.objectContaining({
          method: 'GET',
        })
      );
    });
  });

  describe('Model Info Integration', () => {
    it('should successfully parse model info response', async () => {
      const mockModelResponse = {
        service_status: { model_loaded: true },
        api_version: '1.0.0',
        supported_formats: ['JPEG', 'PNG', 'WebP'],
        max_image_size_mb: 10,
        max_caption_length: 50
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockModelResponse,
      } as Response);

      const result = await apiClient.getModelInfo();

      expect(result).toEqual(mockModelResponse);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/model-info',
        expect.objectContaining({
          method: 'GET',
        })
      );
    });
  });
});