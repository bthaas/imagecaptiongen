/**
 * API Client service for backend communication
 * Handles image upload, caption generation, and error handling with retry logic
 */

import { CaptionRequest, CaptionResponse, ErrorResponse } from '../types';

// Configuration constants
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const DEFAULT_TIMEOUT = 30000; // 30 seconds
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second

// Error types for better error handling
export class APIError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public errorCode?: string,
    public requestId?: string
  ) {
    super(message);
    this.name = 'APIError';
  }
}

export class NetworkError extends Error {
  constructor(message: string, public originalError?: Error) {
    super(message);
    this.name = 'NetworkError';
  }
}

export class TimeoutError extends Error {
  constructor(message: string = 'Request timeout') {
    super(message);
    this.name = 'TimeoutError';
  }
}

// Utility function to convert File to base64
const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result);
      } else {
        reject(new Error('Failed to convert file to base64'));
      }
    };
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
};

// Utility function to create timeout promise
const createTimeoutPromise = (timeout: number): Promise<never> => {
  return new Promise((_, reject) => {
    setTimeout(() => reject(new TimeoutError()), timeout);
  });
};

// Utility function to delay execution
const delay = (ms: number): Promise<void> => {
  return new Promise(resolve => setTimeout(resolve, ms));
};

// Main API client class
export class APIClient {
  private baseURL: string;
  private timeout: number;

  constructor(baseURL: string = API_BASE_URL, timeout: number = DEFAULT_TIMEOUT) {
    this.baseURL = baseURL.replace(/\/$/, ''); // Remove trailing slash
    this.timeout = timeout;
  }

  /**
   * Make HTTP request with timeout and error handling
   */
  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {},
    timeout: number = this.timeout
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const requestOptions: RequestInit = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    };

    try {
      const response = await Promise.race([
        fetch(url, requestOptions),
        createTimeoutPromise(timeout)
      ]);

      if (!response.ok) {
        // Try to parse error response
        let errorData: ErrorResponse | null = null;
        try {
          errorData = await response.json();
        } catch {
          // If JSON parsing fails, use status text
        }

        const errorMessage = errorData?.message || response.statusText || 'Request failed';
        throw new APIError(
          errorMessage,
          response.status,
          errorData?.error,
          errorData?.request_id
        );
      }

      return await response.json();
    } catch (error) {
      if (error instanceof APIError || error instanceof TimeoutError) {
        throw error;
      }

      // Handle network errors
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new NetworkError('Network connection failed. Please check your internet connection.', error);
      }

      throw new NetworkError('An unexpected network error occurred', error as Error);
    }
  }

  /**
   * Make request with retry logic
   */
  private async makeRequestWithRetry<T>(
    endpoint: string,
    options: RequestInit = {},
    maxRetries: number = MAX_RETRIES
  ): Promise<T> {
    let lastError: Error;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await this.makeRequest<T>(endpoint, options);
      } catch (error) {
        lastError = error as Error;

        // Don't retry on client errors (4xx) except for 408 (timeout) and 429 (rate limit)
        if (error instanceof APIError) {
          const shouldRetry = !error.statusCode || 
            error.statusCode >= 500 || 
            error.statusCode === 408 || 
            error.statusCode === 429;
          
          if (!shouldRetry) {
            throw error;
          }
        }

        // Don't retry on the last attempt
        if (attempt === maxRetries) {
          break;
        }

        // Wait before retrying (exponential backoff)
        const delayMs = RETRY_DELAY * Math.pow(2, attempt);
        await delay(delayMs);
      }
    }

    throw lastError!;
  }

  /**
   * Check API health status
   */
  async checkHealth(): Promise<{
    status: string;
    timestamp: string;
    version: string;
    system_info: Record<string, any>;
    services: Record<string, string>;
  }> {
    return this.makeRequest('/api/v1/health', { method: 'GET' });
  }

  /**
   * Get model information
   */
  async getModelInfo(): Promise<{
    service_status: Record<string, any>;
    api_version: string;
    supported_formats: string[];
    max_image_size_mb: number;
    max_caption_length: number;
  }> {
    return this.makeRequest('/api/v1/model-info', { method: 'GET' });
  }

  /**
   * Generate caption for an image
   */
  async generateCaption(
    imageFile: File,
    options: {
      maxLength?: number;
      temperature?: number;
    } = {}
  ): Promise<CaptionResponse> {
    // Validate file before processing
    this.validateImageFile(imageFile);

    // Convert file to base64
    const imageData = await fileToBase64(imageFile);

    // Prepare request payload
    const requestPayload: CaptionRequest = {
      image_data: imageData,
      max_length: options.maxLength || 20,
      temperature: options.temperature || 1.0,
    };

    // Make request with retry logic
    return this.makeRequestWithRetry<CaptionResponse>(
      '/api/v1/generate-caption',
      {
        method: 'POST',
        body: JSON.stringify(requestPayload),
      }
    );
  }

  /**
   * Validate image file before upload
   */
  private validateImageFile(file: File): void {
    // Check file size (10MB limit)
    const maxSizeBytes = 10 * 1024 * 1024;
    if (file.size > maxSizeBytes) {
      throw new APIError(
        `File size ${(file.size / 1024 / 1024).toFixed(1)}MB exceeds maximum allowed size of 10MB`,
        413,
        'FILE_TOO_LARGE'
      );
    }

    // Check file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!allowedTypes.includes(file.type.toLowerCase())) {
      throw new APIError(
        `Unsupported file type: ${file.type}. Supported formats: JPEG, PNG, WebP`,
        400,
        'INVALID_FILE_TYPE'
      );
    }

    // Check if file is empty
    if (file.size === 0) {
      throw new APIError(
        'File is empty',
        400,
        'EMPTY_FILE'
      );
    }
  }
}

// Create default API client instance
export const apiClient = new APIClient();

// Convenience functions for easier usage
export const generateCaption = (
  imageFile: File,
  options?: { maxLength?: number; temperature?: number }
): Promise<CaptionResponse> => {
  return apiClient.generateCaption(imageFile, options);
};

export const checkAPIHealth = (): Promise<any> => {
  return apiClient.checkHealth();
};

export const getModelInfo = (): Promise<any> => {
  return apiClient.getModelInfo();
};