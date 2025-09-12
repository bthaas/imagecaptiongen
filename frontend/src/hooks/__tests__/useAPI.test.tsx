/**
 * Integration tests for useAPI hook
 */

import React from 'react';
import { renderHook, act } from '@testing-library/react';
import { AppProvider } from '../../context/AppContext';
import { useAPI } from '../useAPI';
import * as apiClient from '../../services/apiClient';

// Mock the API client
jest.mock('../../services/apiClient');
const mockApiClient = apiClient as jest.Mocked<typeof apiClient>;

// Test wrapper component
const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <AppProvider>{children}</AppProvider>
);

describe('useAPI Hook', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('generateCaption', () => {
    it('should successfully generate caption and update state', async () => {
      const mockFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
      const mockResponse = {
        caption: 'A beautiful landscape',
        confidence: 0.85,
        processing_time: 2.34,
        image_id: 'img_123456789',
        timestamp: '2024-01-15T10:30:00Z'
      };

      mockApiClient.generateCaption.mockResolvedValueOnce(mockResponse);

      const { result } = renderHook(() => useAPI(), { wrapper: TestWrapper });

      let success: boolean;
      await act(async () => {
        success = await result.current.generateCaption(mockFile, {
          maxLength: 20,
          temperature: 1.0
        });
      });

      expect(success!).toBe(true);
      expect(mockApiClient.generateCaption).toHaveBeenCalledWith(mockFile, {
        maxLength: 20,
        temperature: 1.0
      });
    });

    it('should handle API errors and update error state', async () => {
      const mockFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
      const mockError = new apiClient.APIError(
        'Invalid image format',
        400,
        'INVALID_IMAGE_FORMAT',
        'req_123'
      );

      mockApiClient.generateCaption.mockRejectedValueOnce(mockError);

      const { result } = renderHook(() => useAPI(), { wrapper: TestWrapper });

      let success: boolean;
      await act(async () => {
        success = await result.current.generateCaption(mockFile);
      });

      expect(success!).toBe(false);
      expect(mockApiClient.generateCaption).toHaveBeenCalledWith(mockFile, {});
    });

    it('should handle network errors', async () => {
      const mockFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
      const mockError = new apiClient.NetworkError('Network connection failed');

      mockApiClient.generateCaption.mockRejectedValueOnce(mockError);

      const { result } = renderHook(() => useAPI(), { wrapper: TestWrapper });

      let success: boolean;
      await act(async () => {
        success = await result.current.generateCaption(mockFile);
      });

      expect(success!).toBe(false);
    });

    it('should handle timeout errors', async () => {
      const mockFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
      const mockError = new apiClient.TimeoutError('Request timeout');

      mockApiClient.generateCaption.mockRejectedValueOnce(mockError);

      const { result } = renderHook(() => useAPI(), { wrapper: TestWrapper });

      let success: boolean;
      await act(async () => {
        success = await result.current.generateCaption(mockFile);
      });

      expect(success!).toBe(false);
    });
  });

  describe('checkHealth', () => {
    it('should successfully check health', async () => {
      const mockResponse = {
        status: 'healthy',
        timestamp: '2024-01-15T10:30:00Z',
        version: '1.0.0',
        system_info: {},
        services: {}
      };

      mockApiClient.checkAPIHealth.mockResolvedValueOnce(mockResponse);

      const { result } = renderHook(() => useAPI(), { wrapper: TestWrapper });

      let isHealthy: boolean;
      await act(async () => {
        isHealthy = await result.current.checkHealth();
      });

      expect(isHealthy!).toBe(true);
      expect(mockApiClient.checkAPIHealth).toHaveBeenCalled();
    });

    it('should handle health check failure', async () => {
      mockApiClient.checkAPIHealth.mockRejectedValueOnce(new Error('Service unavailable'));

      const { result } = renderHook(() => useAPI(), { wrapper: TestWrapper });

      let isHealthy: boolean;
      await act(async () => {
        isHealthy = await result.current.checkHealth();
      });

      expect(isHealthy!).toBe(false);
    });
  });

  describe('getModelInfo', () => {
    it('should successfully get model info', async () => {
      const mockResponse = {
        service_status: { model_loaded: true },
        api_version: '1.0.0',
        supported_formats: ['JPEG', 'PNG', 'WebP'],
        max_image_size_mb: 10,
        max_caption_length: 50
      };

      mockApiClient.getModelInfo.mockResolvedValueOnce(mockResponse);

      const { result } = renderHook(() => useAPI(), { wrapper: TestWrapper });

      let modelInfo: any;
      await act(async () => {
        modelInfo = await result.current.getModelInfo();
      });

      expect(modelInfo).toEqual(mockResponse);
      expect(mockApiClient.getModelInfo).toHaveBeenCalled();
    });

    it('should handle model info failure', async () => {
      mockApiClient.getModelInfo.mockRejectedValueOnce(new Error('Model not available'));

      const { result } = renderHook(() => useAPI(), { wrapper: TestWrapper });

      let modelInfo: any;
      await act(async () => {
        modelInfo = await result.current.getModelInfo();
      });

      expect(modelInfo).toBeNull();
    });
  });

  describe('retryCaption', () => {
    it('should retry caption generation with same parameters', async () => {
      const mockFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
      const mockResponse = {
        caption: 'A beautiful landscape',
        confidence: 0.85,
        processing_time: 2.34,
        image_id: 'img_123456789',
        timestamp: '2024-01-15T10:30:00Z'
      };

      mockApiClient.generateCaption.mockResolvedValueOnce(mockResponse);

      const { result } = renderHook(() => useAPI(), { wrapper: TestWrapper });

      let success: boolean;
      await act(async () => {
        success = await result.current.retryCaption(mockFile, {
          maxLength: 15,
          temperature: 0.8
        });
      });

      expect(success!).toBe(true);
      expect(mockApiClient.generateCaption).toHaveBeenCalledWith(mockFile, {
        maxLength: 15,
        temperature: 0.8
      });
    });
  });
});