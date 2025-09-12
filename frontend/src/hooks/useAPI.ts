/**
 * Custom hook for API operations with error handling and retry logic
 */

import { useCallback } from 'react';
import { useAppContext } from '../context/AppContext';
import { generateCaption, checkAPIHealth, getModelInfo, APIError, NetworkError, TimeoutError } from '../services/apiClient';
import { ErrorType, AppError } from '../types';

export const useAPI = () => {
  const { dispatch } = useAppContext();

  /**
   * Convert API errors to AppError format
   */
  const mapErrorToAppError = useCallback((error: Error): AppError => {
    if (error instanceof APIError) {
      return {
        type: ErrorType.API_ERROR,
        message: error.message,
        statusCode: error.statusCode,
        errorCode: error.errorCode,
        requestId: error.requestId,
        retryable: error.statusCode ? error.statusCode >= 500 : false,
      };
    }

    if (error instanceof NetworkError) {
      return {
        type: ErrorType.NETWORK_ERROR,
        message: error.message,
        retryable: true,
      };
    }

    if (error instanceof TimeoutError) {
      return {
        type: ErrorType.TIMEOUT_ERROR,
        message: error.message,
        retryable: true,
      };
    }

    return {
      type: ErrorType.UNKNOWN_ERROR,
      message: error.message || 'An unexpected error occurred',
      retryable: false,
    };
  }, []);

  /**
   * Generate caption for an image with full error handling
   */
  const handleGenerateCaption = useCallback(async (
    imageFile: File,
    options: { maxLength?: number; temperature?: number } = {}
  ): Promise<boolean> => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'CLEAR_ERROR' });

      // Log the request for debugging
      console.log('Generating caption for:', {
        fileName: imageFile.name,
        fileSize: `${(imageFile.size / 1024 / 1024).toFixed(2)}MB`,
        fileType: imageFile.type,
        options
      });

      const response = await generateCaption(imageFile, options);

      // Log successful response
      console.log('Caption generated successfully:', {
        caption: response.caption,
        confidence: response.confidence,
        processingTime: response.processing_time,
        requestId: response.image_id
      });

      dispatch({
        type: 'SET_CAPTION',
        payload: {
          caption: response.caption,
          confidence: response.confidence,
          processingTime: response.processing_time,
          requestId: response.image_id,
        },
      });

      return true;
    } catch (error) {
      console.error('Caption generation failed:', error);
      const appError = mapErrorToAppError(error as Error);
      dispatch({ type: 'SET_ERROR', payload: appError });
      return false;
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [dispatch, mapErrorToAppError]);

  /**
   * Check API health status
   */
  const handleCheckHealth = useCallback(async (): Promise<boolean> => {
    try {
      await checkAPIHealth();
      return true;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }, []);

  /**
   * Get model information
   */
  const handleGetModelInfo = useCallback(async () => {
    try {
      return await getModelInfo();
    } catch (error) {
      console.error('Failed to get model info:', error);
      return null;
    }
  }, []);

  /**
   * Retry caption generation with the same parameters
   */
  const retryCaption = useCallback(async (
    imageFile: File,
    options: { maxLength?: number; temperature?: number } = {}
  ): Promise<boolean> => {
    return handleGenerateCaption(imageFile, options);
  }, [handleGenerateCaption]);

  return {
    generateCaption: handleGenerateCaption,
    checkHealth: handleCheckHealth,
    getModelInfo: handleGetModelInfo,
    retryCaption,
  };
};