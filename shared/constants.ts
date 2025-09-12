/**
 * Shared constants for the application
 */

export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const SUPPORTED_IMAGE_FORMATS = ['image/jpeg', 'image/png', 'image/webp'];

export const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

export const API_ENDPOINTS = {
  GENERATE_CAPTION: '/api/v1/generate-caption',
  HEALTH: '/api/v1/health',
  MODEL_INFO: '/api/v1/model-info'
} as const;