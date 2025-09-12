// Shared types for the frontend application

export interface CaptionRequest {
  image_data: string;
  max_length?: number;
  temperature?: number;
}

export interface CaptionResponse {
  caption: string;
  confidence: number;
  processing_time: number;
  image_id: string;
  timestamp?: string;
}

export interface ErrorResponse {
  error: string;
  message: string;
  timestamp: string;
  request_id?: string;
  details?: Record<string, any>;
}

export interface ImageMetadata {
  width: number;
  height: number;
  format: string;
  size_bytes: number;
  upload_timestamp: string;
}

export interface AppState {
  selectedImage: File | null;
  imagePreview: string | null;
  caption: string | null;
  isLoading: boolean;
  error: string | null;
  confidence: number | null;
  processingTime: number | null;
  requestId?: string | null;
}

// API Health response
export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  system_info: Record<string, any>;
  services: Record<string, string>;
}

// Model info response
export interface ModelInfoResponse {
  service_status: Record<string, any>;
  api_version: string;
  supported_formats: string[];
  max_image_size_mb: number;
  max_caption_length: number;
}

// Error types for better error handling
export enum ErrorType {
  NETWORK_ERROR = 'NETWORK_ERROR',
  TIMEOUT_ERROR = 'TIMEOUT_ERROR',
  API_ERROR = 'API_ERROR',
  VALIDATION_ERROR = 'VALIDATION_ERROR',
  UNKNOWN_ERROR = 'UNKNOWN_ERROR'
}

export interface AppError {
  type: ErrorType;
  message: string;
  statusCode?: number;
  errorCode?: string;
  requestId?: string;
  retryable?: boolean;
}