/**
 * Shared TypeScript types for frontend and backend communication
 */

export interface CaptionRequest {
  image_data: string; // Base64 encoded image
  max_length?: number;
  temperature?: number;
}

export interface CaptionResponse {
  caption: string;
  confidence: number;
  processing_time: number;
  image_id: string;
}

export interface ErrorResponse {
  error: string;
  message: string;
  timestamp: string;
  request_id?: string;
}

export interface ImageMetadata {
  width: number;
  height: number;
  format: string;
  size_bytes: number;
  upload_timestamp: string;
}