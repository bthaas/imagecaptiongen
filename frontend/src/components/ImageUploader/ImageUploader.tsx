import React, { useState, useRef, useCallback } from 'react';
import { Button, Spinner } from '../UI';

interface ImageUploaderProps {
  onImageSelect: (file: File) => void;
  onImageUpload?: (file: File) => Promise<void>;
  maxFileSize?: number; // in bytes
  acceptedFormats?: string[];
  disabled?: boolean;
  className?: string;
}

interface UploadState {
  isDragOver: boolean;
  isUploading: boolean;
  uploadProgress: number;
  error: string | null;
}

const ImageUploader: React.FC<ImageUploaderProps> = ({
  onImageSelect,
  onImageUpload,
  maxFileSize = 10 * 1024 * 1024, // 10MB default
  acceptedFormats = ['image/jpeg', 'image/png', 'image/webp'],
  disabled = false,
  className = ''
}) => {
  const [uploadState, setUploadState] = useState<UploadState>({
    isDragOver: false,
    isUploading: false,
    uploadProgress: 0,
    error: null
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = useCallback((file: File): string | null => {
    // Check file type
    if (!acceptedFormats.includes(file.type)) {
      return `Invalid file format. Supported formats: ${acceptedFormats.map(format => 
        format.split('/')[1].toUpperCase()).join(', ')}`;
    }

    // Check file size
    if (file.size > maxFileSize) {
      const maxSizeMB = (maxFileSize / (1024 * 1024)).toFixed(1);
      return `File size too large. Maximum size: ${maxSizeMB}MB`;
    }

    // Basic validation passed
    return null;
  }, [acceptedFormats, maxFileSize]);

  const handleFileSelect = useCallback(async (file: File) => {
    setUploadState(prev => ({ ...prev, error: null, uploadProgress: 0 }));

    // Basic file validation
    const validationError = validateFile(file);
    if (validationError) {
      setUploadState(prev => ({ ...prev, error: validationError }));
      return;
    }

    // Additional async validation for image integrity
    try {
      await new Promise<void>((resolve, reject) => {
        const img = new Image();
        const url = URL.createObjectURL(file);
        
        img.onload = () => {
          URL.revokeObjectURL(url);
          resolve();
        };
        
        img.onerror = () => {
          URL.revokeObjectURL(url);
          reject(new Error('Invalid image file or corrupted data'));
        };
        
        img.src = url;
      });
    } catch (error) {
      setUploadState(prev => ({ 
        ...prev, 
        error: error instanceof Error ? error.message : 'Invalid image file' 
      }));
      return;
    }

    // Call the onImageSelect callback first
    onImageSelect(file);

    // If onImageUpload is provided, handle the upload and caption generation process
    if (onImageUpload) {
      setUploadState(prev => ({ ...prev, isUploading: true, uploadProgress: 0 }));
      
      try {
        // Simulate upload progress (since we don't have real progress from the API)
        const progressInterval = setInterval(() => {
          setUploadState(prev => ({
            ...prev,
            uploadProgress: Math.min(prev.uploadProgress + 10, 90)
          }));
        }, 200);

        await onImageUpload(file);
        
        clearInterval(progressInterval);
        setUploadState(prev => ({ 
          ...prev, 
          isUploading: false, 
          uploadProgress: 100 
        }));
      } catch (error) {
        setUploadState(prev => ({
          ...prev,
          isUploading: false,
          uploadProgress: 0,
          error: error instanceof Error ? error.message : 'Caption generation failed'
        }));
      }
    }
  }, [onImageSelect, onImageUpload, validateFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) {
      setUploadState(prev => ({ ...prev, isDragOver: true }));
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setUploadState(prev => ({ ...prev, isDragOver: false }));
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setUploadState(prev => ({ ...prev, isDragOver: false }));

    if (disabled) return;

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]); // Only handle the first file
    }
  }, [disabled, handleFileSelect]);

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  const handleBrowseClick = useCallback(() => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.click();
    }
  }, [disabled]);

  const clearError = useCallback(() => {
    setUploadState(prev => ({ ...prev, error: null }));
  }, []);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const { isDragOver, isUploading, uploadProgress, error } = uploadState;

  const containerClasses = [
    'image-uploader',
    isDragOver ? 'image-uploader--drag-over' : '',
    disabled ? 'image-uploader--disabled' : '',
    isUploading ? 'image-uploader--uploading' : '',
    error ? 'image-uploader--error' : '',
    className
  ].filter(Boolean).join(' ');

  return (
    <div className={containerClasses}>
      <div
        className="image-uploader__drop-zone"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleBrowseClick}
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Upload image"
        onKeyDown={(e) => {
          if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
            e.preventDefault();
            handleBrowseClick();
          }
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={acceptedFormats.join(',')}
          onChange={handleFileInputChange}
          className="image-uploader__file-input"
          disabled={disabled}
          aria-hidden="true"
        />

        <div className="image-uploader__content">
          {isUploading ? (
            <div className="image-uploader__uploading">
              <Spinner size="lg" className="mb-3" />
              <p className="image-uploader__uploading-text">
                Uploading... {uploadProgress}%
              </p>
              <div className="image-uploader__progress-bar">
                <div 
                  className="image-uploader__progress-fill"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          ) : (
            <>
              <div className="image-uploader__icon">
                <svg
                  width="48"
                  height="48"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                  <circle cx="8.5" cy="8.5" r="1.5" />
                  <polyline points="21,15 16,10 5,21" />
                </svg>
              </div>
              
              <div className="image-uploader__text">
                <p className="image-uploader__primary-text">
                  {isDragOver 
                    ? 'Drop your image here' 
                    : 'Drag and drop an image here, or click to browse'
                  }
                </p>
                <p className="image-uploader__secondary-text">
                  Supports {acceptedFormats.map(format => 
                    format.split('/')[1].toUpperCase()).join(', ')} • 
                  Max size: {formatFileSize(maxFileSize)}
                </p>
              </div>

              <Button
                variant="primary"
                size="md"
                disabled={disabled}
                className="image-uploader__browse-button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleBrowseClick();
                }}
              >
                Browse Files
              </Button>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="image-uploader__error">
          <div className="image-uploader__error-content">
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="image-uploader__error-icon"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="15" y1="9" x2="9" y2="15" />
              <line x1="9" y1="9" x2="15" y2="15" />
            </svg>
            <span className="image-uploader__error-text">{error}</span>
            <button
              type="button"
              className="image-uploader__error-close"
              onClick={clearError}
              aria-label="Clear error"
            >
              ×
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUploader;