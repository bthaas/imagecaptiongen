import React, { useState, useCallback } from 'react';
import { Button, Spinner } from '../UI';

interface ImagePreviewProps {
  file: File | null;
  imageUrl?: string | null;
  caption?: string | null;
  isLoading?: boolean;
  confidence?: number | null;
  processingTime?: number | null;
  error?: string | null;
  onRemove?: () => void;
  onRetry?: () => void;
  className?: string;
}

const ImagePreview: React.FC<ImagePreviewProps> = ({
  file,
  imageUrl,
  caption,
  isLoading = false,
  confidence,
  processingTime,
  error,
  onRemove,
  onRetry,
  className = ''
}) => {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // Create preview URL from file
  React.useEffect(() => {
    if (file) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setImageLoaded(false);
      setImageError(false);

      return () => {
        URL.revokeObjectURL(url);
      };
    } else {
      setPreviewUrl(null);
    }
  }, [file]);

  const handleImageLoad = useCallback(() => {
    setImageLoaded(true);
    setImageError(false);
  }, []);

  const handleImageError = useCallback(() => {
    setImageError(true);
    setImageLoaded(false);
  }, []);

  const copyToClipboard = useCallback(async () => {
    if (caption) {
      try {
        await navigator.clipboard.writeText(caption);
        // You could add a toast notification here
      } catch (error) {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = caption;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
      }
    }
  }, [caption]);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatProcessingTime = (time: number): string => {
    if (time === null || time === undefined) return '';
    return time < 1 ? `${(time * 1000).toFixed(0)}ms` : `${time.toFixed(2)}s`;
  };

  if (!file && !imageUrl) {
    return null;
  }

  const displayUrl = imageUrl || previewUrl;
  const containerClasses = [
    'image-preview',
    isLoading ? 'image-preview--loading' : '',
    className
  ].filter(Boolean).join(' ');

  return (
    <div className={containerClasses}>
      <div className="image-preview__container">
        {/* Image Display */}
        <div className="image-preview__image-container">
          {displayUrl && !imageError ? (
            <>
              <img
                src={displayUrl}
                alt="Preview"
                className={`image-preview__image ${imageLoaded ? 'image-preview__image--loaded' : ''}`}
                onLoad={handleImageLoad}
                onError={handleImageError}
              />
              {!imageLoaded && (
                <div className="image-preview__image-loading">
                  <Spinner size="md" />
                </div>
              )}
            </>
          ) : (
            <div className="image-preview__image-error">
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
                <line x1="1" y1="1" x2="23" y2="23" />
              </svg>
              <p>Failed to load image</p>
            </div>
          )}

          {/* Loading Overlay */}
          {isLoading && (
            <div className="image-preview__loading-overlay">
              <div className="image-preview__loading-content">
                <Spinner size="lg" />
                <p className="image-preview__loading-text">
                  Generating caption...
                </p>
              </div>
            </div>
          )}

          {/* Remove Button */}
          {onRemove && (
            <button
              type="button"
              className="image-preview__remove-button"
              onClick={onRemove}
              aria-label="Remove image"
              title="Remove image"
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          )}
        </div>

        {/* Image Metadata */}
        {file && (
          <div className="image-preview__metadata">
            <div className="image-preview__metadata-item">
              <span className="image-preview__metadata-label">File:</span>
              <span className="image-preview__metadata-value">{file.name}</span>
            </div>
            <div className="image-preview__metadata-item">
              <span className="image-preview__metadata-label">Size:</span>
              <span className="image-preview__metadata-value">{formatFileSize(file.size)}</span>
            </div>
            <div className="image-preview__metadata-item">
              <span className="image-preview__metadata-label">Type:</span>
              <span className="image-preview__metadata-value">{file.type}</span>
            </div>
          </div>
        )}

        {/* Caption Display */}
        {(caption || isLoading || error) && (
          <div className="image-preview__caption-section">
            <h3 className="image-preview__caption-title">Generated Caption</h3>
            
            {isLoading ? (
              <div className="image-preview__caption-loading">
                <div className="image-preview__caption-skeleton">
                  <div className="image-preview__skeleton-line image-preview__skeleton-line--long"></div>
                  <div className="image-preview__skeleton-line image-preview__skeleton-line--medium"></div>
                  <div className="image-preview__skeleton-line image-preview__skeleton-line--short"></div>
                </div>
              </div>
            ) : error ? (
              <div className="image-preview__caption-error">
                <div className="image-preview__error-icon">
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
                    <circle cx="12" cy="12" r="10" />
                    <line x1="15" y1="9" x2="9" y2="15" />
                    <line x1="9" y1="9" x2="15" y2="15" />
                  </svg>
                </div>
                <div className="image-preview__error-content">
                  <h4 className="image-preview__error-title">Caption Generation Failed</h4>
                  <p className="image-preview__error-message">{error}</p>
                  {onRetry && (
                    <Button
                      variant="primary"
                      size="sm"
                      onClick={onRetry}
                      className="image-preview__error-retry-button"
                    >
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className="mr-1"
                      >
                        <polyline points="23 4 23 10 17 10" />
                        <polyline points="1 20 1 14 7 14" />
                        <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15" />
                      </svg>
                      Try Again
                    </Button>
                  )}
                </div>
              </div>
            ) : caption ? (
              <div className="image-preview__caption-content">
                <p className="image-preview__caption-text">{caption}</p>
                
                <div className="image-preview__caption-actions">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={copyToClipboard}
                    className="image-preview__copy-button"
                  >
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="mr-1"
                    >
                      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                    </svg>
                    Copy
                  </Button>
                  
                  {onRetry && (
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={onRetry}
                      className="image-preview__retry-button"
                    >
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className="mr-1"
                      >
                        <polyline points="23 4 23 10 17 10" />
                        <polyline points="1 20 1 14 7 14" />
                        <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15" />
                      </svg>
                      Retry
                    </Button>
                  )}
                </div>

                {/* Caption Metadata */}
                {(confidence !== null || processingTime !== null) && (
                  <div className="image-preview__caption-metadata">
                    {confidence !== null && confidence !== undefined && (
                      <div className="image-preview__metadata-item">
                        <span className="image-preview__metadata-label">Confidence:</span>
                        <span className="image-preview__metadata-value">
                          {(confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}
                    {processingTime !== null && processingTime !== undefined && (
                      <div className="image-preview__metadata-item">
                        <span className="image-preview__metadata-label">Processing Time:</span>
                        <span className="image-preview__metadata-value">
                          {formatProcessingTime(processingTime)}
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
};

export default ImagePreview;