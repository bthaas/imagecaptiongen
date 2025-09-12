import React, { useState, useCallback } from 'react';
import { Button } from '../UI';

interface CaptionDisplayProps {
  caption: string | null;
  confidence?: number | null;
  processingTime?: number | null;
  isLoading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  onCopy?: (caption: string) => void;
  className?: string;
}

const CaptionDisplay: React.FC<CaptionDisplayProps> = ({
  caption,
  confidence,
  processingTime,
  isLoading = false,
  error,
  onRetry,
  onCopy,
  className = ''
}) => {
  const [copySuccess, setCopySuccess] = useState(false);

  const handleCopyToClipboard = useCallback(async () => {
    if (!caption) return;

    try {
      await navigator.clipboard.writeText(caption);
      setCopySuccess(true);
      onCopy?.(caption);
      
      // Reset copy success state after 2 seconds
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (error) {
      // Fallback for older browsers
      try {
        const textArea = document.createElement('textarea');
        textArea.value = caption;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        
        setCopySuccess(true);
        onCopy?.(caption);
        setTimeout(() => setCopySuccess(false), 2000);
      } catch (fallbackError) {
        console.error('Failed to copy to clipboard:', fallbackError);
      }
    }
  }, [caption, onCopy]);

  const formatProcessingTime = (time: number): string => {
    if (time === null || time === undefined) return '';
    return time < 1 ? `${(time * 1000).toFixed(0)}ms` : `${time.toFixed(2)}s`;
  };

  const containerClasses = [
    'caption-display',
    isLoading ? 'caption-display--loading' : '',
    error ? 'caption-display--error' : '',
    className
  ].filter(Boolean).join(' ');

  return (
    <div className={containerClasses}>
      <div className="caption-display__header">
        <h3 className="caption-display__title">Generated Caption</h3>
        
        {/* Status indicator */}
        {isLoading && (
          <div className="caption-display__status caption-display__status--loading">
            <div className="caption-display__spinner" />
            <span>Generating...</span>
          </div>
        )}
        
        {error && (
          <div className="caption-display__status caption-display__status--error">
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
              <circle cx="12" cy="12" r="10" />
              <line x1="15" y1="9" x2="9" y2="15" />
              <line x1="9" y1="9" x2="15" y2="15" />
            </svg>
            <span>Failed</span>
          </div>
        )}
        
        {caption && caption.trim() && !isLoading && !error && (
          <div className="caption-display__status caption-display__status--success">
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
              <polyline points="20,6 9,17 4,12" />
            </svg>
            <span>Complete</span>
          </div>
        )}
      </div>

      <div className="caption-display__content">
        {/* Loading State */}
        {isLoading && (
          <div className="caption-display__loading">
            <div className="caption-display__skeleton">
              <div className="caption-display__skeleton-line caption-display__skeleton-line--long" />
              <div className="caption-display__skeleton-line caption-display__skeleton-line--medium" />
              <div className="caption-display__skeleton-line caption-display__skeleton-line--short" />
            </div>
            <div className="caption-display__loading-text">
              <div className="caption-display__loading-dots">
                <span>Analyzing image</span>
                <span className="caption-display__dot">.</span>
                <span className="caption-display__dot">.</span>
                <span className="caption-display__dot">.</span>
              </div>
            </div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="caption-display__error">
            <div className="caption-display__error-icon">
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
            <div className="caption-display__error-content">
              <h4 className="caption-display__error-title">Caption Generation Failed</h4>
              <p className="caption-display__error-message">{error}</p>
              {onRetry && (
                <Button
                  variant="primary"
                  size="sm"
                  onClick={onRetry}
                  className="caption-display__retry-button"
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
        )}

        {/* Success State */}
        {caption && caption.trim() && !isLoading && !error && (
          <div className="caption-display__success">
            <div className="caption-display__text-container">
              <p className="caption-display__text">{caption}</p>
            </div>

            <div className="caption-display__actions">
              <Button
                variant="secondary"
                size="sm"
                onClick={handleCopyToClipboard}
                className="caption-display__copy-button"
                disabled={!caption || !caption.trim()}
              >
                {copySuccess ? (
                  <>
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
                      <polyline points="20,6 9,17 4,12" />
                    </svg>
                    Copied!
                  </>
                ) : (
                  <>
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
                  </>
                )}
              </Button>

              {onRetry && (
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={onRetry}
                  className="caption-display__regenerate-button"
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
                  Regenerate
                </Button>
              )}
            </div>

            {/* Metadata */}
            {(confidence !== null || processingTime !== null) && (
              <div className="caption-display__metadata">
                {confidence !== null && confidence !== undefined && (
                  <div className="caption-display__metadata-item">
                    <span className="caption-display__metadata-label">Confidence:</span>
                    <div className="caption-display__confidence">
                      <div className="caption-display__confidence-bar">
                        <div 
                          className="caption-display__confidence-fill"
                          style={{ width: `${confidence * 100}%` }}
                        />
                      </div>
                      <span className="caption-display__confidence-value">
                        {(confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                )}
                {processingTime !== null && processingTime !== undefined && (
                  <div className="caption-display__metadata-item">
                    <span className="caption-display__metadata-label">Processing Time:</span>
                    <span className="caption-display__metadata-value">
                      {formatProcessingTime(processingTime)}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default CaptionDisplay;