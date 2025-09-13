import React, { useCallback, useEffect } from 'react';
import { ImageUploader, ImagePreview } from '../components';
import { useAppContext } from '../context/AppContext';
import { useAPI } from '../hooks/useAPI';
import { ErrorType } from '../types';
import '../components/ImageUploader/ImageUploader.css';
import '../components/ImagePreview/ImagePreview.css';
import '../components/CaptionDisplay/CaptionDisplay.css';

const Home: React.FC = () => {
  const { state, dispatch } = useAppContext();
  const { generateCaption, retryCaption, checkHealth } = useAPI();

  // Check API health on component mount
  useEffect(() => {
    const performHealthCheck = async () => {
      const isHealthy = await checkHealth();
      if (!isHealthy) {
        dispatch({
          type: 'SET_ERROR',
          payload: {
            type: ErrorType.NETWORK_ERROR,
            message: 'Backend service is not available. Please try again later.',
            retryable: true
          }
        });
      }
    };

    performHealthCheck();
  }, [checkHealth, dispatch]);

  const handleImageSelect = useCallback((file: File) => {
    // Create preview URL
    const previewUrl = URL.createObjectURL(file);
    
    dispatch({
      type: 'SET_SELECTED_IMAGE',
      payload: { file, preview: previewUrl }
    });
  }, [dispatch]);

  const handleImageUpload = useCallback(async (file: File): Promise<void> => {
    try {
      // Generate caption immediately after image selection
      const success = await generateCaption(file, {
        maxLength: 20,
        temperature: 1.0
      });

      if (!success) {
        console.error('Caption generation failed');
      }
    } catch (error) {
      console.error('Error during image upload:', error);
      dispatch({
        type: 'SET_ERROR',
        payload: {
          type: ErrorType.UNKNOWN_ERROR,
          message: 'Failed to process image. Please try again.',
          retryable: true
        }
      });
    }
  }, [generateCaption, dispatch]);

  const handleRemoveImage = useCallback(() => {
    // Clean up the preview URL to prevent memory leaks
    if (state.imagePreview) {
      URL.revokeObjectURL(state.imagePreview);
    }
    dispatch({ type: 'RESET_STATE' });
  }, [state.imagePreview, dispatch]);

  const handleRetryCaption = useCallback(async () => {
    if (state.selectedImage) {
      await retryCaption(state.selectedImage, {
        maxLength: 20,
        temperature: 1.0
      });
    }
  }, [state.selectedImage, retryCaption]);

  return (
    <div className="home">
      <div className="home__content">
        <h2>Welcome to AI Image Caption Generator</h2>
        <p>
          Upload an image and our AI will generate a descriptive caption for you.
          Our system uses advanced CNN + LSTM architecture to analyze your images
          and produce natural language descriptions.
        </p>
        
        <div className="home__features">
          <div className="feature">
            <h3>🖼️ Multiple Formats</h3>
            <p>Supports JPEG, PNG, and WebP image formats</p>
          </div>
          <div className="feature">
            <h3>⚡ Fast Processing</h3>
            <p>Get captions in under 10 seconds</p>
          </div>
          <div className="feature">
            <h3>🎯 Accurate Results</h3>
            <p>Powered by state-of-the-art deep learning models</p>
          </div>
        </div>

        {/* Image Upload Section */}
        <div className="home__upload-section">
          {!state.selectedImage ? (
            <ImageUploader
              onImageSelect={handleImageSelect}
              onImageUpload={handleImageUpload}
              maxFileSize={10 * 1024 * 1024} // 10MB
              acceptedFormats={['image/jpeg', 'image/png', 'image/webp']}
              disabled={state.isLoading}
            />
          ) : (
            <ImagePreview
              file={state.selectedImage}
              imageUrl={state.imagePreview}
              caption={state.caption}
              isLoading={state.isLoading}
              confidence={state.confidence}
              processingTime={state.processingTime}
              error={state.error}
              onRemove={handleRemoveImage}
              onRetry={handleRetryCaption}
            />
          )}
        </div>

        {/* Error Display */}
        {state.error && (
          <div className="home__error">
            <div className="alert alert--error">
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="alert__icon"
              >
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
              <span>{state.error}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Home;