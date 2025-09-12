import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ImagePreview from '../ImagePreview';

// Mock URL.createObjectURL and URL.revokeObjectURL
global.URL.createObjectURL = jest.fn(() => 'mock-url');
global.URL.revokeObjectURL = jest.fn();

// Mock clipboard API
Object.assign(navigator, {
  clipboard: {
    writeText: jest.fn(() => Promise.resolve()),
  },
});

// Mock document.execCommand for fallback clipboard functionality
document.execCommand = jest.fn();

describe('ImagePreview', () => {
  const mockOnRemove = jest.fn();
  const mockOnRetry = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  const createMockFile = (name = 'test.jpg', type = 'image/jpeg', size = 1024) => {
    // Create content that matches the desired size
    const content = size > 0 ? 'x'.repeat(size) : '';
    return new File([content], name, { type });
  };

  it('renders nothing when no file or imageUrl is provided', () => {
    const { container } = render(<ImagePreview file={null} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders image preview with file metadata', async () => {
    const file = createMockFile('test-image.jpg', 'image/jpeg', 2048);
    
    render(<ImagePreview file={file} />);
    
    // Check that metadata is displayed
    expect(screen.getByText('test-image.jpg')).toBeInTheDocument();
    expect(screen.getByText('2 KB')).toBeInTheDocument(); // 2048 bytes = 2 KB
    expect(screen.getByText('image/jpeg')).toBeInTheDocument();
    
    // The image should be rendered, but might show error state initially
    const imageElement = screen.queryByAltText('Preview');
    if (imageElement) {
      expect(imageElement).toBeInTheDocument();
    } else {
      // If image fails to load, error state should be shown
      expect(screen.getByText(/failed to load image/i)).toBeInTheDocument();
    }
  });

  it('shows loading state correctly', () => {
    const file = createMockFile();
    
    render(<ImagePreview file={file} isLoading={true} />);
    
    expect(screen.getByText(/generating caption/i)).toBeInTheDocument();
    expect(screen.getByRole('status', { name: /loading/i })).toBeInTheDocument();
  });

  it('displays caption when provided', () => {
    const file = createMockFile();
    const caption = 'A beautiful sunset over the mountains';
    
    render(
      <ImagePreview 
        file={file} 
        caption={caption}
        confidence={0.85}
        processingTime={2.5}
      />
    );
    
    expect(screen.getByText(caption)).toBeInTheDocument();
    expect(screen.getByText('85.0%')).toBeInTheDocument();
    expect(screen.getByText('2.50s')).toBeInTheDocument();
  });

  it('shows skeleton loading for caption generation', () => {
    const file = createMockFile();
    
    render(<ImagePreview file={file} isLoading={true} />);
    
    const skeletonLines = document.querySelectorAll('.image-preview__skeleton-line');
    expect(skeletonLines).toHaveLength(3);
  });

  it('handles image load and error states', async () => {
    const file = createMockFile();
    
    render(<ImagePreview file={file} />);
    
    // Check if image exists or if error state is shown
    const image = screen.queryByAltText('Preview');
    if (image) {
      // Test image load
      fireEvent.load(image);
      expect(image).toHaveClass('image-preview__image--loaded');
      
      // Test image error
      fireEvent.error(image);
      await waitFor(() => {
        expect(screen.getByText(/failed to load image/i)).toBeInTheDocument();
      });
    } else {
      // If image fails to load immediately, error state should be shown
      expect(screen.getByText(/failed to load image/i)).toBeInTheDocument();
    }
  });

  it('calls onRemove when remove button is clicked', async () => {
    const file = createMockFile();
    
    render(<ImagePreview file={file} onRemove={mockOnRemove} />);
    
    const removeButton = screen.getByRole('button', { name: /remove image/i });
    await userEvent.click(removeButton);
    
    expect(mockOnRemove).toHaveBeenCalledTimes(1);
  });

  it('calls onRetry when retry button is clicked', async () => {
    const file = createMockFile();
    
    render(
      <ImagePreview 
        file={file} 
        caption="Test caption"
        onRetry={mockOnRetry} 
      />
    );
    
    const retryButton = screen.getByRole('button', { name: /retry/i });
    await userEvent.click(retryButton);
    
    expect(mockOnRetry).toHaveBeenCalledTimes(1);
  });

  it('copies caption to clipboard when copy button is clicked', async () => {
    const file = createMockFile();
    const caption = 'Test caption to copy';
    
    render(<ImagePreview file={file} caption={caption} />);
    
    const copyButton = screen.getByRole('button', { name: /copy/i });
    await userEvent.click(copyButton);
    
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(caption);
  });

  it('falls back to execCommand for clipboard when clipboard API fails', async () => {
    const file = createMockFile();
    const caption = 'Test caption to copy';
    
    // Mock clipboard API to fail
    (navigator.clipboard.writeText as jest.Mock).mockRejectedValue(new Error('Clipboard API failed'));
    
    render(<ImagePreview file={file} caption={caption} />);
    
    const copyButton = screen.getByRole('button', { name: /copy/i });
    await userEvent.click(copyButton);
    
    await waitFor(() => {
      expect(document.execCommand).toHaveBeenCalledWith('copy');
    });
  });

  it('formats file sizes correctly', () => {
    const testCases = [
      { size: 1024, expected: '1 KB' },
      { size: 1024 * 1024, expected: '1 MB' },
      { size: 1536, expected: '1.5 KB' },
    ];

    testCases.forEach(({ size, expected }) => {
      const file = createMockFile('test.jpg', 'image/jpeg', size);
      const { unmount } = render(<ImagePreview file={file} />);
      
      expect(screen.getByText(expected)).toBeInTheDocument();
      unmount();
    });
    
    // Test 0 bytes separately
    const zeroFile = createMockFile('test.jpg', 'image/jpeg', 0);
    render(<ImagePreview file={zeroFile} />);
    expect(screen.getByText('0 Bytes')).toBeInTheDocument();
  });

  it('formats processing time correctly', () => {
    const file = createMockFile();
    
    // Test seconds
    const { rerender } = render(
      <ImagePreview file={file} caption="Test" processingTime={2.5} />
    );
    expect(screen.getByText('2.50s')).toBeInTheDocument();
    
    // Test milliseconds
    rerender(
      <ImagePreview file={file} caption="Test" processingTime={0.5} />
    );
    expect(screen.getByText('500ms')).toBeInTheDocument();
  });

  it('cleans up object URLs when file changes', () => {
    const file1 = createMockFile('file1.jpg');
    const file2 = createMockFile('file2.jpg');
    
    const { rerender } = render(<ImagePreview file={file1} />);
    
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    
    rerender(<ImagePreview file={file2} />);
    
    expect(URL.revokeObjectURL).toHaveBeenCalled();
    expect(URL.createObjectURL).toHaveBeenCalledTimes(2);
  });

  it('shows remove button only when onRemove is provided', () => {
    const file = createMockFile();
    
    const { rerender } = render(<ImagePreview file={file} />);
    expect(screen.queryByRole('button', { name: /remove image/i })).not.toBeInTheDocument();
    
    rerender(<ImagePreview file={file} onRemove={mockOnRemove} />);
    expect(screen.getByRole('button', { name: /remove image/i })).toBeInTheDocument();
  });

  it('shows retry button only when onRetry is provided and caption exists', () => {
    const file = createMockFile();
    
    const { rerender } = render(<ImagePreview file={file} caption="Test" />);
    expect(screen.queryByRole('button', { name: /retry/i })).not.toBeInTheDocument();
    
    rerender(<ImagePreview file={file} caption="Test" onRetry={mockOnRetry} />);
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
  });

  it('displays error state for failed caption generation', () => {
    const file = createMockFile();
    const errorMessage = 'Failed to process image';
    
    render(<ImagePreview file={file} error={errorMessage} onRetry={mockOnRetry} />);
    
    expect(screen.getByText('Caption Generation Failed')).toBeInTheDocument();
    expect(screen.getByText(errorMessage)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
  });

  it('calls onRetry when error retry button is clicked', async () => {
    const file = createMockFile();
    
    render(<ImagePreview file={file} error="Test error" onRetry={mockOnRetry} />);
    
    const retryButton = screen.getByRole('button', { name: /try again/i });
    await userEvent.click(retryButton);
    
    expect(mockOnRetry).toHaveBeenCalledTimes(1);
  });

  it('does not show error retry button when onRetry is not provided', () => {
    const file = createMockFile();
    
    render(<ImagePreview file={file} error="Test error" />);
    
    expect(screen.queryByRole('button', { name: /try again/i })).not.toBeInTheDocument();
  });

  it('applies custom className correctly', () => {
    const file = createMockFile();
    
    render(<ImagePreview file={file} className="custom-class" />);
    
    const container = document.querySelector('.image-preview');
    expect(container).toHaveClass('custom-class');
  });

  it('handles imageUrl prop when provided instead of file', () => {
    const imageUrl = 'https://example.com/image.jpg';
    
    render(<ImagePreview file={null} imageUrl={imageUrl} />);
    
    const image = screen.getByAltText('Preview');
    expect(image).toHaveAttribute('src', imageUrl);
  });

  it('prioritizes imageUrl over file when both are provided', () => {
    const file = createMockFile();
    const imageUrl = 'https://example.com/image.jpg';
    
    render(<ImagePreview file={file} imageUrl={imageUrl} />);
    
    const image = screen.getByAltText('Preview');
    expect(image).toHaveAttribute('src', imageUrl);
  });

  it('shows confidence and processing time metadata when provided', () => {
    const file = createMockFile();
    
    render(
      <ImagePreview 
        file={file} 
        caption="Test caption"
        confidence={0.92}
        processingTime={1.25}
      />
    );
    
    expect(screen.getByText('92.0%')).toBeInTheDocument();
    expect(screen.getByText('1.25s')).toBeInTheDocument();
  });

  it('does not show metadata section when confidence and processingTime are null', () => {
    const file = createMockFile();
    
    render(<ImagePreview file={file} caption="Test caption" />);
    
    expect(screen.queryByText(/confidence/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/processing time/i)).not.toBeInTheDocument();
  });
});