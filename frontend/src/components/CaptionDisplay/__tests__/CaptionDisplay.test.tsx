import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import CaptionDisplay from '../CaptionDisplay';

// Mock clipboard API
Object.assign(navigator, {
  clipboard: {
    writeText: jest.fn(() => Promise.resolve()),
  },
});

// Mock document.execCommand for fallback clipboard functionality
document.execCommand = jest.fn();

describe('CaptionDisplay', () => {
  const mockOnRetry = jest.fn();
  const mockOnCopy = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('renders loading state correctly', () => {
    render(<CaptionDisplay caption={null} isLoading={true} />);
    
    expect(screen.getByText('Generated Caption')).toBeInTheDocument();
    expect(screen.getByText('Generating...')).toBeInTheDocument();
    expect(screen.getByText('Analyzing image')).toBeInTheDocument();
    
    // Check skeleton loading elements
    const skeletonLines = document.querySelectorAll('.caption-display__skeleton-line');
    expect(skeletonLines).toHaveLength(3);
  });

  it('renders error state correctly', () => {
    const errorMessage = 'Failed to process image';
    
    render(
      <CaptionDisplay 
        caption={null} 
        error={errorMessage}
        onRetry={mockOnRetry}
      />
    );
    
    expect(screen.getByText('Caption Generation Failed')).toBeInTheDocument();
    expect(screen.getByText(errorMessage)).toBeInTheDocument();
    expect(screen.getByText('Failed')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
  });

  it('renders success state with caption correctly', () => {
    const caption = 'A beautiful sunset over the mountains';
    
    render(
      <CaptionDisplay 
        caption={caption}
        confidence={0.85}
        processingTime={2.5}
      />
    );
    
    expect(screen.getByText(caption)).toBeInTheDocument();
    expect(screen.getByText('Complete')).toBeInTheDocument();
    expect(screen.getByText('85.0%')).toBeInTheDocument();
    expect(screen.getByText('2.50s')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /copy/i })).toBeInTheDocument();
  });

  it('calls onRetry when retry button is clicked in error state', async () => {
    render(
      <CaptionDisplay 
        caption={null} 
        error="Test error"
        onRetry={mockOnRetry}
      />
    );
    
    const retryButton = screen.getByRole('button', { name: /try again/i });
    await userEvent.click(retryButton);
    
    expect(mockOnRetry).toHaveBeenCalledTimes(1);
  });

  it('calls onRetry when regenerate button is clicked in success state', async () => {
    render(
      <CaptionDisplay 
        caption="Test caption"
        onRetry={mockOnRetry}
      />
    );
    
    const regenerateButton = screen.getByRole('button', { name: /regenerate/i });
    await userEvent.click(regenerateButton);
    
    expect(mockOnRetry).toHaveBeenCalledTimes(1);
  });

  it('copies caption to clipboard when copy button is clicked', async () => {
    const caption = 'Test caption to copy';
    
    render(
      <CaptionDisplay 
        caption={caption}
        onCopy={mockOnCopy}
      />
    );
    
    const copyButton = screen.getByRole('button', { name: /copy/i });
    
    await act(async () => {
      await userEvent.click(copyButton);
    });
    
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(caption);
    expect(mockOnCopy).toHaveBeenCalledWith(caption);
    
    // Check that button text changes to "Copied!"
    await waitFor(() => {
      expect(screen.getByText('Copied!')).toBeInTheDocument();
    });
  });

  it('shows copy success state and resets after timeout', async () => {
    const caption = 'Test caption';
    
    render(<CaptionDisplay caption={caption} />);
    
    const copyButton = screen.getByRole('button', { name: /copy/i });
    
    await act(async () => {
      await userEvent.click(copyButton);
    });
    
    await waitFor(() => {
      expect(screen.getByText('Copied!')).toBeInTheDocument();
    });
    
    // Fast-forward time to trigger reset
    act(() => {
      jest.advanceTimersByTime(2000);
    });
    
    await waitFor(() => {
      expect(screen.getByText('Copy')).toBeInTheDocument();
    });
  });

  it('falls back to execCommand when clipboard API fails', async () => {
    const caption = 'Test caption';
    
    // Mock clipboard API to fail
    (navigator.clipboard.writeText as jest.Mock).mockRejectedValue(new Error('Clipboard API failed'));
    
    render(<CaptionDisplay caption={caption} onCopy={mockOnCopy} />);
    
    const copyButton = screen.getByRole('button', { name: /copy/i });
    
    await act(async () => {
      await userEvent.click(copyButton);
    });
    
    await waitFor(() => {
      expect(document.execCommand).toHaveBeenCalledWith('copy');
      expect(mockOnCopy).toHaveBeenCalledWith(caption);
    });
  });

  it('formats processing time correctly', () => {
    const { rerender } = render(
      <CaptionDisplay caption="Test" processingTime={2.5} />
    );
    expect(screen.getByText('2.50s')).toBeInTheDocument();
    
    rerender(<CaptionDisplay caption="Test" processingTime={0.5} />);
    expect(screen.getByText('500ms')).toBeInTheDocument();
    
    rerender(<CaptionDisplay caption="Test" processingTime={0.123} />);
    expect(screen.getByText('123ms')).toBeInTheDocument();
  });

  it('displays confidence bar with correct width', () => {
    render(<CaptionDisplay caption="Test" confidence={0.75} />);
    
    const confidenceFill = document.querySelector('.caption-display__confidence-fill');
    expect(confidenceFill).toHaveStyle({ width: '75%' });
    expect(screen.getByText('75.0%')).toBeInTheDocument();
  });

  it('does not show metadata when confidence and processingTime are null', () => {
    render(<CaptionDisplay caption="Test caption" />);
    
    expect(screen.queryByText(/confidence/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/processing time/i)).not.toBeInTheDocument();
  });

  it('does not show retry button when onRetry is not provided', () => {
    render(<CaptionDisplay caption="Test caption" />);
    
    expect(screen.queryByRole('button', { name: /regenerate/i })).not.toBeInTheDocument();
  });

  it('does not show retry button in error state when onRetry is not provided', () => {
    render(<CaptionDisplay caption={null} error="Test error" />);
    
    expect(screen.queryByRole('button', { name: /try again/i })).not.toBeInTheDocument();
  });

  it('applies custom className correctly', () => {
    render(<CaptionDisplay caption="Test" className="custom-class" />);
    
    const container = document.querySelector('.caption-display');
    expect(container).toHaveClass('custom-class');
  });

  it('applies loading className when isLoading is true', () => {
    render(<CaptionDisplay caption={null} isLoading={true} />);
    
    const container = document.querySelector('.caption-display');
    expect(container).toHaveClass('caption-display--loading');
  });

  it('applies error className when error is present', () => {
    render(<CaptionDisplay caption={null} error="Test error" />);
    
    const container = document.querySelector('.caption-display');
    expect(container).toHaveClass('caption-display--error');
  });

  it('disables copy button when no caption is provided', () => {
    render(<CaptionDisplay caption={null} />);
    
    // Should not render copy button when no caption
    expect(screen.queryByRole('button', { name: /copy/i })).not.toBeInTheDocument();
  });

  it('handles copy button click when caption is empty string', async () => {
    render(<CaptionDisplay caption="" />);
    
    // Empty caption should not render the success state, so no copy button
    expect(screen.queryByRole('button', { name: /copy/i })).not.toBeInTheDocument();
  });

  it('shows correct status indicators for different states', () => {
    const { rerender } = render(<CaptionDisplay caption={null} isLoading={true} />);
    expect(screen.getByText('Generating...')).toBeInTheDocument();
    
    rerender(<CaptionDisplay caption={null} error="Test error" />);
    expect(screen.getByText('Failed')).toBeInTheDocument();
    
    rerender(<CaptionDisplay caption="Test caption" />);
    expect(screen.getByText('Complete')).toBeInTheDocument();
  });

  it('renders without crashing when all props are null/undefined', () => {
    render(<CaptionDisplay caption={null} />);
    
    expect(screen.getByText('Generated Caption')).toBeInTheDocument();
  });

  it('handles confidence value of 0 correctly', () => {
    render(<CaptionDisplay caption="Test" confidence={0} />);
    
    expect(screen.getByText('0.0%')).toBeInTheDocument();
    
    const confidenceFill = document.querySelector('.caption-display__confidence-fill');
    expect(confidenceFill).toHaveStyle({ width: '0%' });
  });

  it('handles confidence value of 1 correctly', () => {
    render(<CaptionDisplay caption="Test" confidence={1} />);
    
    expect(screen.getByText('100.0%')).toBeInTheDocument();
    
    const confidenceFill = document.querySelector('.caption-display__confidence-fill');
    expect(confidenceFill).toHaveStyle({ width: '100%' });
  });

  it('handles very small processing times correctly', () => {
    render(<CaptionDisplay caption="Test" processingTime={0.001} />);
    
    expect(screen.getByText('1ms')).toBeInTheDocument();
  });

  it('handles clipboard copy failure gracefully', async () => {
    const caption = 'Test caption';
    
    // Mock both clipboard API and execCommand to fail
    (navigator.clipboard.writeText as jest.Mock).mockRejectedValue(new Error('Clipboard API failed'));
    (document.execCommand as jest.Mock).mockImplementation(() => {
      throw new Error('execCommand failed');
    });
    
    // Mock console.error to avoid test output noise
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
    
    render(<CaptionDisplay caption={caption} />);
    
    const copyButton = screen.getByRole('button', { name: /copy/i });
    
    await act(async () => {
      await userEvent.click(copyButton);
    });
    
    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith('Failed to copy to clipboard:', expect.any(Error));
    });
    
    consoleSpy.mockRestore();
  });
});