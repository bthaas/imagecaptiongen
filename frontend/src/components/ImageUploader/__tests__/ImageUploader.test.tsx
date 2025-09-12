import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ImageUploader from '../ImageUploader';

// Mock URL.createObjectURL and URL.revokeObjectURL
global.URL.createObjectURL = jest.fn(() => 'mock-url');
global.URL.revokeObjectURL = jest.fn();

// Mock Image constructor for file validation
const mockImage = {
  onload: null as any,
  onerror: null as any,
  src: '',
};

global.Image = jest.fn(() => mockImage) as any;

// Mock clipboard API
Object.assign(navigator, {
  clipboard: {
    writeText: jest.fn(() => Promise.resolve()),
  },
});

describe('ImageUploader', () => {
  const mockOnImageSelect = jest.fn();
  const mockOnImageUpload = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  const defaultProps = {
    onImageSelect: mockOnImageSelect,
    maxFileSize: 10 * 1024 * 1024, // 10MB
    acceptedFormats: ['image/jpeg', 'image/png', 'image/webp'],
  };

  it('renders the upload interface correctly', () => {
    render(<ImageUploader {...defaultProps} />);
    
    expect(screen.getByText(/drag and drop an image here/i)).toBeInTheDocument();
    expect(screen.getByText(/supports jpeg, png, webp/i)).toBeInTheDocument();
    expect(screen.getByText(/max size: 10 mb/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /browse files/i })).toBeInTheDocument();
  });

  it('handles file selection via file input', async () => {
    render(<ImageUploader {...defaultProps} />);
    
    const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    
    await userEvent.upload(fileInput, file);
    
    // Simulate successful image validation
    setTimeout(() => {
      if (mockImage.onload) mockImage.onload();
    }, 0);
    
    await waitFor(() => {
      expect(mockOnImageSelect).toHaveBeenCalledWith(file);
    });
  });

  it('handles drag and drop functionality', async () => {
    render(<ImageUploader {...defaultProps} />);
    
    const dropZone = screen.getByRole('button', { name: /upload image/i });
    const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
    
    // Test drag over
    fireEvent.dragOver(dropZone, {
      dataTransfer: {
        files: [file],
      },
    });
    
    expect(screen.getByText(/drop your image here/i)).toBeInTheDocument();
    
    // Test drop
    fireEvent.drop(dropZone, {
      dataTransfer: {
        files: [file],
      },
    });
    
    // Simulate successful image validation
    setTimeout(() => {
      if (mockImage.onload) mockImage.onload();
    }, 0);
    
    await waitFor(() => {
      expect(mockOnImageSelect).toHaveBeenCalledWith(file);
    });
  });

  it('validates file format correctly', async () => {
    render(<ImageUploader {...defaultProps} />);
    
    const invalidFile = new File(['test'], 'test.txt', { type: 'text/plain' });
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    
    await userEvent.upload(fileInput, invalidFile);
    
    await waitFor(() => {
      expect(screen.getByText(/invalid file format/i)).toBeInTheDocument();
    });
    
    expect(mockOnImageSelect).not.toHaveBeenCalled();
  });

  it('validates file size correctly', async () => {
    render(<ImageUploader {...defaultProps} maxFileSize={1024} />); // 1KB limit
    
    const largeFile = new File(['x'.repeat(2048)], 'large.jpg', { type: 'image/jpeg' });
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    
    await userEvent.upload(fileInput, largeFile);
    
    await waitFor(() => {
      expect(screen.getByText(/file size too large/i)).toBeInTheDocument();
    });
    
    expect(mockOnImageSelect).not.toHaveBeenCalled();
  });

  it('handles corrupted image files', async () => {
    render(<ImageUploader {...defaultProps} />);
    
    const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    
    await userEvent.upload(fileInput, file);
    
    // Simulate image validation failure
    setTimeout(() => {
      if (mockImage.onerror) mockImage.onerror();
    }, 0);
    
    await waitFor(() => {
      expect(screen.getByText(/invalid image file or corrupted data/i)).toBeInTheDocument();
    });
    
    expect(mockOnImageSelect).not.toHaveBeenCalled();
  });

  it('shows upload progress when onImageUpload is provided', async () => {
    const mockUpload = jest.fn(() => new Promise(resolve => setTimeout(resolve, 1000)));
    
    render(<ImageUploader {...defaultProps} onImageUpload={mockUpload} />);
    
    const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    
    await userEvent.upload(fileInput, file);
    
    // Simulate successful image validation
    setTimeout(() => {
      if (mockImage.onload) mockImage.onload();
    }, 0);
    
    await waitFor(() => {
      expect(screen.getByText(/uploading/i)).toBeInTheDocument();
    });
    
    expect(mockUpload).toHaveBeenCalledWith(file);
  });

  it('handles upload errors correctly', async () => {
    const mockUpload = jest.fn(() => Promise.reject(new Error('Upload failed')));
    
    render(<ImageUploader {...defaultProps} onImageUpload={mockUpload} />);
    
    const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    
    await userEvent.upload(fileInput, file);
    
    // Simulate successful image validation
    setTimeout(() => {
      if (mockImage.onload) mockImage.onload();
    }, 0);
    
    await waitFor(() => {
      expect(screen.getByText(/upload failed/i)).toBeInTheDocument();
    });
  });

  it('can clear error messages', async () => {
    render(<ImageUploader {...defaultProps} />);
    
    const invalidFile = new File(['test'], 'test.txt', { type: 'text/plain' });
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    
    await userEvent.upload(fileInput, invalidFile);
    
    await waitFor(() => {
      expect(screen.getByText(/invalid file format/i)).toBeInTheDocument();
    });
    
    const closeButton = screen.getByRole('button', { name: /clear error/i });
    await userEvent.click(closeButton);
    
    expect(screen.queryByText(/invalid file format/i)).not.toBeInTheDocument();
  });

  it('is disabled when disabled prop is true', () => {
    render(<ImageUploader {...defaultProps} disabled={true} />);
    
    const dropZone = screen.getByRole('button', { name: /upload image/i });
    const browseButton = screen.getByRole('button', { name: /browse files/i });
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    
    expect(dropZone).toHaveAttribute('tabIndex', '-1');
    expect(browseButton).toBeDisabled();
    expect(fileInput).toBeDisabled();
  });

  it('supports keyboard navigation', async () => {
    render(<ImageUploader {...defaultProps} />);
    
    const dropZone = screen.getByRole('button', { name: /upload image/i });
    
    // Focus the drop zone
    dropZone.focus();
    expect(dropZone).toHaveFocus();
    
    // Press Enter to trigger file selection
    await userEvent.keyboard('{Enter}');
    
    // The file input should be triggered (we can't easily test this without mocking)
    expect(dropZone).toHaveFocus();
  });

  it('handles multiple file selection by taking only the first file', async () => {
    render(<ImageUploader {...defaultProps} />);
    
    const file1 = new File(['test1'], 'test1.jpg', { type: 'image/jpeg' });
    const file2 = new File(['test2'], 'test2.jpg', { type: 'image/jpeg' });
    
    const dropZone = screen.getByRole('button', { name: /upload image/i });
    
    fireEvent.drop(dropZone, {
      dataTransfer: {
        files: [file1, file2],
      },
    });
    
    // Simulate successful image validation
    setTimeout(() => {
      if (mockImage.onload) mockImage.onload();
    }, 0);
    
    await waitFor(() => {
      expect(mockOnImageSelect).toHaveBeenCalledWith(file1);
      expect(mockOnImageSelect).toHaveBeenCalledTimes(1);
    });
  });

  it('formats file sizes correctly', () => {
    render(<ImageUploader {...defaultProps} maxFileSize={1024 * 1024} />);
    
    expect(screen.getByText(/max size: 1 mb/i)).toBeInTheDocument();
  });

  it('shows correct accepted formats in the UI', () => {
    render(
      <ImageUploader 
        {...defaultProps} 
        acceptedFormats={['image/jpeg', 'image/png']} 
      />
    );
    
    expect(screen.getByText(/supports jpeg, png/i)).toBeInTheDocument();
  });

  it('prevents drag and drop when disabled', () => {
    render(<ImageUploader {...defaultProps} disabled={true} />);
    
    const dropZone = screen.getByRole('button', { name: /upload image/i });
    const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
    
    fireEvent.dragOver(dropZone, {
      dataTransfer: {
        files: [file],
      },
    });
    
    // Should not show drag over state when disabled
    expect(screen.queryByText(/drop your image here/i)).not.toBeInTheDocument();
  });
});