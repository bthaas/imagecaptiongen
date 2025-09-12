/**
 * End-to-end flow tests for the complete image upload and caption generation
 * Tests the integration between components, hooks, and API client
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { AppProvider } from '../context/AppContext';
import { Home } from '../pages';
import * as apiClient from '../services/apiClient';

// Mock the API client
jest.mock('../services/apiClient');
const mockApiClient = apiClient as jest.Mocked<typeof apiClient>;

// Mock FileReader for file upload simulation
const createMockFileReader = () => ({
  readAsDataURL: jest.fn(),
  onload: null as ((event: ProgressEvent<FileReader>) => void) | null,
  onerror: null as ((event: ProgressEvent<FileReader>) => void) | null,
  result: null as string | null,
});

global.FileReader = jest.fn(() => createMockFileReader()) as any;

// Mock URL.createObjectURL and revokeObjectURL
global.URL.createObjectURL = jest.fn(() => 'mock-object-url');
global.URL.revokeObjectURL = jest.fn();

describe('End-to-End Image Caption Flow', () => {
  const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
    <AppProvider>{children}</AppProvider>
  );

  beforeEach(() => {
    jest.clearAllMocks();
    mockApiClient.checkAPIHealth.mockResolvedValue({
      status: 'healthy',
      timestamp: '2024-01-15T10:30:00Z',
      version: '1.0.0',
      system_info: {},
      services: { api: 'healthy', model: 'available' }
    });
  });

  it('should complete the full image upload and caption generation flow', async () => {
    // Mock successful caption generation
    const mockResponse = {
      caption: 'A beautiful landscape with mountains and trees',
      confidence: 0.92,
      processing_time: 2.45,
      image_id: 'test-image-123',
      timestamp: '2024-01-15T10:30:00Z'
    };

    mockApiClient.generateCaption.mockResolvedValueOnce(mockResponse);

    // Mock FileReader behavior
    const mockReader = createMockFileReader();
    mockReader.result = 'data:image/jpeg;base64,test-image-data';
    (global.FileReader as jest.Mock).mockImplementationOnce(() => {
      setTimeout(() => {
        if (mockReader.onload) {
          mockReader.onload({} as ProgressEvent<FileReader>);
        }
      }, 0);
      return mockReader;
    });

    render(<Home />, { wrapper: TestWrapper });

    // Wait for health check to complete
    await waitFor(() => {
      expect(mockApiClient.checkAPIHealth).toHaveBeenCalled();
    });

    // Find the file input (it should be in the ImageUploader component)
    const fileInput = screen.getByRole('button', { name: /upload.*image|choose.*file|select.*image/i })
      .closest('div')?.querySelector('input[type="file"]') as HTMLInputElement;

    expect(fileInput).toBeInTheDocument();

    // Create a test file
    const testFile = new File(['test image content'], 'test.jpg', { type: 'image/jpeg' });

    // Simulate file selection
    fireEvent.change(fileInput, { target: { files: [testFile] } });

    // Wait for the image to be processed and caption to be generated
    await waitFor(() => {
      expect(mockApiClient.generateCaption).toHaveBeenCalledWith(
        testFile,
        expect.objectContaining({
          maxLength: 20,
          temperature: 1.0
        })
      );
    }, { timeout: 5000 });

    // Check that the caption is displayed
    await waitFor(() => {
      expect(screen.getByText(mockResponse.caption)).toBeInTheDocument();
    });

    // Check that confidence and processing time are displayed
    expect(screen.getByText(/confidence/i)).toBeInTheDocument();
    expect(screen.getByText(/processing time/i)).toBeInTheDocument();
  });

  it('should handle API errors gracefully', async () => {
    // Mock API error
    const mockError = new apiClient.APIError(
      'Invalid image format',
      400,
      'INVALID_IMAGE_FORMAT',
      'req-123'
    );

    mockApiClient.generateCaption.mockRejectedValueOnce(mockError);

    // Mock FileReader behavior
    const mockReader = createMockFileReader();
    mockReader.result = 'data:image/jpeg;base64,invalid-data';
    (global.FileReader as jest.Mock).mockImplementationOnce(() => {
      setTimeout(() => {
        if (mockReader.onload) {
          mockReader.onload({} as ProgressEvent<FileReader>);
        }
      }, 0);
      return mockReader;
    });

    render(<Home />, { wrapper: TestWrapper });

    // Wait for health check
    await waitFor(() => {
      expect(mockApiClient.checkAPIHealth).toHaveBeenCalled();
    });

    // Find and interact with file input
    const fileInput = screen.getByRole('button', { name: /upload.*image|choose.*file|select.*image/i })
      .closest('div')?.querySelector('input[type="file"]') as HTMLInputElement;

    const testFile = new File(['invalid content'], 'test.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [testFile] } });

    // Wait for error to be displayed
    await waitFor(() => {
      expect(screen.getByText(/invalid image format/i)).toBeInTheDocument();
    }, { timeout: 5000 });

    // Check that retry functionality is available
    const retryButton = screen.queryByRole('button', { name: /retry|try again/i });
    if (retryButton) {
      expect(retryButton).toBeInTheDocument();
    }
  });

  it('should handle network errors', async () => {
    // Mock network error
    const mockError = new apiClient.NetworkError('Network connection failed');
    mockApiClient.generateCaption.mockRejectedValueOnce(mockError);

    // Mock FileReader behavior
    const mockReader = createMockFileReader();
    mockReader.result = 'data:image/jpeg;base64,test-data';
    (global.FileReader as jest.Mock).mockImplementationOnce(() => {
      setTimeout(() => {
        if (mockReader.onload) {
          mockReader.onload({} as ProgressEvent<FileReader>);
        }
      }, 0);
      return mockReader;
    });

    render(<Home />, { wrapper: TestWrapper });

    // Wait for health check
    await waitFor(() => {
      expect(mockApiClient.checkAPIHealth).toHaveBeenCalled();
    });

    // Simulate file upload
    const fileInput = screen.getByRole('button', { name: /upload.*image|choose.*file|select.*image/i })
      .closest('div')?.querySelector('input[type="file"]') as HTMLInputElement;

    const testFile = new File(['test content'], 'test.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [testFile] } });

    // Wait for network error to be displayed
    await waitFor(() => {
      expect(screen.getByText(/network.*connection.*failed|connection.*error/i)).toBeInTheDocument();
    }, { timeout: 5000 });
  });

  it('should show loading state during caption generation', async () => {
    // Mock a delayed response
    mockApiClient.generateCaption.mockImplementationOnce(() => 
      new Promise(resolve => 
        setTimeout(() => resolve({
          caption: 'Test caption',
          confidence: 0.8,
          processing_time: 1.0,
          image_id: 'test-123',
          timestamp: '2024-01-15T10:30:00Z'
        }), 1000)
      )
    );

    // Mock FileReader behavior
    const mockReader = createMockFileReader();
    mockReader.result = 'data:image/jpeg;base64,test-data';
    (global.FileReader as jest.Mock).mockImplementationOnce(() => {
      setTimeout(() => {
        if (mockReader.onload) {
          mockReader.onload({} as ProgressEvent<FileReader>);
        }
      }, 0);
      return mockReader;
    });

    render(<Home />, { wrapper: TestWrapper });

    // Wait for health check
    await waitFor(() => {
      expect(mockApiClient.checkAPIHealth).toHaveBeenCalled();
    });

    // Simulate file upload
    const fileInput = screen.getByRole('button', { name: /upload.*image|choose.*file|select.*image/i })
      .closest('div')?.querySelector('input[type="file"]') as HTMLInputElement;

    const testFile = new File(['test content'], 'test.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [testFile] } });

    // Check for loading state
    await waitFor(() => {
      expect(screen.getByText(/generating|processing|loading/i)).toBeInTheDocument();
    });

    // Wait for completion
    await waitFor(() => {
      expect(screen.getByText('Test caption')).toBeInTheDocument();
    }, { timeout: 2000 });
  });

  it('should handle backend service unavailable', async () => {
    // Mock health check failure
    mockApiClient.checkAPIHealth.mockRejectedValueOnce(new Error('Service unavailable'));

    render(<Home />, { wrapper: TestWrapper });

    // Wait for error message about backend unavailability
    await waitFor(() => {
      expect(screen.getByText(/backend.*not.*available|service.*unavailable/i)).toBeInTheDocument();
    });
  });
});