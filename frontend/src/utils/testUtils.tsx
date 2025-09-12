import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { AppProvider } from '../context/AppContext';

// Custom render function that includes providers
const AllTheProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <AppProvider>
      {children}
    </AppProvider>
  );
};

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) => render(ui, { wrapper: AllTheProviders, ...options });

// Re-export everything
export * from '@testing-library/react';

// Override render method
export { customRender as render };

// Helper function to create mock files for testing
export const createMockFile = (
  name: string = 'test.jpg',
  size: number = 1024,
  type: string = 'image/jpeg'
): File => {
  const file = new File([''], name, { type });
  Object.defineProperty(file, 'size', {
    value: size,
    writable: false,
  });
  return file;
};

// Helper function to create mock FileReader
export const mockFileReader = (result: string | ArrayBuffer | null = 'data:image/jpeg;base64,test') => {
  const mockReader = {
    readAsDataURL: jest.fn(),
    result,
    onload: null as ((event: ProgressEvent<FileReader>) => void) | null,
    onerror: null as ((event: ProgressEvent<FileReader>) => void) | null,
  };

  // Simulate successful file read
  mockReader.readAsDataURL.mockImplementation(() => {
    setTimeout(() => {
      if (mockReader.onload) {
        mockReader.onload({} as ProgressEvent<FileReader>);
      }
    }, 0);
  });

  return mockReader;
};

// Mock fetch for API testing
export const mockFetch = (response: any, ok: boolean = true, status: number = 200) => {
  return jest.fn().mockResolvedValue({
    ok,
    status,
    json: jest.fn().mockResolvedValue(response),
    text: jest.fn().mockResolvedValue(JSON.stringify(response)),
  });
};