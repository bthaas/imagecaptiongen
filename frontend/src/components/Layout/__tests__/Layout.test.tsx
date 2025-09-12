import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Layout from '../Layout';

// Mock the Header and Footer components to avoid router dependencies
jest.mock('../Header', () => {
  return function MockHeader() {
    return <div data-testid="mock-header">AI Image Caption Generator</div>;
  };
});

jest.mock('../Footer', () => {
  return function MockFooter() {
    return <div data-testid="mock-footer">© 2024 AI Image Caption Generator. Powered by TensorFlow and React.</div>;
  };
});

describe('Layout', () => {
  it('renders children content', () => {
    render(
      <Layout>
        <div>Test content</div>
      </Layout>
    );

    expect(screen.getByText('Test content')).toBeInTheDocument();
  });

  it('renders header and footer components', () => {
    render(
      <Layout>
        <div>Test content</div>
      </Layout>
    );

    // Check for mocked header and footer
    expect(screen.getByTestId('mock-header')).toBeInTheDocument();
    expect(screen.getByTestId('mock-footer')).toBeInTheDocument();
  });

  it('has the correct CSS structure', () => {
    const { container } = render(
      <Layout>
        <div>Test content</div>
      </Layout>
    );

    expect(container.querySelector('.layout')).toBeInTheDocument();
    expect(container.querySelector('.layout__main')).toBeInTheDocument();
    expect(container.querySelector('.layout__container')).toBeInTheDocument();
  });

  it('renders main content area', () => {
    render(
      <Layout>
        <div data-testid="main-content">Main content</div>
      </Layout>
    );

    const mainContent = screen.getByTestId('main-content');
    expect(mainContent).toBeInTheDocument();
    expect(mainContent.closest('.layout__container')).toBeInTheDocument();
  });
});