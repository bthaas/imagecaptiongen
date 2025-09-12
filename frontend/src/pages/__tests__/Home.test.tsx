import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Home from '../Home';

describe('Home', () => {
  it('renders the welcome message', () => {
    render(<Home />);

    expect(screen.getByText('Welcome to AI Image Caption Generator')).toBeInTheDocument();
  });

  it('renders the description text', () => {
    render(<Home />);

    expect(screen.getByText(/Upload an image and our AI will generate a descriptive caption/)).toBeInTheDocument();
    expect(screen.getByText(/Our system uses advanced CNN \+ LSTM architecture/)).toBeInTheDocument();
  });

  it('renders all feature sections', () => {
    render(<Home />);

    expect(screen.getByText('🖼️ Multiple Formats')).toBeInTheDocument();
    expect(screen.getByText('Supports JPEG, PNG, and WebP image formats')).toBeInTheDocument();

    expect(screen.getByText('⚡ Fast Processing')).toBeInTheDocument();
    expect(screen.getByText('Get captions in under 10 seconds')).toBeInTheDocument();

    expect(screen.getByText('🎯 Accurate Results')).toBeInTheDocument();
    expect(screen.getByText('Powered by state-of-the-art deep learning models')).toBeInTheDocument();
  });

  it('renders placeholder for upload section', () => {
    render(<Home />);

    expect(screen.getByText('Image upload functionality will be implemented in the next task.')).toBeInTheDocument();
  });

  it('has the correct CSS classes', () => {
    const { container } = render(<Home />);

    expect(container.querySelector('.home')).toBeInTheDocument();
    expect(container.querySelector('.home__content')).toBeInTheDocument();
    expect(container.querySelector('.home__features')).toBeInTheDocument();
    expect(container.querySelector('.home__upload-section')).toBeInTheDocument();
  });
});