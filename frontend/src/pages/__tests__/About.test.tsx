import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import About from '../About';

describe('About', () => {
  it('renders the main heading', () => {
    render(<About />);

    expect(screen.getByText('About AI Image Caption Generator')).toBeInTheDocument();
  });

  it('renders all section headings', () => {
    render(<About />);

    expect(screen.getByText('How It Works')).toBeInTheDocument();
    expect(screen.getByText('Technology Stack')).toBeInTheDocument();
    expect(screen.getByText('Features')).toBeInTheDocument();
  });

  it('renders technology stack information', () => {
    render(<About />);

    expect(screen.getByText(/Frontend:/)).toBeInTheDocument();
    expect(screen.getByText(/React with TypeScript/)).toBeInTheDocument();
    
    expect(screen.getByText(/Backend:/)).toBeInTheDocument();
    expect(screen.getByText(/FastAPI with Python/)).toBeInTheDocument();
    
    expect(screen.getByText(/Machine Learning:/)).toBeInTheDocument();
    expect(screen.getByText(/TensorFlow/)).toBeInTheDocument();
    
    expect(screen.getByText(/Deployment:/)).toBeInTheDocument();
    expect(screen.getByText(/Google Cloud Platform/)).toBeInTheDocument();
  });

  it('renders feature list', () => {
    render(<About />);

    expect(screen.getByText('Real-time image caption generation')).toBeInTheDocument();
    expect(screen.getByText('Support for multiple image formats')).toBeInTheDocument();
    expect(screen.getByText('Responsive web interface')).toBeInTheDocument();
    expect(screen.getByText('Fast processing with cloud deployment')).toBeInTheDocument();
    expect(screen.getByText('Error handling and user feedback')).toBeInTheDocument();
  });

  it('renders how it works description', () => {
    render(<About />);

    expect(screen.getByText(/sophisticated deep learning architecture/)).toBeInTheDocument();
    expect(screen.getByText(/Convolutional Neural Networks \(CNN\)/)).toBeInTheDocument();
    expect(screen.getByText(/Long Short-Term Memory \(LSTM\)/)).toBeInTheDocument();
  });

  it('has the correct CSS classes', () => {
    const { container } = render(<About />);

    expect(container.querySelector('.about')).toBeInTheDocument();
    expect(container.querySelector('.about__content')).toBeInTheDocument();
    expect(container.querySelectorAll('.about__section')).toHaveLength(3);
  });
});