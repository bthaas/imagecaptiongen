import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Spinner from '../Spinner';

describe('Spinner', () => {
  it('renders with default props', () => {
    render(<Spinner />);
    
    const spinner = screen.getByRole('status');
    expect(spinner).toBeInTheDocument();
    expect(spinner).toHaveClass('spinner');
    expect(spinner).toHaveAttribute('aria-label', 'Loading');
  });

  it('renders with large size', () => {
    render(<Spinner size="lg" />);
    
    const spinner = screen.getByRole('status');
    expect(spinner).toHaveClass('spinner', 'spinner-lg');
  });

  it('renders with custom className', () => {
    render(<Spinner className="custom-class" />);
    
    const spinner = screen.getByRole('status');
    expect(spinner).toHaveClass('spinner', 'custom-class');
  });

  it('includes screen reader text', () => {
    render(<Spinner />);
    
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });
});