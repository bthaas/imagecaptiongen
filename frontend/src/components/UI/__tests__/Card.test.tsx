import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Card from '../Card';

describe('Card', () => {
  it('renders children content', () => {
    render(
      <Card>
        <p>Card content</p>
      </Card>
    );
    
    expect(screen.getByText('Card content')).toBeInTheDocument();
  });

  it('applies default classes', () => {
    const { container } = render(
      <Card>
        <p>Content</p>
      </Card>
    );
    
    const card = container.firstChild as HTMLElement;
    expect(card).toHaveClass('bg-white', 'border-radius-lg', 'p-4', 'shadow-sm');
  });

  it('applies different padding sizes', () => {
    const { container, rerender } = render(
      <Card padding="sm">
        <p>Small padding</p>
      </Card>
    );
    
    let card = container.firstChild as HTMLElement;
    expect(card).toHaveClass('p-3');

    rerender(
      <Card padding="lg">
        <p>Large padding</p>
      </Card>
    );
    
    card = container.firstChild as HTMLElement;
    expect(card).toHaveClass('p-5');
  });

  it('applies different shadow sizes', () => {
    const { container, rerender } = render(
      <Card shadow="md">
        <p>Medium shadow</p>
      </Card>
    );
    
    let card = container.firstChild as HTMLElement;
    expect(card).toHaveClass('shadow-md');

    rerender(
      <Card shadow="lg">
        <p>Large shadow</p>
      </Card>
    );
    
    card = container.firstChild as HTMLElement;
    expect(card).toHaveClass('shadow-lg');
  });

  it('applies custom className', () => {
    const { container } = render(
      <Card className="custom-card">
        <p>Custom card</p>
      </Card>
    );
    
    const card = container.firstChild as HTMLElement;
    expect(card).toHaveClass('custom-card');
  });

  it('combines all classes correctly', () => {
    const { container } = render(
      <Card padding="lg" shadow="md" className="custom">
        <p>All props</p>
      </Card>
    );
    
    const card = container.firstChild as HTMLElement;
    expect(card).toHaveClass(
      'bg-white',
      'border-radius-lg', 
      'p-5',
      'shadow-md',
      'custom'
    );
  });
});