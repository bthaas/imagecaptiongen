import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Footer from '../Footer';

describe('Footer', () => {
  it('renders the footer text', () => {
    render(<Footer />);

    expect(screen.getByText('© 2024 AI Image Caption Generator. Powered by TensorFlow and React.')).toBeInTheDocument();
  });

  it('has the correct CSS classes', () => {
    const { container } = render(<Footer />);
    
    expect(container.firstChild).toHaveClass('footer');
    expect(container.querySelector('.footer__container')).toBeInTheDocument();
    expect(container.querySelector('.footer__text')).toBeInTheDocument();
  });
});