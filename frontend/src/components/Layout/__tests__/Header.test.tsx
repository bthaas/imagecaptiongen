import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Header from '../Header';

// Mock react-router-dom
const mockUseLocation = jest.fn();
jest.mock('react-router-dom', () => ({
  Link: ({ to, children, className }: any) => (
    <a href={to} className={className}>
      {children}
    </a>
  ),
  useLocation: () => mockUseLocation(),
}));

describe('Header', () => {
  beforeEach(() => {
    mockUseLocation.mockReturnValue({ pathname: '/' });
  });

  it('renders the main title and subtitle', () => {
    render(<Header />);

    expect(screen.getByText('AI Image Caption Generator')).toBeInTheDocument();
    expect(screen.getByText('Upload an image and get an AI-generated caption')).toBeInTheDocument();
  });

  it('renders navigation links', () => {
    render(<Header />);

    const homeLink = screen.getByRole('link', { name: 'Home' });
    const aboutLink = screen.getByRole('link', { name: 'About' });

    expect(homeLink).toBeInTheDocument();
    expect(aboutLink).toBeInTheDocument();
    expect(homeLink).toHaveAttribute('href', '/');
    expect(aboutLink).toHaveAttribute('href', '/about');
  });

  it('highlights active navigation link for home page', () => {
    mockUseLocation.mockReturnValue({ pathname: '/' });
    render(<Header />);

    const homeLink = screen.getByRole('link', { name: 'Home' });
    const aboutLink = screen.getByRole('link', { name: 'About' });

    expect(homeLink).toHaveClass('active');
    expect(aboutLink).not.toHaveClass('active');
  });

  it('highlights active navigation link for about page', () => {
    mockUseLocation.mockReturnValue({ pathname: '/about' });
    render(<Header />);

    const homeLink = screen.getByRole('link', { name: 'Home' });
    const aboutLink = screen.getByRole('link', { name: 'About' });

    expect(homeLink).not.toHaveClass('active');
    expect(aboutLink).toHaveClass('active');
  });

  it('has the correct CSS structure', () => {
    const { container } = render(<Header />);

    expect(container.querySelector('.header')).toBeInTheDocument();
    expect(container.querySelector('.header__container')).toBeInTheDocument();
    expect(container.querySelector('.header__title')).toBeInTheDocument();
    expect(container.querySelector('.header__subtitle')).toBeInTheDocument();
    expect(container.querySelector('.header__nav')).toBeInTheDocument();
  });

  it('applies correct CSS classes to navigation links', () => {
    render(<Header />);

    const homeLink = screen.getByRole('link', { name: 'Home' });
    const aboutLink = screen.getByRole('link', { name: 'About' });

    expect(homeLink).toHaveClass('header__nav-link');
    expect(aboutLink).toHaveClass('header__nav-link');
  });
});