import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { AppProvider, useAppContext } from '../AppContext';

// Test component that uses the context
const TestComponent: React.FC = () => {
  const { state, dispatch } = useAppContext();

  return (
    <div>
      <div data-testid="loading">{state.isLoading ? 'Loading' : 'Not Loading'}</div>
      <div data-testid="error">{state.error || 'No Error'}</div>
      <div data-testid="caption">{state.caption || 'No Caption'}</div>
      <div data-testid="confidence">{state.confidence || 'No Confidence'}</div>
      
      <button 
        onClick={() => dispatch({ type: 'SET_LOADING', payload: true })}
        data-testid="set-loading"
      >
        Set Loading
      </button>
      
      <button 
        onClick={() => dispatch({ type: 'SET_ERROR', payload: 'Test error' })}
        data-testid="set-error"
      >
        Set Error
      </button>
      
      <button 
        onClick={() => dispatch({ 
          type: 'SET_CAPTION', 
          payload: { caption: 'Test caption', confidence: 0.95, processingTime: 1.5 }
        })}
        data-testid="set-caption"
      >
        Set Caption
      </button>
      
      <button 
        onClick={() => dispatch({ type: 'CLEAR_ERROR' })}
        data-testid="clear-error"
      >
        Clear Error
      </button>
      
      <button 
        onClick={() => dispatch({ type: 'RESET_STATE' })}
        data-testid="reset-state"
      >
        Reset State
      </button>
    </div>
  );
};

const renderWithProvider = (component: React.ReactElement) => {
  return render(
    <AppProvider>
      {component}
    </AppProvider>
  );
};

describe('AppContext', () => {
  it('provides initial state', () => {
    renderWithProvider(<TestComponent />);

    expect(screen.getByTestId('loading')).toHaveTextContent('Not Loading');
    expect(screen.getByTestId('error')).toHaveTextContent('No Error');
    expect(screen.getByTestId('caption')).toHaveTextContent('No Caption');
    expect(screen.getByTestId('confidence')).toHaveTextContent('No Confidence');
  });

  it('handles SET_LOADING action', () => {
    renderWithProvider(<TestComponent />);

    fireEvent.click(screen.getByTestId('set-loading'));
    expect(screen.getByTestId('loading')).toHaveTextContent('Loading');
  });

  it('handles SET_ERROR action', () => {
    renderWithProvider(<TestComponent />);

    fireEvent.click(screen.getByTestId('set-error'));
    expect(screen.getByTestId('error')).toHaveTextContent('Test error');
    expect(screen.getByTestId('loading')).toHaveTextContent('Not Loading');
  });

  it('handles SET_CAPTION action', () => {
    renderWithProvider(<TestComponent />);

    fireEvent.click(screen.getByTestId('set-caption'));
    expect(screen.getByTestId('caption')).toHaveTextContent('Test caption');
    expect(screen.getByTestId('confidence')).toHaveTextContent('0.95');
    expect(screen.getByTestId('loading')).toHaveTextContent('Not Loading');
    expect(screen.getByTestId('error')).toHaveTextContent('No Error');
  });

  it('handles CLEAR_ERROR action', () => {
    renderWithProvider(<TestComponent />);

    // First set an error
    fireEvent.click(screen.getByTestId('set-error'));
    expect(screen.getByTestId('error')).toHaveTextContent('Test error');

    // Then clear it
    fireEvent.click(screen.getByTestId('clear-error'));
    expect(screen.getByTestId('error')).toHaveTextContent('No Error');
  });

  it('handles RESET_STATE action', () => {
    renderWithProvider(<TestComponent />);

    // Set some state
    fireEvent.click(screen.getByTestId('set-loading'));
    fireEvent.click(screen.getByTestId('set-error'));
    fireEvent.click(screen.getByTestId('set-caption'));

    // Reset state
    fireEvent.click(screen.getByTestId('reset-state'));

    expect(screen.getByTestId('loading')).toHaveTextContent('Not Loading');
    expect(screen.getByTestId('error')).toHaveTextContent('No Error');
    expect(screen.getByTestId('caption')).toHaveTextContent('No Caption');
    expect(screen.getByTestId('confidence')).toHaveTextContent('No Confidence');
  });

  it('throws error when used outside provider', () => {
    // Suppress console.error for this test
    const originalError = console.error;
    console.error = jest.fn();

    expect(() => {
      render(<TestComponent />);
    }).toThrow('useAppContext must be used within an AppProvider');

    console.error = originalError;
  });
});