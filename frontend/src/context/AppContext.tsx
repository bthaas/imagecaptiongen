import React, { createContext, useContext, useReducer, ReactNode } from 'react';
import { AppState, AppError } from '../types';

// Define action types
export type AppAction =
  | { type: 'SET_SELECTED_IMAGE'; payload: { file: File; preview: string } }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_CAPTION'; payload: { caption: string; confidence: number; processingTime: number; requestId?: string } }
  | { type: 'SET_ERROR'; payload: string | AppError }
  | { type: 'CLEAR_ERROR' }
  | { type: 'RESET_STATE' };

// Initial state
const initialState: AppState = {
  selectedImage: null,
  imagePreview: null,
  caption: null,
  isLoading: false,
  error: null,
  confidence: null,
  processingTime: null,
  requestId: null,
};

// Reducer function
const appReducer = (state: AppState, action: AppAction): AppState => {
  switch (action.type) {
    case 'SET_SELECTED_IMAGE':
      return {
        ...state,
        selectedImage: action.payload.file,
        imagePreview: action.payload.preview,
        caption: null,
        error: null,
        confidence: null,
        processingTime: null,
      };
    case 'SET_LOADING':
      return {
        ...state,
        isLoading: action.payload,
        error: null,
      };
    case 'SET_CAPTION':
      return {
        ...state,
        caption: action.payload.caption,
        confidence: action.payload.confidence,
        processingTime: action.payload.processingTime,
        requestId: action.payload.requestId || null,
        isLoading: false,
        error: null,
      };
    case 'SET_ERROR':
      return {
        ...state,
        error: typeof action.payload === 'string' ? action.payload : action.payload.message,
        isLoading: false,
      };
    case 'CLEAR_ERROR':
      return {
        ...state,
        error: null,
      };
    case 'RESET_STATE':
      return initialState;
    default:
      return state;
  }
};

// Context type
interface AppContextType {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
}

// Create context
const AppContext = createContext<AppContextType | undefined>(undefined);

// Provider component
interface AppProviderProps {
  children: ReactNode;
}

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
};

// Custom hook to use the context
export const useAppContext = (): AppContextType => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
};