/**
 * Debug component to test API connectivity
 */

import React, { useState } from 'react';
import { useAPI } from '../hooks/useAPI';

const DebugTest: React.FC = () => {
  const [testResult, setTestResult] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const { checkHealth, generateCaption } = useAPI();

  const testHealthCheck = async () => {
    setIsLoading(true);
    setTestResult('Testing health check...');
    
    try {
      const isHealthy = await checkHealth();
      setTestResult(`Health check result: ${isHealthy ? 'SUCCESS' : 'FAILED'}`);
    } catch (error) {
      setTestResult(`Health check error: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  const testImageUpload = async () => {
    setIsLoading(true);
    setTestResult('Testing image upload...');
    
    try {
      // Create a simple test image (red square)
      const canvas = document.createElement('canvas');
      canvas.width = 64;
      canvas.height = 64;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = 'red';
        ctx.fillRect(0, 0, 64, 64);
      }

      // Convert to blob and then to File
      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((blob) => {
          resolve(blob!);
        }, 'image/jpeg');
      });

      const testFile = new File([blob], 'test.jpg', { type: 'image/jpeg' });
      
      const success = await generateCaption(testFile, {
        maxLength: 20,
        temperature: 1.0
      });

      setTestResult(`Image upload result: ${success ? 'SUCCESS' : 'FAILED'}`);
    } catch (error) {
      setTestResult(`Image upload error: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ 
      padding: '20px', 
      border: '2px solid #ccc', 
      margin: '20px', 
      borderRadius: '8px',
      backgroundColor: '#f9f9f9'
    }}>
      <h3>🔧 Debug Test Panel</h3>
      <p>Use this panel to test the API connectivity:</p>
      
      <div style={{ marginBottom: '10px' }}>
        <button 
          onClick={testHealthCheck}
          disabled={isLoading}
          style={{
            padding: '10px 20px',
            marginRight: '10px',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: isLoading ? 'not-allowed' : 'pointer'
          }}
        >
          Test Health Check
        </button>
        
        <button 
          onClick={testImageUpload}
          disabled={isLoading}
          style={{
            padding: '10px 20px',
            backgroundColor: '#28a745',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: isLoading ? 'not-allowed' : 'pointer'
          }}
        >
          Test Image Upload
        </button>
      </div>

      <div style={{
        padding: '10px',
        backgroundColor: '#fff',
        border: '1px solid #ddd',
        borderRadius: '4px',
        minHeight: '50px',
        fontFamily: 'monospace'
      }}>
        {isLoading ? 'Loading...' : testResult || 'Click a button to run a test'}
      </div>

      <div style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
        <strong>Instructions:</strong>
        <ul>
          <li>First, test the health check to verify backend connectivity</li>
          <li>Then, test image upload to verify the full flow</li>
          <li>Check the browser console (F12) for detailed logs</li>
          <li>Backend should be running on http://localhost:8000</li>
        </ul>
      </div>
    </div>
  );
};

export default DebugTest;