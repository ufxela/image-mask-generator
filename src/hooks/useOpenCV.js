import { useState, useEffect } from 'react';

/**
 * Custom hook to manage OpenCV.js loading state
 * OpenCV.js is loaded from CDN in index.html
 * This hook monitors when it becomes available
 */
export function useOpenCV() {
  const [cv, setCv] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Check if OpenCV is already loaded
    if (window.cv && window.cv.Mat) {
      setCv(window.cv);
      setLoading(false);
      return;
    }

    // Set up global callback for when OpenCV loads
    let checkInterval;

    const onOpenCvReady = () => {
      if (window.cv && window.cv.Mat) {
        setCv(window.cv);
        setLoading(false);
        clearInterval(checkInterval);
      }
    };

    // Poll for OpenCV availability
    checkInterval = setInterval(() => {
      if (window.cv && window.cv.Mat) {
        onOpenCvReady();
      }
    }, 100);

    // Timeout after 30 seconds
    const timeout = setTimeout(() => {
      if (!window.cv) {
        setError('Failed to load OpenCV.js. Please refresh the page.');
        setLoading(false);
        clearInterval(checkInterval);
      }
    }, 30000);

    return () => {
      clearInterval(checkInterval);
      clearTimeout(timeout);
    };
  }, []);

  return { cv, loading, error };
}
