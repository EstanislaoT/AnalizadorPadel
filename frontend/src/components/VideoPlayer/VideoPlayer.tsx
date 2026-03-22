import React, { useState, useRef, useEffect } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

export interface VideoPlayerProps {
  videoId: number;
  autoPlay?: boolean;
  controls?: boolean;
  className?: string;
  poster?: string;
  onLoadedMetadata?: (event: React.SyntheticEvent<HTMLVideoElement>) => void;
  onError?: (event: React.SyntheticEvent<HTMLVideoElement, Event>) => void;
  onPlay?: (event: React.SyntheticEvent<HTMLVideoElement>) => void;
  onPause?: (event: React.SyntheticEvent<HTMLVideoElement>) => void;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({
  videoId,
  autoPlay = false,
  controls = true,
  className = '',
  poster,
  onLoadedMetadata,
  onError,
  onPlay,
  onPause,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const videoSrc = `${API_BASE_URL}/videos/${videoId}/stream`;

  useEffect(() => {
    // Reset state when videoId changes
    setIsLoading(true);
    setError(null);
  }, [videoId]);

  const handleLoadedMetadata = (event: React.SyntheticEvent<HTMLVideoElement>) => {
    setIsLoading(false);
    onLoadedMetadata?.(event);
  };

  const handleError = (event: React.SyntheticEvent<HTMLVideoElement, Event>) => {
    setIsLoading(false);
    setError('Failed to load video. Please check if the video exists and the server is running.');
    onError?.(event);
  };

  const handlePlay = (event: React.SyntheticEvent<HTMLVideoElement>) => {
    setIsLoading(false);
    onPlay?.(event);
  };

  const handlePause = (event: React.SyntheticEvent<HTMLVideoElement>) => {
    onPause?.(event);
  };

  const containerStyle: React.CSSProperties = {
    position: 'relative',
    width: '100%',
    maxWidth: '100%',
    backgroundColor: '#000',
    borderRadius: '8px',
    overflow: 'hidden',
  };

  const loadingStyle: React.CSSProperties = {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#1a1a1a',
    color: '#fff',
    fontSize: '16px',
    zIndex: 1,
  };

  const errorStyle: React.CSSProperties = {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#1a1a1a',
    color: '#ff6b6b',
    fontSize: '14px',
    textAlign: 'center',
    padding: '20px',
    zIndex: 1,
  };

  const videoStyle: React.CSSProperties = {
    width: '100%',
    maxWidth: '100%',
    display: 'block',
    backgroundColor: '#000',
  };

  return (
    <div className={className} style={containerStyle}>
      {isLoading && !error && (
        <div style={loadingStyle}>
          Loading video...
        </div>
      )}
      {error && (
        <div style={errorStyle}>
          {error}
        </div>
      )}
      <video
        ref={videoRef}
        src={videoSrc}
        autoPlay={autoPlay}
        controls={controls}
        poster={poster}
        style={videoStyle}
        onLoadedMetadata={handleLoadedMetadata}
        onError={handleError}
        onPlay={handlePlay}
        onPause={handlePause}
      >
        Your browser does not support the video tag.
      </video>
    </div>
  );
};

export default VideoPlayer;
