import React, { useEffect, useState } from 'react';
import { Box, Grid, Typography, CircularProgress, Alert, List, ListItem, ListItemButton, ListItemText, Chip, Paper } from '@mui/material';
import VideoPlayer from '../../../shared/components/VideoPlayer/VideoPlayer';
import { AnalizadorPadelApiService } from '../../../shared/services/api/generated/services/AnalizadorPadelApiService';
import type { VideoDto } from '../../../shared/services/api/generated/models/VideoDto';

export const Videos: React.FC = () => {
  const [videos, setVideos] = useState<VideoDto[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<VideoDto | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchVideos = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await AnalizadorPadelApiService.getVideos();
        if (response.data?.success && response.data?.data) {
          setVideos(response.data.data as VideoDto[]);
        }
      } catch (err: any) {
        setError(err.message || 'Failed to fetch videos');
      } finally {
        setLoading(false);
      }
    };
    fetchVideos();
  }, []);

  const handleSelectVideo = (video: VideoDto) => {
    setSelectedVideo(video);
  };

  const getStatusColor = (status: string | undefined): 'success' | 'warning' | 'error' | 'default' | 'info' => {
    switch (status) {
      case 'Completed':
        return 'success';
      case 'Processing':
        return 'warning';
      case 'Failed':
        return 'error';
      case 'Uploaded':
        return 'info';
      default:
        return 'default';
    }
  };

  const formatDate = (dateString: string | undefined): string => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleDateString('es-CL', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '50vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box>
        <Typography variant="h4" sx={{ mb: 4 }}>
          Videos
        </Typography>
        <Alert severity="error">
          {error}
        </Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 4 }}>
        Videos
      </Typography>

      <Grid container spacing={3}>
        {/* Video List */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, maxHeight: '70vh', overflow: 'auto' }}>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Lista de Videos
            </Typography>
            {videos.length > 0 ? (
              <List>
                {videos.map((video) => (
                  <ListItem key={video.id} disablePadding>
                    <ListItemButton
                      selected={selectedVideo?.id === video.id}
                      onClick={() => handleSelectVideo(video)}
                      sx={{
                        borderRadius: 1,
                        mb: 1,
                        '&.Mui-selected': {
                          bgcolor: 'primary.light',
                          '&:hover': {
                            bgcolor: 'primary.light',
                          },
                        },
                      }}
                    >
                      <ListItemText
                        primary={video.name || `Video #${video.id}`}
                        secondary={formatDate(video.uploadedAt)}
                        primaryTypographyProps={{
                          noWrap: true,
                          sx: { fontWeight: selectedVideo?.id === video.id ? 600 : 400 }
                        }}
                      />
                      <Chip
                        label={video.status || 'Unknown'}
                        color={getStatusColor(video.status)}
                        size="small"
                      />
                    </ListItemButton>
                  </ListItem>
                ))}
              </List>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No hay videos disponibles
              </Typography>
            )}
          </Paper>
        </Grid>

        {/* Video Player */}
        <Grid item xs={12} md={8}>
          {selectedVideo ? (
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" sx={{ mb: 2 }}>
                {selectedVideo.name || `Video #${selectedVideo.id}`}
              </Typography>
              <Box sx={{ mb: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <Chip
                  label={selectedVideo.status || 'Unknown'}
                  color={getStatusColor(selectedVideo.status)}
                />
                <Typography variant="body2" color="text.secondary">
                  Subido: {formatDate(selectedVideo.uploadedAt)}
                </Typography>
              </Box>
              {selectedVideo.id ? (
                <VideoPlayer videoId={selectedVideo.id} />
              ) : (
                <Alert severity="warning">
                  Este video no tiene un ID válido
                </Alert>
              )}
            </Paper>
          ) : (
            <Paper sx={{ p: 4, textAlign: 'center', minHeight: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Typography variant="body1" color="text.secondary">
                Selecciona un video de la lista para reproducirlo
              </Typography>
            </Paper>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default Videos;
