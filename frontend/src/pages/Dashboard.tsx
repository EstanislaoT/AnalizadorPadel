import React, { useEffect, useState } from 'react';
import { Box, Grid, Card, CardContent, Typography, CircularProgress, Alert, List, ListItem, ListItemText, Chip, Paper } from '@mui/material';
import { VideoLibrary, Analytics, CheckCircle, TrendingUp } from '@mui/icons-material';
import axios from 'axios';

interface DashboardStats {
  totalVideos: number;
  totalAnalyses: number;
  completedAnalyses: number;
  failedAnalyses: number;
  successRatePercent: number;
  avgDetectionRate: number;
  recentVideos: Array<{
    id: number;
    name: string;
    status: string;
    uploadedAt: string;
  }>;
  recentAnalyses: Array<{
    id: number;
    videoId: number;
    status: string;
    startedAt: string;
  }>;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

export const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDashboardStats = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/dashboard/stats`);
        if (response.data.success) {
          setStats(response.data.data);
        } else {
          setError(response.data.message);
        }
      } catch (err) {
        setError('Error al cargar las estadísticas del dashboard');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardStats();
  }, []);

  const getStatusColor = (status: string): 'success' | 'warning' | 'error' | 'default' | 'info' => {
    switch (status) {
      case 'Completed':
        return 'success';
      case 'Processing':
      case 'Running':
        return 'warning';
      case 'Failed':
        return 'error';
      case 'Uploaded':
        return 'info';
      default:
        return 'default';
    }
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
      <Alert severity="error" sx={{ mt: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 4 }}>
        Dashboard
      </Typography>

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ bgcolor: '#1976d2', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h6">Videos</Typography>
                  <Typography variant="h3">{stats?.totalVideos || 0}</Typography>
                </Box>
                <VideoLibrary sx={{ fontSize: 48, opacity: 0.7 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ bgcolor: '#ed6c02', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h6">Análisis</Typography>
                  <Typography variant="h3">{stats?.totalAnalyses || 0}</Typography>
                </Box>
                <Analytics sx={{ fontSize: 48, opacity: 0.7 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ bgcolor: '#2e7d32', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h6">Exitosos</Typography>
                  <Typography variant="h3">{stats?.completedAnalyses || 0}</Typography>
                </Box>
                <CheckCircle sx={{ fontSize: 48, opacity: 0.7 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ bgcolor: stats && stats.successRatePercent >= 70 ? '#2e7d32' : '#d32f2f', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h6">Tasa de Éxito</Typography>
                  <Typography variant="h3">{stats?.successRatePercent.toFixed(1)}%</Typography>
                </Box>
                <TrendingUp sx={{ fontSize: 48, opacity: 0.7 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Recent Videos and Analyses */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Videos Recientes
            </Typography>
            {stats?.recentVideos && stats.recentVideos.length > 0 ? (
              <List>
                {stats.recentVideos.map((video) => (
                  <ListItem key={video.id} divider>
                    <ListItemText
                      primary={video.name}
                      secondary={`Subido: ${new Date(video.uploadedAt).toLocaleDateString()}`}
                    />
                    <Chip
                      label={video.status}
                      color={getStatusColor(video.status)}
                      size="small"
                    />
                  </ListItem>
                ))}
              </List>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No hay videos subidos aún
              </Typography>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Análisis Recientes
            </Typography>
            {stats?.recentAnalyses && stats.recentAnalyses.length > 0 ? (
              <List>
                {stats.recentAnalyses.map((analysis) => (
                  <ListItem key={analysis.id} divider>
                    <ListItemText
                      primary={`Análisis #${analysis.id}`}
                      secondary={`Video #${analysis.videoId} - Iniciado: ${new Date(analysis.startedAt).toLocaleDateString()}`}
                    />
                    <Chip
                      label={analysis.status}
                      color={getStatusColor(analysis.status)}
                      size="small"
                    />
                  </ListItem>
                ))}
              </List>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No hay análisis realizados aún
              </Typography>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Detection Rate Info */}
      {stats && stats.totalAnalyses > 0 && (
        <Paper sx={{ p: 2, mt: 3, bgcolor: '#f5f5f5' }}>
          <Typography variant="h6" sx={{ mb: 1 }}>
            📊 Información Adicional
          </Typography>
          <Typography variant="body2">
            <strong>Tasa de detección promedio:</strong> {stats.avgDetectionRate.toFixed(1)}%
          </Typography>
          <Typography variant="body2">
            <strong>Análisis fallidos:</strong> {stats.failedAnalyses}
          </Typography>
        </Paper>
      )}
    </Box>
  );
};
