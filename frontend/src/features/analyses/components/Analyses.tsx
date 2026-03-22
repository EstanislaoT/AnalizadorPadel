import React from 'react';
import { Box, Typography, Alert } from '@mui/material';

export const Analyses: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 4 }}>
        Análisis
      </Typography>
      <Alert severity="info">
        Esta página está en desarrollo. Los análisis se mostrarán aquí una vez completados.
      </Alert>
    </Box>
  );
};
