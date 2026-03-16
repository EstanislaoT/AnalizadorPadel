import React from 'react';
import { Box, Typography, Alert } from '@mui/material';

export const Videos: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 4 }}>
        Videos
      </Typography>
      <Alert severity="info">
        Esta página está en desarrollo. Por favor usa la funcionalidad existente en el Dashboard.
      </Alert>
    </Box>
  );
};
