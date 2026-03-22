import React from 'react';
import { Box, Typography, Alert } from '@mui/material';

export const Reports: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 4 }}>
        Reportes
      </Typography>
      <Alert severity="info">
        Esta página está en desarrollo. Los reportes PDF se podrán descargar desde aquí.
      </Alert>
    </Box>
  );
};
