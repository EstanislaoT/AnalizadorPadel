import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { Analyses } from './features/analyses/components/Analyses';
import { Dashboard } from './features/dashboard/components/Dashboard';
import { Reports } from './features/reports/components/Reports';
import { Videos } from './features/videos/components/Videos';
import { Layout } from './shared/components/Layout';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#ed6c02',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/videos" element={<Videos />} />
            <Route path="/analyses" element={<Analyses />} />
            <Route path="/reports" element={<Reports />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
