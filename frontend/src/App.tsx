import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard/Dashboard';
import { Videos } from './pages/Videos';
import { Analyses } from './pages/Analyses';
import { Reports } from './pages/Reports';

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
