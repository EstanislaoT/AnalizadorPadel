import { http, HttpResponse } from 'msw';
import type { components } from '../../services/api/generated/types';

// Type aliases for convenience
type VideoDto = components['schemas']['VideoDto'];
type AnalysisDto = components['schemas']['AnalysisDto'];
type DashboardStats = components['schemas']['DashboardStats'];
type ApiResponseOfObject = components['schemas']['ApiResponseOfObject'];
type ApiResponseOfDashboardStats = components['schemas']['ApiResponseOfDashboardStats'];

// Helper type for API responses
type ApiResponse<T> = {
  success: boolean;
  message: string;
  data: T;
};

// Mock data
const mockVideos: VideoDto[] = [
  {
    id: 1,
    name: 'Partido 1',
    description: 'Partido de prueba 1',
    filePath: '/uploads/video1.mp4',
    uploadedAt: '2026-03-20T10:00:00Z',
    status: 'Completed',
    analysisId: 1,
  },
  {
    id: 2,
    name: 'Partido 2',
    description: null,
    filePath: '/uploads/video2.mp4',
    uploadedAt: '2026-03-20T11:00:00Z',
    status: 'Processing',
    analysisId: 2,
  },
  {
    id: 3,
    name: 'Partido 3',
    description: 'Partido fallido',
    filePath: '/uploads/video3.mp4',
    uploadedAt: '2026-03-20T12:00:00Z',
    status: 'Failed',
    analysisId: null,
  },
];

const mockAnalyses: AnalysisDto[] = [
  {
    id: 1,
    videoId: 1,
    startedAt: '2026-03-20T10:05:00Z',
    completedAt: '2026-03-20T10:15:00Z',
    status: 'Completed',
    errorMessage: null,
    result: {
      status: 'success',
      videoPath: '/uploads/video1.mp4',
      processingTimeSeconds: 600,
      totalFrames: 15000,
      playersDetected: 4,
      avgDetectionsPerFrame: 3.8,
      framesWith4Players: 12000,
      detectionRatePercent: 80,
      modelUsed: 'yolov8m.pt',
      timestamp: '2026-03-20T10:15:00Z',
    },
  },
  {
    id: 2,
    videoId: 2,
    startedAt: '2026-03-20T11:00:00Z',
    completedAt: null,
    status: 'Running',
    errorMessage: null,
    result: null,
  },
];

const mockDashboardStats: DashboardStats = {
  totalVideos: 10,
  totalAnalyses: 8,
  completedAnalyses: 6,
  failedAnalyses: 2,
  successRatePercent: 75,
  avgDetectionRate: 82.5,
  recentVideos: mockVideos,
  recentAnalyses: mockAnalyses,
};

const mockAnalysisStats = {
  totalFrames: 15000,
  framesWith4Players: 12000,
  detectionRatePercent: 80,
  avgDetectionsPerFrame: 3.8,
  playersDetected: 4,
  processingTimeSeconds: 600,
  modelUsed: 'yolov8m.pt',
};

const mockHeatmapData = {
  points: Array.from({ length: 100 }, () => ({
    x: Math.random() * 23.77,
    y: Math.random() * 10.97,
    intensity: Math.floor(Math.random() * 10) + 1,
  })),
  courtDimensions: '23.77m x 10.97m',
};

// Helper to create handlers for both path-only and full URL patterns
const createHandler = (method: string, path: string, handler: any) => {
  const fullPath = path.startsWith('/') ? path : `/${path}`;
  return [
    // Path-only pattern (e.g., /api/health)
    (http as any)[method](fullPath, handler),
    // Full URL pattern (e.g., http://localhost:5000/api/health)
    (http as any)[method](`http://localhost:5000${fullPath}`, handler),
    // Also support any localhost port
    (http as any)[method](`http://localhost:*/${fullPath.replace(/^\//, '')}`, handler),
  ];
};

// MSW Handlers
export const handlers = [
  // Health check
  http.get('/api/health', () => {
    const response: ApiResponse<object> = {
      success: true,
      message: 'API healthy',
      data: { status: 'healthy', timestamp: new Date().toISOString() },
    };
    return HttpResponse.json(response);
  }),

  // Videos
  http.get('/api/videos', () => {
    const response: ApiResponse<VideoDto[]> = {
      success: true,
      message: `${mockVideos.length} videos encontrados`,
      data: mockVideos,
    };
    return HttpResponse.json(response);
  }),

  http.get('/api/videos/:id', ({ params }) => {
    const video = mockVideos.find((v) => v.id === Number(params.id));
    if (!video) {
      const response: ApiResponse<null> = {
        success: false,
        message: `Video ${params.id} no encontrado`,
        data: null,
      };
      return HttpResponse.json(response, { status: 404 });
    }
    const response: ApiResponse<VideoDto> = {
      success: true,
      message: 'Video encontrado',
      data: video,
    };
    return HttpResponse.json(response);
  }),

  http.post('/api/videos', async ({ request }) => {
    const formData = await request.formData();
    const file = formData.get('file') as File;

    if (!file) {
      const response: ApiResponse<null> = {
        success: false,
        message: 'No se proporcionó ningún video',
        data: null,
      };
      return HttpResponse.json(response, { status: 400 });
    }

    const newVideo: VideoDto = {
      id: mockVideos.length + 1,
      name: file.name.replace(/\.[^/.]+$/, ''),
      description: null,
      filePath: `/uploads/${file.name}`,
      uploadedAt: new Date().toISOString(),
      status: 'Uploaded',
      analysisId: null,
    };

    const response: ApiResponse<VideoDto> = {
      success: true,
      message: 'Video subido exitosamente',
      data: newVideo,
    };
    return HttpResponse.json(response, { status: 201 });
  }),

  http.delete('/api/videos/:id', ({ params }) => {
    const video = mockVideos.find((v) => v.id === Number(params.id));
    if (!video) {
      const response: ApiResponse<null> = {
        success: false,
        message: `Video ${params.id} no encontrado`,
        data: null,
      };
      return HttpResponse.json(response, { status: 404 });
    }
    const response: ApiResponse<null> = {
      success: true,
      message: 'Video eliminado exitosamente',
      data: null,
    };
    return HttpResponse.json(response);
  }),

  // Video stream
  http.get('/api/videos/:id/stream', () => {
    return new HttpResponse(null, {
      status: 200,
      headers: {
        'Content-Type': 'video/mp4',
        'Accept-Ranges': 'bytes',
      },
    });
  }),

  // Analyses
  http.get('/api/analyses/:id', ({ params }) => {
    const analysis = mockAnalyses.find((a) => a.id === Number(params.id));
    if (!analysis) {
      const response: ApiResponse<null> = {
        success: false,
        message: `Análisis ${params.id} no encontrado`,
        data: null,
      };
      return HttpResponse.json(response, { status: 404 });
    }
    const response: ApiResponse<AnalysisDto> = {
      success: true,
      message: 'Análisis encontrado',
      data: analysis,
    };
    return HttpResponse.json(response);
  }),

  http.post('/api/videos/:id/analyse', ({ params }) => {
    const video = mockVideos.find((v) => v.id === Number(params.id));
    if (!video) {
      const response: ApiResponse<null> = {
        success: false,
        message: `Video ${params.id} no encontrado`,
        data: null,
      };
      return HttpResponse.json(response, { status: 404 });
    }

    const newAnalysis: AnalysisDto = {
      id: mockAnalyses.length + 1,
      videoId: Number(params.id),
      startedAt: new Date().toISOString(),
      completedAt: null,
      status: 'Running',
      errorMessage: null,
      result: null,
    };

    const response: ApiResponse<AnalysisDto> = {
      success: true,
      message: 'Análisis iniciado',
      data: newAnalysis,
    };
    return HttpResponse.json(response, {
      status: 202,
      headers: {
        Location: `/api/analyses/${newAnalysis.id}`,
      },
    });
  }),

  http.get('/api/analyses/:id/stats', ({ params }) => {
    const analysis = mockAnalyses.find((a) => a.id === Number(params.id));
    if (!analysis || analysis.status !== 'Completed') {
      const response: ApiResponse<null> = {
        success: false,
        message: `Análisis ${params.id} no encontrado o sin resultados`,
        data: null,
      };
      return HttpResponse.json(response, { status: 404 });
    }

    const response: ApiResponse<typeof mockAnalysisStats> = {
      success: true,
      message: 'Estadísticas obtenidas',
      data: mockAnalysisStats,
    };
    return HttpResponse.json(response);
  }),

  http.get('/api/analyses/:id/heatmap', ({ params }) => {
    const analysis = mockAnalyses.find((a) => a.id === Number(params.id));
    if (!analysis || analysis.status !== 'Completed') {
      const response: ApiResponse<null> = {
        success: false,
        message: `Análisis ${params.id} no encontrado o sin resultados`,
        data: null,
      };
      return HttpResponse.json(response, { status: 404 });
    }

    const response = {
      success: true,
      message: 'Datos de heatmap obtenidos',
      data: mockHeatmapData,
    };
    return HttpResponse.json(response);
  }),

  http.get('/api/analyses/:id/report', ({ params }) => {
    const analysis = mockAnalyses.find((a) => a.id === Number(params.id));
    if (!analysis || analysis.status !== 'Completed') {
      const response: ApiResponse<null> = {
        success: false,
        message: `Análisis ${params.id} no encontrado o sin resultados`,
        data: null,
      };
      return HttpResponse.json(response, { status: 404 });
    }

    const response: ApiResponse<string> = {
      success: true,
      message: 'Report path',
      data: `/api/analyses/${params.id}/report.pdf`,
    };
    return HttpResponse.json(response);
  }),

  // Dashboard
  http.get('/api/dashboard/stats', () => {
    const response: ApiResponseOfDashboardStats = {
      success: true,
      message: 'Estadísticas del dashboard',
      data: mockDashboardStats,
    };
    return HttpResponse.json(response);
  }),
];
