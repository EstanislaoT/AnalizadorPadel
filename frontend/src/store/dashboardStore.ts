import { create } from 'zustand';
import { AnalizadorPadelApiService } from '../services/api/generated/services/AnalizadorPadelApiService';
import { DashboardStats } from '../services/api/generated/models/DashboardStats';
import { ApiResponseOfDashboardStats } from '../services/api/generated/models/ApiResponseOfDashboardStats';

interface DashboardStore {
  stats: DashboardStats | null;
  loading: boolean;
  error: string | null;
  fetchStats: () => Promise<void>;
  clearError: () => void;
  clearStats: () => void;
}

export const useDashboardStore = create<DashboardStore>((set) => ({
  stats: null,
  loading: false,
  error: null,

  fetchStats: async () => {
    set({ loading: true, error: null });
    try {
      const response: ApiResponseOfDashboardStats = await AnalizadorPadelApiService.getDashboardStats();
      if (response.success && response.data) {
        set({ stats: response.data as DashboardStats, loading: false });
      } else {
        set({ error: response.message || 'Error fetching stats', loading: false });
      }
    } catch (error: any) {
      set({ error: error.message || 'Error fetching stats', loading: false });
    }
  },

  clearError: () => set({ error: null }),

  clearStats: () => set({ stats: null, error: null }),
}));
