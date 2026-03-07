import { create } from 'zustand';

export type Modality = 'image' | 'video' | 'audio' | 'multimodal';

export interface PredictionResult {
  prediction: 'real' | 'deepfake';
  prob_fake: number;
  confidence: number;
  details?: Record<string, any>;
  modalityScores?: Record<string, number>;
}

export interface HistoryItem {
  id: string;
  filename: string;
  timestamp: string;
  modality: Modality;
  result: PredictionResult;
}

// Ensure default respects env vars in production
const defaultApiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

interface AppState {
  apiBaseUrl: string;
  setApiBaseUrl: (url: string) => void;
  history: HistoryItem[];
  addHistory: (item: HistoryItem) => void;
  clearHistory: () => void;
  settings: {
    fuseModalities: boolean;
    autoRun: boolean;
  };
  updateSettings: (settings: Partial<AppState['settings']>) => void;
}

export const useStore = create<AppState>((set) => ({
  apiBaseUrl: defaultApiUrl,
  setApiBaseUrl: (url) => set({ apiBaseUrl: url }),
  history: [],
  addHistory: (item) => set((state) => ({ history: [item, ...state.history] })),
  clearHistory: () => set({ history: [] }),
  settings: {
    fuseModalities: true,
    autoRun: false,
  },
  updateSettings: (newSettings) => set((state) => ({ settings: { ...state.settings, ...newSettings } })),
}));

// API Client
export const api = {
  checkHealth: async (baseUrl: string): Promise<boolean> => {
    try {
      const res = await fetch(`${baseUrl}/health`, { method: 'GET' });
      return res.ok;
    } catch {
      return false;
    }
  },

  infer: async (baseUrl: string, modality: Modality, file: File): Promise<PredictionResult> => {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${baseUrl}/infer/${modality}`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let detail = res.statusText;
      try {
        const payload = await res.json();
        detail = payload.detail || payload.message || detail;
      } catch {
        // keep default status text
      }
      throw new Error(`API error (${res.status}): ${detail}`);
    }

    return await res.json();
  }
};
