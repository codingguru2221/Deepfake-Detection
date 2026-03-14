import { create } from 'zustand';

export type Modality = 'image' | 'video' | 'audio' | 'multimodal';

export interface PredictionResult {
  prediction: 'real' | 'deepfake' | 'uncertain';
  prob_fake: number;
  confidence: number;
  details?: Record<string, any>;
  modalityScores?: Record<string, number>;
}

export interface CrawlerStatus {
  enabled: boolean;
  running: boolean;
  crawl_cycle_running?: boolean;
  service_running?: boolean;
  last_run: string | null;
  last_error: string | null;
  records: number;
  genuine_records?: number;
  output: string;
  log?: string;
  auto_train_enabled?: boolean;
  auto_train_min_records?: number;
}

export interface RuntimeTrainingStatus {
  running: boolean;
  last_run: string | null;
  last_error: string | null;
  last_result: {
    status: string;
    reason?: string | null;
    manifest_path?: string;
    calibrator_path?: string | null;
    user_labeled_count?: number;
    pseudo_count?: number;
    crawler_refs_count?: number;
    trainable_samples?: number;
    calibrator_accuracy?: number | null;
    calibrator_auc?: number | null;
  } | null;
}

export interface ModelStatus {
  label: string;
  parameter_count: number;
  parameter_summary: string;
  trained_data_points: number;
  accuracy: number | null;
  evaluated_samples: number;
  last_trained_at: string | null;
  successful_training_runs: number;
  model_available: boolean;
}

export interface HealthResponse {
  status: string;
  modelFiles: Record<string, boolean>;
  models: Record<string, ModelStatus>;
  datasetCrawler: CrawlerStatus;
  runtimeLearning: {
    enabled: boolean;
    training: RuntimeTrainingStatus;
    fullTraining?: {
      enabled: boolean;
      status: {
        running: boolean;
        last_run: string | null;
        last_error: string | null;
        last_result: Record<string, any> | null;
      };
    };
  };
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

  getHealth: async (baseUrl: string): Promise<HealthResponse> => {
    const res = await fetch(`${baseUrl}/health`, { method: 'GET' });
    if (!res.ok) throw new Error(`Failed to fetch health (${res.status})`);
    return await res.json();
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
  },

  getCrawlerStatus: async (baseUrl: string): Promise<CrawlerStatus> => {
    const res = await fetch(`${baseUrl}/crawler/status`, { method: 'GET' });
    if (!res.ok) throw new Error(`Failed to fetch crawler status (${res.status})`);
    return await res.json();
  },

  getCrawlerLogs: async (baseUrl: string, limit = 80): Promise<{ lines: string[]; runtimeLearning: string[] }> => {
    const res = await fetch(`${baseUrl}/crawler/logs?limit=${limit}`, { method: 'GET' });
    if (!res.ok) throw new Error(`Failed to fetch crawler logs (${res.status})`);
    return await res.json();
  },

  setCrawlerEnabled: async (baseUrl: string, enabled: boolean): Promise<{ enabled: boolean }> => {
    const res = await fetch(`${baseUrl}/crawler/control`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    if (!res.ok) throw new Error(`Failed to update crawler state (${res.status})`);
    return await res.json();
  },

  runCrawler: async (baseUrl: string): Promise<{ started: boolean; message: string }> => {
    const res = await fetch(`${baseUrl}/crawler/run`, { method: 'POST' });
    if (!res.ok) throw new Error(`Failed to run crawler (${res.status})`);
    return await res.json();
  },

  submitAccuracyFeedback: async (
    baseUrl: string,
    payload: { sample_id: string; actual_label: 'real' | 'deepfake'; rating?: number; comment?: string },
  ): Promise<{ status: string }> => {
    const res = await fetch(`${baseUrl}/feedback/accuracy`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `Feedback submit failed (${res.status})`);
    }
    return await res.json();
  },

  runRuntimeTraining: async (baseUrl: string, includePseudo = true): Promise<{ started: boolean; message?: string }> => {
    const res = await fetch(`${baseUrl}/train/runtime`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ include_pseudo: includePseudo }),
    });
    if (!res.ok) throw new Error(`Runtime training failed (${res.status})`);
    return await res.json();
  },

  getRuntimeTrainingStatus: async (baseUrl: string): Promise<RuntimeTrainingStatus> => {
    const res = await fetch(`${baseUrl}/train/runtime/status`, { method: 'GET' });
    if (!res.ok) throw new Error(`Failed to fetch training status (${res.status})`);
    return await res.json();
  },
};
