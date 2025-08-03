import axios, { AxiosInstance } from 'axios';

// API Types
export interface Feature {
  id: string;
  layer: number;
  activation: number;
  description: string;
  topTokens: Array<{
    token: string;
    strength: number;
  }>;
  frequency: number;
}

export interface Patch {
  id: string;
  featureId: string;
  name: string;
  isEnabled: boolean;
  strength: number;
  description: string;
}

export interface ModelStatus {
  status: 'idle' | 'loading' | 'ready' | 'error';
  modelName?: string;
  error?: string;
  memoryUsage?: number;
}

export interface SAEStatus {
  status: 'idle' | 'loading' | 'ready' | 'error';
  layerCount?: number;
  featureCount?: number;
  error?: string;
}

export interface FeatureDetail {
  id: string;
  layer: number;
  activation: number;
  description: string;
  topTokens: Array<{
    token: string;
    strength: number;
  }>;
  frequency: number;
  contextExamples?: string[];
}

export interface InferenceResult {
  inputText: string;
  tokenProbabilities: Array<{
    token: string;
    probability: number;
    logit: number;
  }>;
  latencyMs: number;
}

export interface LoadModelRequest {
  model_name: string;
}

export interface LoadSAERequest {
  layer_idx: number;
  saePath: string;
  activationsPath: string;
}

export interface CreatePatchRequest {
  featureId: string;
  name: string;
  strength: number;
  description?: string;
}

export interface UpdatePatchRequest {
  strength?: number;
  isEnabled?: boolean;
}

export interface InferenceRequest {
  text: string;
  maxTokens?: number;
}

export interface TrainSAERequest {
  layer_idx: number;
  activation_data_path: string;
  latent_dim?: number;
  sparsity_coef?: number;
  learning_rate?: number;
  max_epochs?: number;
  tied_weights?: boolean;
  activation_fn?: string;
  output_dir?: string;
}

export interface SAETrainingJob {
  job_id: string;
  status: 'starting' | 'training' | 'completed' | 'failed' | 'cancelled';
  layer_idx: number;
  output_dir: string;
  config: TrainSAERequest;
  progress: {
    current_epoch: number;
    total_epochs: number;
  };
  metrics: {
    train_loss: number[];
    reconstruction_loss: number[];
    sparsity_loss: number[];
  };
  created_at: string;
  completed_at?: string;
  failed_at?: string;
  cancelled_at?: string;
  model_path?: string;
  metadata_path?: string;
  error?: string;
}

// API Client Class
class APIClient {
  private client: AxiosInstance;
  private baseURL: string;

  constructor(baseURL: string = '') {
    // Handle relative URLs by using the current origin
    if (baseURL && !baseURL.startsWith('http://') && !baseURL.startsWith('https://')) {
      this.baseURL = baseURL;
    } else {
      this.baseURL = baseURL || (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8000');
    }
    
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 30000, // 30 second timeout
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add request/response interceptors for debugging
    this.client.interceptors.request.use((config) => {
      console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
      return config;
    });

    this.client.interceptors.response.use(
      (response) => {
        console.log(`API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error(`API Error: ${error.response?.status} ${error.config?.url}`, error.response?.data);
        return Promise.reject(error);
      }
    );
  }

  // Model Management
  async loadModel(request: LoadModelRequest): Promise<ModelStatus> {
    const response = await this.client.post('/load-model', request);
    return response.data;
  }

  async getModelStatus(): Promise<ModelStatus> {
    const response = await this.client.get('/model/status');
    return response.data;
  }

  // SAE Management
  async loadSAE(request: LoadSAERequest): Promise<SAEStatus> {
    const response = await this.client.post('/load-sae', request);
    return response.data;
  }

  async getSAEStatus(): Promise<SAEStatus> {
    const response = await this.client.get('/sae/status');
    return response.data;
  }

  // Feature Management
  async getFeatures(params: {
    layer?: number;
    limit?: number;
    offset?: number;
    search?: string;
    sortBy?: 'activation' | 'frequency' | 'layer';
    sortOrder?: 'asc' | 'desc';
  } = {}): Promise<{
    features: Feature[];
    total: number;
    hasMore: boolean;
  }> {
    const response = await this.client.get('/features', { params });
    return response.data;
  }

  async getFeatureDetail(featureId: string): Promise<FeatureDetail> {
    const response = await this.client.get(`/feature/${featureId}/details`);
    return response.data;
  }

  // Patch Management
  async createPatch(request: CreatePatchRequest): Promise<Patch> {
    const response = await this.client.post('/patch', request);
    return response.data;
  }

  async getPatches(): Promise<Patch[]> {
    const response = await this.client.get('/patches');
    return response.data;
  }

  async updatePatch(patchId: string, request: UpdatePatchRequest): Promise<Patch> {
    const response = await this.client.patch(`/patch/${patchId}`, request);
    return response.data;
  }

  async deletePatch(patchId: string): Promise<void> {
    await this.client.delete(`/patch/${patchId}`);
  }

  async togglePatch(patchId: string): Promise<Patch> {
    const response = await this.client.post(`/patch/${patchId}/toggle`);
    return response.data;
  }

  // Inference
  async runInference(request: InferenceRequest): Promise<InferenceResult> {
    const response = await this.client.post('/inference', request);
    return response.data;
  }

  // Export/Import
  async exportFeatures(featureIds?: string[]): Promise<Blob> {
    const response = await this.client.post('/export-features', 
      { featureIds }, 
      { responseType: 'blob' }
    );
    return response.data;
  }

  async exportPatches(patchIds?: string[]): Promise<Blob> {
    const response = await this.client.post('/export-patches', 
      { patchIds }, 
      { responseType: 'blob' }
    );
    return response.data;
  }

  async exportSAE(): Promise<Blob> {
    const response = await this.client.get('/export-sae', { responseType: 'blob' });
    return response.data;
  }

  // SAE Training
  async trainSAE(request: TrainSAERequest): Promise<{ success: boolean; job_id: string; message: string; output_dir: string }> {
    const response = await this.client.post('/sae/train', request);
    return response.data;
  }

  async getTrainingStatus(jobId: string): Promise<SAETrainingJob> {
    const response = await this.client.get(`/sae/training/status/${jobId}`);
    return response.data;
  }

  async listTrainingJobs(): Promise<{ jobs: SAETrainingJob[]; total: number }> {
    const response = await this.client.get('/sae/training/jobs');
    return response.data;
  }

  async cancelTraining(jobId: string): Promise<{ success: boolean; message: string }> {
    const response = await this.client.delete(`/sae/training/${jobId}`);
    return response.data;
  }

  async loadTrainedSAE(jobId: string): Promise<{ success: boolean; message: string; layer_idx: number; model_path: string }> {
    const response = await this.client.post('/sae/load-trained', { job_id: jobId });
    return response.data;
  }

  // Health Check
  async ping(): Promise<{ status: string; timestamp: number }> {
    const response = await this.client.get('/ping');
    return response.data;
  }

  // WebSocket URL
  getWebSocketURL(): string {
    if (typeof window !== 'undefined') {
      const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      // Use window.location.origin to avoid IP address issues
      const origin = window.location.origin;
      return `${origin.replace(/^http/, 'ws')}/ws`;
    } else {
      // Fallback for SSR
      return 'ws://localhost:8000/ws';
    }
  }
}

// Create default instance
const apiClient = new APIClient(
  import.meta.env.VITE_API_URL || ''
);

export default apiClient;
export { APIClient };