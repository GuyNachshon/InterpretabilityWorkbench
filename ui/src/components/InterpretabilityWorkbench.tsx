import React, { useState, useEffect, useMemo } from 'react';
import { Search, Settings, Play, Trash2, ToggleLeft, ToggleRight, Moon, Sun, Loader2, Activity, Cpu, Layers, Eye, Plus, BarChart3, Cloud, FileText, ZoomIn, ZoomOut, RotateCcw, HelpCircle, Database, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Drawer, DrawerContent, DrawerHeader, DrawerTitle, DrawerTrigger } from '@/components/ui/drawer';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { toast } from 'sonner';
import apiClient, { Feature, Patch, ModelStatus, SAEStatus } from '../services/api';
import type { FeatureDetail } from '../services/api';
import wsClient from '../services/websocket';

// Local UI Types (extending API types)
interface UIFeature extends Feature {
  // UI-specific properties can be added here
}

interface UIPatch extends Patch {
  // UI-specific properties can be added here
}

interface ProgressInfo {
  id: string;
  type: 'model_loading' | 'sae_loading' | 'sae_training' | 'feature_analysis' | 'patch_creation';
  status: 'starting' | 'in_progress' | 'completed' | 'failed';
  title: string;
  description: string;
  progress: number; // 0-100
  details?: {
    currentStep?: string;
    totalSteps?: number;
    currentStepNumber?: number;
    estimatedTime?: string;
    metrics?: Record<string, number>;
    trainingMode?: 'standard' | 'fast';  // New: training mode indicator
    speedMultiplier?: number;  // New: speed improvement indicator
  };
  startedAt: string;
  completedAt?: string;
  error?: string;
}

interface AppState {
  model: ModelStatus;
  sae: SAEStatus;
  features: UIFeature[];
  patches: UIPatch[];
  ui: { theme: 'light' | 'dark'; drawerOpen: boolean; wsConnected: boolean };
  selectedFeature: UIFeature | null;
  selectedFeatureDetail: FeatureDetail | null;
  searchQuery: string;
  layerFilter: string;
  sortBy: 'activation' | 'frequency' | 'layer';
  sortOrder: 'asc' | 'desc';
  pagination: { offset: number; limit: number; total: number; hasMore: boolean };
  loading: {
    features: boolean;
    model: boolean;
    sae: boolean;
    patches: boolean;
    inference: boolean;
  };
  errors: {
    model?: string;
    sae?: string;
    features?: string;
    patches?: string;
    websocket?: string;
  };
  progress: ProgressInfo[];
      activeJobs: {
      saeTraining: string[];  // job_ids of active training jobs
    },
    trainingConfig: {
      mode: 'standard' | 'fast';
      batchSize: number;
      numWorkers: number;
      maxSamples: number | null;
      mixedPrecision: boolean;
    }
}

// Initialize empty state - data will be loaded from API

// Store implementation using React state with API integration
const useAppStore = () => {
  const [state, setState] = useState<AppState>({
    model: { status: 'idle' },
    sae: { status: 'idle' },
    features: [],
    patches: [],
    ui: { theme: 'light', drawerOpen: false, wsConnected: false },
    selectedFeature: null,
    selectedFeatureDetail: null,
    searchQuery: '',
    layerFilter: 'all',
    sortBy: 'activation',
    sortOrder: 'desc',
    pagination: { offset: 0, limit: 100, total: 0, hasMore: false },
    loading: {
      features: false,
      model: false,
      sae: false,
      patches: false,
      inference: false
    },
    errors: {},
    progress: [],
    activeJobs: {
      saeTraining: []
    },
    trainingConfig: {
      mode: 'standard' as const,
      batchSize: 512,
      numWorkers: 8,
      maxSamples: null,
      mixedPrecision: true,
    }
  });

  const updateState = (updates: Partial<AppState>) => {
    setState(prev => ({ ...prev, ...updates }));
  };

  const updateUI = (updates: Partial<AppState['ui']>) => {
    setState(prev => ({ ...prev, ui: { ...prev.ui, ...updates } }));
  };

  const updateModel = (updates: Partial<AppState['model']>) => {
    setState(prev => ({ ...prev, model: { ...prev.model, ...updates } }));
  };

  const addPatch = (patch: Patch) => {
    setState(prev => ({ 
      ...prev, 
      patches: [...(Array.isArray(prev.patches) ? prev.patches : []), patch] 
    }));
  };

  // Progress tracking helpers
  const startProgress = (id: string, type: ProgressInfo['type'], title: string, description: string) => {
    const progress: ProgressInfo = {
      id,
      type,
      status: 'starting',
      title,
      description,
      progress: 0,
      startedAt: new Date().toISOString()
    };
    setState(prev => ({
      ...prev,
      progress: [...prev.progress.filter(p => p.id !== id), progress]
    }));
  };

  const updateProgress = (id: string, updates: Partial<ProgressInfo>) => {
    setState(prev => ({
      ...prev,
      progress: prev.progress.map(p => 
        p.id === id ? { ...p, ...updates } : p
      )
    }));
  };

  const completeProgress = (id: string, success: boolean = true, error?: string) => {
    setState(prev => ({
      ...prev,
      progress: prev.progress.map(p => 
        p.id === id ? {
          ...p,
          status: success ? 'completed' as const : 'failed' as const,
          progress: success ? 100 : p.progress,
          completedAt: new Date().toISOString(),
          error
        } : p
      )
    }));

    // Remove completed/failed progress after 5 seconds
    setTimeout(() => {
      setState(prev => ({
        ...prev,
        progress: prev.progress.filter(p => p.id !== id)
      }));
    }, 5000);
  };

  const removeProgress = (id: string) => {
    setState(prev => ({
      ...prev, 
      progress: prev.progress.filter(p => p.id !== id)
    }));
  };

  const addActiveJob = (jobId: string) => {
    setState(prev => ({
      ...prev,
      activeJobs: {
        ...prev.activeJobs,
        saeTraining: [...prev.activeJobs.saeTraining.filter(id => id !== jobId), jobId]
      }
    }));
  };

  const removeActiveJob = (jobId: string) => {
    setState(prev => ({
      ...prev,
      activeJobs: {
        ...prev.activeJobs,
        saeTraining: prev.activeJobs.saeTraining.filter(id => id !== jobId)
      }
    }));
  };

  const updatePatch = (id: string, updates: Partial<Patch>) => {
    setState(prev => ({
      ...prev,
      patches: Array.isArray(prev.patches) 
        ? prev.patches.map(p => p.id === id ? { ...p, ...updates } : p)
        : []
    }));
  };

  const removePatch = (id: string) => {
    setState(prev => ({ 
      ...prev, 
      patches: Array.isArray(prev.patches) ? prev.patches.filter(p => p.id !== id) : [] 
    }));
  };

  // API Integration Methods
  const loadModel = async (modelName: string) => {
    const progressId = `model-load-${Date.now()}`;
    
    try {
      updateLoading({ model: true });
      updateModel({ status: 'loading' });
      clearError('model');
      
      // Start progress tracking
      startProgress(progressId, 'model_loading', 'Loading Model', `Loading ${modelName}...`);
      updateProgress(progressId, { 
        status: 'in_progress', 
        progress: 20,
        details: { currentStep: 'Initializing model download...' }
      });
      
      const result = await apiClient.loadModel({ model_name: modelName });
      
      updateProgress(progressId, { 
        progress: 80,
        details: { currentStep: 'Setting up tokenizer...' }
      });
      
      // Simulate some additional loading time for progress display
      await new Promise(resolve => setTimeout(resolve, 500));
      
      updateModel(result);
      completeProgress(progressId, true);
      toast.success(`Model ${modelName} loaded successfully`);
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Failed to load model';
      updateModel({ status: 'error', error: errorMsg });
      updateError('model', errorMsg);
      completeProgress(progressId, false, errorMsg);
      toast.error(`Failed to load model: ${errorMsg}`);
    } finally {
      updateLoading({ model: false });
    }
  };

  const loadSAE = async (saePath: string, activationsPath: string, layerIdx: number = 6) => {
    try {
      updateLoading({ sae: true });
      updateSAE({ status: 'loading' });
      clearError('sae');
      
      const result = await apiClient.loadSAE({ 
        layer_idx: layerIdx,
        saePath, 
        activationsPath 
      });
      updateSAE(result);
      toast.success('SAE loaded successfully');
      
      // Auto-load features after SAE is loaded
      await loadFeatures();
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Failed to load SAE';
      updateSAE({ status: 'error', error: errorMsg });
      updateError('sae', errorMsg);
      toast.error(`Failed to load SAE: ${errorMsg}`);
    } finally {
      updateLoading({ sae: false });
    }
  };

  const loadFeatures = async (resetPagination = false) => {
    try {
      updateLoading({ features: true });
      clearError('features');
      
      const offset = resetPagination ? 0 : state.pagination.offset;
      const params = {
        offset,
        limit: state.pagination.limit,
        search: state.searchQuery || undefined,
        layer: state.layerFilter !== 'all' ? parseInt(state.layerFilter) : undefined,
        sortBy: state.sortBy,
        sortOrder: state.sortOrder
      };
      
      const result = await apiClient.getFeatures(params);
      
      if (resetPagination) {
        updateState({
          features: Array.isArray(result.features) ? result.features : [],
          pagination: {
            ...state.pagination,
            offset: 0,
            total: result.total,
            hasMore: result.hasMore
          }
        });
      } else {
        updateState({
          features: [...(Array.isArray(state.features) ? state.features : []), ...(Array.isArray(result.features) ? result.features : [])],
          pagination: {
            ...state.pagination,
            offset: offset + (Array.isArray(result.features) ? result.features.length : 0),
            total: result.total,
            hasMore: result.hasMore
          }
        });
      }
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Failed to load features';
      updateError('features', errorMsg);
      toast.error(`Failed to load features: ${errorMsg}`);
      // Set empty array on error
      updateState({ features: [] });
    } finally {
      updateLoading({ features: false });
    }
  };

  const loadFeatureDetail = async (featureId: string) => {
    try {
      const detail = await apiClient.getFeatureDetail(featureId);
      updateState({ selectedFeatureDetail: detail });
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Failed to load feature details';
      toast.error(`Failed to load feature details: ${errorMsg}`);
    }
  };

  const loadPatches = async () => {
    try {
      updateLoading({ patches: true });
      clearError('patches');
      
      const patches = await apiClient.getPatches();
      // Ensure patches is always an array
      const patchesArray = Array.isArray(patches) ? patches : [];
      updateState({ patches: patchesArray });
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Failed to load patches';
      updateError('patches', errorMsg);
      toast.error(`Failed to load patches: ${errorMsg}`);
      // Set empty array on error
      updateState({ patches: [] });
    } finally {
      updateLoading({ patches: false });
    }
  };

  const createPatch = async (featureId: string, name: string, strength: number, description?: string) => {
    try {
      const patch = await apiClient.createPatch({ featureId, name, strength, description });
      updateState({ patches: [...(Array.isArray(state.patches) ? state.patches : []), patch] });
      toast.success('Patch created successfully');
      return patch;
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Failed to create patch';
      toast.error(`Failed to create patch: ${errorMsg}`);
      throw error;
    }
  };

  const togglePatch = async (patchId: string) => {
    try {
      const updatedPatch = await apiClient.togglePatch(patchId);
      updatePatch(patchId, updatedPatch);
      toast.success(`Patch ${updatedPatch.isEnabled ? 'enabled' : 'disabled'}`);
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Failed to toggle patch';
      toast.error(`Failed to toggle patch: ${errorMsg}`);
    }
  };

  const deletePatch = async (patchId: string) => {
    try {
      await apiClient.deletePatch(patchId);
      removePatch(patchId);
      toast.success('Patch deleted successfully');
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Failed to delete patch';
      toast.error(`Failed to delete patch: ${errorMsg}`);
    }
  };

  const runInference = async (text: string) => {
    try {
      updateLoading({ inference: true });
      const result = await apiClient.runInference({ text });
      toast.success(`Inference completed in ${result.latencyMs}ms`);
      return result;
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Failed to run inference';
      toast.error(`Failed to run inference: ${errorMsg}`);
      throw error;
    } finally {
      updateLoading({ inference: false });
    }
  };

  const exportAllPatches = async () => {
    try {
      const blob = await apiClient.exportPatches();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'patches.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success('All patches exported successfully');
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Failed to export patches';
      toast.error(`Failed to export patches: ${errorMsg}`);
    }
  };

  const exportSAE = async () => {
    try {
      const blob = await apiClient.exportSAE();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'sae_export.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success('SAE exported successfully');
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Failed to export SAE';
      toast.error(`Failed to export SAE: ${errorMsg}`);
    }
  };

  // Helper update functions
  const updateLoading = (updates: Partial<AppState['loading']>) => {
    setState(prev => ({ ...prev, loading: { ...prev.loading, ...updates } }));
  };

  const updateSAE = (updates: Partial<SAEStatus>) => {
    setState(prev => ({ ...prev, sae: { ...prev.sae, ...updates } }));
  };

  const updateError = (key: keyof AppState['errors'], error: string) => {
    setState(prev => ({ ...prev, errors: { ...prev.errors, [key]: error } }));
  };

  const clearError = (key: keyof AppState['errors']) => {
    setState(prev => ({ ...prev, errors: { ...prev.errors, [key]: undefined } }));
  };

  // WebSocket setup
  useEffect(() => {
    const connectWebSocket = async () => {
      try {
        await wsClient.connect();
        updateUI({ wsConnected: true });
        
        // Subscribe to updates
        wsClient.subscribe('patch_updates');
        wsClient.subscribe('model_status');
        
        // Handle messages
        wsClient.on('message', (message) => {
          switch (message.type) {
            case 'patch_toggled':
              // Update patch state from WebSocket
              const patchData = message.data;
              updatePatch(patchData.patchId, { 
                isEnabled: patchData.action === 'enabled',
                ...(patchData.strength !== undefined && { strength: patchData.strength })
              });
              break;
            case 'model_status':
              updateModel(message.data);
              break;
            case 'training_progress':
              // Handle training progress updates
              const progressId = `sae_training_${message.job_id}`;
              const existingProgress = state.progress.find(p => p.id === progressId);
              
              if (existingProgress) {
                updateProgress(progressId, {
                  progress: message.progress || 0,
                  status: message.status === 'training' ? 'in_progress' : 
                         message.status === 'completed' ? 'completed' : 'failed',
                  details: {
                    currentStep: `Epoch ${message.epoch || 0}/${message.total_epochs || 0}`,
                    totalSteps: message.total_epochs || 0,
                    currentStepNumber: message.epoch || 0,
                    estimatedTime: message.estimated_remaining ? 
                      `${Math.round(message.estimated_remaining / 60)}m ${Math.round(message.estimated_remaining % 60)}s` : undefined,
                    metrics: message.metrics || {}
                  }
                });
              }
              break;
            case 'training_completed':
              // Handle training completion
              const completedProgressId = `sae_training_${message.job_id}`;
              completeProgress(completedProgressId, true);
              removeActiveJob(message.job_id);
              toast.success('SAE training completed successfully!');
              break;
            case 'training_failed':
              // Handle training failure
              const failedProgressId = `sae_training_${message.job_id}`;
              completeProgress(failedProgressId, false, message.error);
              removeActiveJob(message.job_id);
              toast.error(`SAE training failed: ${message.error}`);
              break;
          }
        });
        
        wsClient.on('close', () => {
          updateUI({ wsConnected: false });
        });
        
      } catch (error) {
        updateError('websocket', 'Failed to connect to WebSocket');
        updateUI({ wsConnected: false });
      }
    };

    connectWebSocket();

    return () => {
      wsClient.disconnect();
    };
  }, []);

  // Load initial data
  useEffect(() => {
    // Check model and SAE status on startup
    apiClient.getModelStatus().then(updateModel).catch(() => {});
    apiClient.getSAEStatus().then((saeStatus) => {
      updateSAE(saeStatus);
      // Auto-load features if SAE is ready
      if (saeStatus.status === 'ready') {
        loadFeatures();
      }
    }).catch(() => {});
    loadPatches();
  }, []);

  return {
    state,
    updateState,
    updateUI,
    updateModel,
    updateSAE,
    addPatch,
    updatePatch,
    removePatch,
    loadFeatures,
    loadFeatureDetail,
    loadModel,
    loadSAE,
    loadPatches,
    createPatch,
    togglePatch,
    deletePatch,
    runInference,
    exportAllPatches,
    exportSAE,
    // Progress tracking
    startProgress,
    updateProgress,
    completeProgress,
    removeProgress,
    addActiveJob,
    removeActiveJob
  };
};

// Helper Components
const TooltipLabel: React.FC<{ label: string; tooltip: string; children?: React.ReactNode }> = ({ label, tooltip, children }) => (
  <div className="flex items-center gap-2">
    <Label>{label}</Label>
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="sm" className="h-auto p-0">
          <HelpCircle className="w-3 h-3 text-muted-foreground hover:text-foreground" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80 text-sm">
        <p>{tooltip}</p>
        {children}
      </PopoverContent>
    </Popover>
  </div>
);

// Header Component
const Header: React.FC<{ store: ReturnType<typeof useAppStore> }> = ({ store }) => {
  const { state, updateState, updateUI, loadModel, loadSAE, startProgress, updateProgress, completeProgress, addActiveJob, removeProgress } = store;
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [trainSAEOpen, setTrainSAEOpen] = useState(false);
  const [modelName, setModelName] = useState('openai-community/gpt2');
  const [saePath, setSaePath] = useState('');
  const [activationsPath, setActivationsPath] = useState('');
  
  // Calculate middle layer based on model (default to layer 6 for 12-layer models)
  const getMiddleLayer = (modelName?: string) => {
    if (!modelName) return 6;
    
    // Common model layer counts
    const layerCounts: Record<string, number> = {
      'gpt2': 12,
      'distilgpt2': 6,
      'microsoft/DialoGPT-small': 12,
      'microsoft/DialoGPT-medium': 24,
      'microsoft/DialoGPT-large': 36,
      'gpt2-medium': 24,
      'gpt2-large': 36,
      'gpt2-xl': 48,
    };
    
    // Find matching model pattern
    for (const [pattern, layers] of Object.entries(layerCounts)) {
      if (modelName.toLowerCase().includes(pattern.toLowerCase())) {
        return Math.floor(layers / 2);
      }
    }
    
    // Default assumption for most transformer models
    return 6; // Assuming 12-layer model
  };

  // SAE Training form state  
  const [saeTrainingConfig, setSaeTrainingConfig] = useState({
    layerIdx: getMiddleLayer(state.model.modelName),
    activationDataPath: `${state.model.modelName ? state.model.modelName.replace('/', '_') : 'model'}_layer_${getMiddleLayer(state.model.modelName)}_activations.parquet`,
    latentDim: 16384,
    sparsityCoef: 0.001,
    learningRate: 0.001,
    maxEpochs: 100,
    tiedWeights: true,
    activationFn: 'relu',
    outputDir: '',
    dataset: 'openwebtext'
  });

  // Update default paths and layer when model changes
  useEffect(() => {
    if (state.model.modelName) {
      const middleLayer = getMiddleLayer(state.model.modelName);
      setSaeTrainingConfig(prev => ({
        ...prev,
        layerIdx: middleLayer,
                    activationDataPath: `${state.model.modelName?.replace('/', '_') || 'model'}_layer_${middleLayer}_activations.parquet`
      }));
    }
  }, [state.model.modelName]);

  const toggleTheme = () => {
    const newTheme = state.ui.theme === 'light' ? 'dark' : 'light';
    updateUI({ theme: newTheme });
    document.body.className = newTheme === 'dark' ? 'dark' : '';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready': return 'bg-green-500';
      case 'loading': return 'bg-yellow-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const handleLoadModel = () => {
    if (modelName.trim()) {
      loadModel(modelName.trim());
    }
  };

  const handleLoadSAE = () => {
    if (saePath.trim() && activationsPath.trim()) {
      loadSAE(saePath.trim(), activationsPath.trim());
    } else {
      toast.error('Please provide both SAE path and activations path');
    }
  };

  const handleTrainSAE = async (mode: 'standard' | 'fast' = 'standard') => {
    if (!saeTrainingConfig.activationDataPath.trim()) {
      toast.error('Please provide activation data path');
      return;
    }

    const progressId = `sae-train-${Date.now()}`;
    const isFastMode = mode === 'fast';

    try {
      // Start progress tracking with mode indicator
      const speedMultiplier = isFastMode ? 5 : 2;
      startProgress(progressId, 'sae_training', `Training SAE (${mode})`, 
        `${isFastMode ? 'Ultra-fast' : 'Optimized'} SAE training for layer ${saeTrainingConfig.layerIdx}...`);
      
      updateProgress(progressId, { 
        status: 'in_progress', 
        progress: 10,
        details: { 
          currentStep: `Starting ${mode} SAE training...`,
          trainingMode: mode,
          speedMultiplier
        }
      });

      // Use appropriate endpoint based on mode
      const endpoint = isFastMode ? apiClient.trainSAEFast : apiClient.trainSAE;
      
      const response = await endpoint({
        layer_idx: saeTrainingConfig.layerIdx,
        activation_data_path: saeTrainingConfig.activationDataPath,
        latent_dim: isFastMode ? Math.min(saeTrainingConfig.latentDim, 8192) : saeTrainingConfig.latentDim,
        sparsity_coef: saeTrainingConfig.sparsityCoef,
        learning_rate: isFastMode ? 2e-3 : saeTrainingConfig.learningRate,
        max_epochs: isFastMode ? Math.min(saeTrainingConfig.maxEpochs, 50) : saeTrainingConfig.maxEpochs,
        tied_weights: saeTrainingConfig.tiedWeights,
        activation_fn: saeTrainingConfig.activationFn,
        output_dir: saeTrainingConfig.outputDir || undefined,
        batch_size: isFastMode ? 1024 : 512,
        num_workers: isFastMode ? 12 : 8,
        max_samples: isFastMode ? 50000 : null
      });

      if (response.success) {
        addActiveJob(response.job_id);
        
        // Update progress ID to match WebSocket messages
        const wsProgressId = `sae_training_${response.job_id}`;
        
        // Update the progress tracking to use WebSocket updates
        updateProgress(progressId, { 
          progress: 30,
          details: { 
            currentStep: `${mode} training started (Job ID: ${response.job_id})`,
            trainingMode: mode,
            speedMultiplier
          }
        });
        
        // Create a new progress entry for WebSocket updates
        startProgress(wsProgressId, 'sae_training', `Training SAE (${mode})`, 
          `${isFastMode ? 'Ultra-fast' : 'Optimized'} SAE training for layer ${saeTrainingConfig.layerIdx}...`);
        
        toast.success(`${mode.charAt(0).toUpperCase() + mode.slice(1)} SAE training started! Job ID: ${response.job_id}`);
        setTrainSAEOpen(false);
        
        // Remove the initial progress entry since WebSocket will handle updates
        setTimeout(() => removeProgress(progressId), 2000);
      }
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Failed to start SAE training';
      completeProgress(progressId, false, errorMsg);
      toast.error(`Failed to start ${mode} SAE training: ${errorMsg}`);
    }
  };

  return (
    <header className="h-16 border-b border-border bg-background px-6 flex items-center justify-between">
      <div className="flex items-center gap-6">
        <h1 className="text-xl font-semibold text-foreground">InterpretabilityWorkbench</h1>
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${getStatusColor(state.model.status)}`} />
            Model: {state.model.status}
            {state.model.modelName && ` (${state.model.modelName})`}
          </Badge>
          <Badge variant="outline" className="flex items-center gap-2">
            <Layers className="w-3 h-3" />
            SAE: {state.sae.status}
            {state.sae.featureCount && ` (${state.sae.featureCount} features)`}
          </Badge>
          <Badge variant="outline" className="flex items-center gap-2">
            <Activity className="w-3 h-3" />
            Patches: {Array.isArray(state.patches) ? state.patches.filter(p => p.isEnabled).length : 0}/{Array.isArray(state.patches) ? state.patches.length : 0}
          </Badge>
          {!state.ui.wsConnected && (
            <Badge variant="destructive" className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              WS Disconnected
            </Badge>
          )}
        </div>
      </div>
      
      <div className="flex items-center gap-3">
        {state.model.status === 'idle' && (
          <Button onClick={handleLoadModel} size="sm" disabled={state.loading.model}>
            {state.loading.model ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Cpu className="w-4 h-4 mr-2" />
            )}
            {state.loading.model ? 'Loading...' : 'Load Model'}
          </Button>
        )}
        
        {state.model.status === 'ready' && state.sae.status === 'idle' && (
          <div className="flex gap-2">
            <Button onClick={handleLoadSAE} size="sm" disabled={state.loading.sae}>
              {state.loading.sae ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Layers className="w-4 h-4 mr-2" />
              )}
              {state.loading.sae ? 'Loading SAE...' : 'Load SAE'}
            </Button>
            <Button onClick={() => setTrainSAEOpen(true)} size="sm" variant="outline">
              <Plus className="w-4 h-4 mr-2" />
              Train SAE
            </Button>
          </div>
        )}
        
        <Button variant="ghost" size="sm" onClick={toggleTheme}>
          {state.ui.theme === 'light' ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
        </Button>
        
        <Dialog open={settingsOpen} onOpenChange={setSettingsOpen}>
          <DialogTrigger asChild>
            <Button variant="ghost" size="sm">
              <Settings className="w-4 h-4" />
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Settings</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label>Dark Mode</Label>
                <Switch checked={state.ui.theme === 'dark'} onCheckedChange={toggleTheme} />
              </div>
              <Separator />
              <div className="space-y-2">
                <Label>Model Name</Label>
                <Input
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder="e.g., openai-community/gpt2"
                />
              </div>
              <Separator />
              <div className="space-y-2">
                <Label>SAE Path</Label>
                <Input
                  value={saePath}
                  onChange={(e) => setSaePath(e.target.value)}
                  placeholder="Path to SAE weights file"
                />
              </div>
              <div className="space-y-2">
                <Label>Activations Path</Label>
                <Input
                  value={activationsPath}
                  onChange={(e) => setActivationsPath(e.target.value)}
                  placeholder="Path to activations parquet file"
                />
              </div>
              <Separator />
              <div className="space-y-2">
                <Label>Connection Status</Label>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${state.ui.wsConnected ? 'bg-green-500' : 'bg-gray-400'}`} />
                  <span className="text-sm">
                    WebSocket: {state.ui.wsConnected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
              </div>
            </div>
          </DialogContent>
        </Dialog>

        <Dialog open={trainSAEOpen} onOpenChange={setTrainSAEOpen}>
          <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Train Sparse Autoencoder</DialogTitle>
              <p className="text-sm text-muted-foreground">
                Configure and train a sparse autoencoder to learn interpretable features from model activations.
              </p>
            </DialogHeader>
            <div className="space-y-6">
              {/* Model Configuration */}
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Model Configuration</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <TooltipLabel 
                      label="Layer Index" 
                      tooltip="Which transformer layer to extract activations from. Different layers capture different types of features - earlier layers often capture syntax, later layers capture semantics."
                    />
                    <Input
                      type="number"
                      value={saeTrainingConfig.layerIdx}
                      onChange={(e) => {
                        const newLayerIdx = parseInt(e.target.value) || 0;
                        setSaeTrainingConfig(prev => ({ 
                          ...prev, 
                          layerIdx: newLayerIdx,
                          activationDataPath: state.model.modelName 
                            ? `${state.model.modelName.replace('/', '_')}_layer_${newLayerIdx}_activations.parquet`
                            : `model_layer_${newLayerIdx}_activations.parquet`
                        }));
                      }}
                      placeholder="10"
                      min="0"
                    />
                  </div>
                  <div className="space-y-2">
                    <TooltipLabel 
                      label="Latent Dimensions" 
                      tooltip="Number of features the SAE will learn. Higher values capture more fine-grained features but require more data and compute. Typical values: 8192-65536."
                    />
                    <Input
                      type="number"
                      value={saeTrainingConfig.latentDim}
                      onChange={(e) => setSaeTrainingConfig(prev => ({ ...prev, latentDim: parseInt(e.target.value) || 16384 }))}
                      placeholder="16384"
                      min="512"
                      max="131072"
                      step="512"
                    />
                  </div>
                </div>
              </div>

              {/* Data Configuration */}
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Data Configuration</h3>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <TooltipLabel 
                      label="Dataset" 
                      tooltip="Dataset to use for extracting activations. OpenWebText provides diverse internet text, WikiText is more structured encyclopedia text, and The Pile contains varied sources."
                    />
                    <Select 
                      value={saeTrainingConfig.dataset} 
                      onValueChange={(value) => setSaeTrainingConfig(prev => ({ ...prev, dataset: value }))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="openwebtext">OpenWebText (Diverse web text)</SelectItem>
                        <SelectItem value="wikitext">WikiText (Encyclopedia articles)</SelectItem>
                        <SelectItem value="the_pile">The Pile (Mixed sources)</SelectItem>
                        <SelectItem value="lmsys-chat-1m">LMSYS-Chat-1M</SelectItem>
                        <SelectItem value="custom">Custom Dataset</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="space-y-2">
                    <TooltipLabel 
                      label="Activation Data Path" 
                      tooltip="Path to the parquet file containing pre-extracted activations. If this file doesn't exist, you'll need to generate it first using the trace command or the UI."
                    >
                      <div className="mt-1 text-xs text-muted-foreground">
                        <p>To generate activations: <code>python -m interpretability_workbench.cli trace --model {state.model.modelName || 'your-model'} --layer {saeTrainingConfig.layerIdx} --out {saeTrainingConfig.activationDataPath}</code></p>
                      </div>
                    </TooltipLabel>
                    <div className="flex gap-2">
                      <Input
                        value={saeTrainingConfig.activationDataPath}
                        onChange={(e) => setSaeTrainingConfig(prev => ({ ...prev, activationDataPath: e.target.value }))}
                        placeholder="Path to activation data (.parquet file)"
                        className="flex-1"
                      />
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={() => {
                          // Generate default path
                          const defaultPath = state.model.modelName 
                            ? `${state.model.modelName.replace('/', '_')}_layer_${saeTrainingConfig.layerIdx}_activations.parquet`
                            : `model_layer_${saeTrainingConfig.layerIdx}_activations.parquet`;
                          setSaeTrainingConfig(prev => ({ ...prev, activationDataPath: defaultPath }));
                        }}
                      >
                        Default
                      </Button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Training Parameters */}
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Training Parameters</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <TooltipLabel 
                      label="Sparsity Coefficient (β)" 
                      tooltip="Controls how sparse the learned features are. Higher values encourage fewer active features per input. Typical range: 1e-4 to 1e-2."
                    />
                    <Input
                      type="number"
                      step="0.0001"
                      value={saeTrainingConfig.sparsityCoef}
                      onChange={(e) => setSaeTrainingConfig(prev => ({ ...prev, sparsityCoef: parseFloat(e.target.value) || 0.001 }))}
                      placeholder="0.001"
                      min="0.0001"
                      max="0.1"
                    />
                  </div>
                  <div className="space-y-2">
                    <TooltipLabel 
                      label="Learning Rate" 
                      tooltip="Step size for gradient updates. Start with 1e-3 and adjust based on training curves. Lower values are more stable but slower."
                    />
                    <Input
                      type="number"
                      step="0.0001"
                      value={saeTrainingConfig.learningRate}
                      onChange={(e) => setSaeTrainingConfig(prev => ({ ...prev, learningRate: parseFloat(e.target.value) || 0.001 }))}
                      placeholder="0.001"
                      min="0.00001"
                      max="0.01"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <TooltipLabel 
                      label="Max Epochs" 
                      tooltip="Maximum number of training epochs. SAEs typically converge within 50-200 epochs depending on data size and complexity."
                    />
                    <Input
                      type="number"
                      value={saeTrainingConfig.maxEpochs}
                      onChange={(e) => setSaeTrainingConfig(prev => ({ ...prev, maxEpochs: parseInt(e.target.value) || 100 }))}
                      placeholder="100"
                      min="1"
                      max="1000"
                    />
                  </div>
                  <div className="space-y-2">
                    <TooltipLabel 
                      label="Activation Function" 
                      tooltip="Nonlinearity applied to latent features. ReLU is standard and ensures true sparsity (exact zeros). GELU is smoother but may not achieve true sparsity."
                    />
                    <Select value={saeTrainingConfig.activationFn} onValueChange={(value) => setSaeTrainingConfig(prev => ({ ...prev, activationFn: value }))}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="relu">ReLU (Recommended)</SelectItem>
                        <SelectItem value="gelu">GELU</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              {/* Advanced Options */}
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Advanced Options</h3>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <TooltipLabel 
                      label="Output Directory" 
                      tooltip="Directory to save trained model and metadata. Leave empty for auto-generated timestamped directory."
                    />
                    <Input
                      value={saeTrainingConfig.outputDir}
                      onChange={(e) => setSaeTrainingConfig(prev => ({ ...prev, outputDir: e.target.value }))}
                      placeholder="Leave empty for auto-generated directory"
                    />
                  </div>

                  <div className="flex items-center space-x-2">
                    <Switch
                      checked={saeTrainingConfig.tiedWeights}
                      onCheckedChange={(checked) => setSaeTrainingConfig(prev => ({ ...prev, tiedWeights: checked }))}
                    />
                    <TooltipLabel 
                      label="Tied Weights" 
                      tooltip="When enabled, the decoder weights are the transpose of encoder weights (W_dec = W_enc^T). This reduces parameters and often improves performance."
                    />
                  </div>
                </div>
              </div>

              <Separator />

              {/* Training Mode Selection */}
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Training Mode</h3>
                <div className="grid grid-cols-2 gap-4">
                  <Card className={`p-4 cursor-pointer transition-all ${
                    state.trainingConfig.mode === 'standard' ? 'ring-2 ring-primary bg-primary/5' : 'hover:bg-muted/50'
                  }`} onClick={() => updateState({ 
                    trainingConfig: { ...state.trainingConfig, mode: 'standard' }
                  })}>
                    <div className="flex items-center gap-2 mb-2">
                      <Play className="w-4 h-4" />
                      <h4 className="font-medium">Standard Training</h4>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">
                      Optimized training with balanced speed and quality
                    </p>
                    <div className="space-y-1 text-xs">
                      <div>• 2-4x faster than baseline</div>
                      <div>• Mixed precision (16-bit)</div>
                      <div>• Optimized data loading</div>
                      <div>• Better convergence</div>
                    </div>
                  </Card>
                  
                  <Card className={`p-4 cursor-pointer transition-all ${
                    state.trainingConfig.mode === 'fast' ? 'ring-2 ring-primary bg-primary/5' : 'hover:bg-muted/50'
                  }`} onClick={() => updateState({ 
                    trainingConfig: { ...state.trainingConfig, mode: 'fast' }
                  })}>
                    <div className="flex items-center gap-2 mb-2">
                      <Activity className="w-4 h-4" />
                      <h4 className="font-medium">Ultra-Fast Training</h4>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">
                      Maximum speed for quick experimentation
                    </p>
                    <div className="space-y-1 text-xs">
                      <div>• 5-8x faster than baseline</div>
                      <div>• Limited samples (50k)</div>
                      <div>• Smaller latent dimension</div>
                      <div>• Aggressive optimizations</div>
                    </div>
                  </Card>
                </div>
              </div>

              <Separator />

              <div className="flex justify-end gap-3">
                <Button variant="outline" onClick={() => setTrainSAEOpen(false)}>
                  Cancel
                </Button>
                <div className="flex gap-2">
                  <Button 
                    onClick={() => handleTrainSAE('standard')} 
                    disabled={!saeTrainingConfig.activationDataPath.trim()}
                    variant={state.trainingConfig.mode === 'standard' ? 'default' : 'outline'}
                  >
                    <Play className="w-4 h-4 mr-2" />
                    Standard Train
                  </Button>
                  <Button 
                    onClick={() => handleTrainSAE('fast')} 
                    disabled={!saeTrainingConfig.activationDataPath.trim()}
                    variant={state.trainingConfig.mode === 'fast' ? 'default' : 'outline'}
                  >
                    <Activity className="w-4 h-4 mr-2" />
                    Fast Train
                  </Button>
                </div>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </header>
  );
};

// Feature Table Component
const FeatureTable: React.FC<{ store: ReturnType<typeof useAppStore> }> = ({ store }) => {
  const { state, updateState } = store;

  const filteredFeatures = useMemo(() => {
    let filtered = Array.isArray(state.features) ? state.features : [];
    
    if (state.searchQuery) {
      filtered = filtered.filter(f => 
        f.description.toLowerCase().includes(state.searchQuery.toLowerCase()) ||
        f.topTokens.some(tokenData => tokenData.token.toLowerCase().includes(state.searchQuery.toLowerCase()))
      );
    }
    
    if (state.layerFilter !== 'all') {
      filtered = filtered.filter(f => f.layer.toString() === state.layerFilter);
    }
    
    filtered.sort((a, b) => {
      const aVal = a[state.sortBy];
      const bVal = b[state.sortBy];
      const multiplier = state.sortOrder === 'asc' ? 1 : -1;
      return (aVal < bVal ? -1 : aVal > bVal ? 1 : 0) * multiplier;
    });
    
    return filtered;
  }, [state.features, state.searchQuery, state.layerFilter, state.sortBy, state.sortOrder]);



  const createPatch = (feature: Feature) => {
    const patch: Patch = {
      id: `patch-${Date.now()}`,
      featureId: feature.id,
      name: `Patch ${feature.description.slice(0, 20)}...`,
      isEnabled: false,
      strength: 1.0,
      description: `Patch for ${feature.description}`
    };
    store.addPatch(patch);
    toast.success('Patch created successfully');
  };

  return (
    <div className="w-80 border-r border-border bg-background p-4 flex flex-col h-full">
      <div className="space-y-4 mb-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search features..."
            value={state.searchQuery}
            onChange={(e) => updateState({ searchQuery: e.target.value })}
            className="pl-10"
          />
        </div>
        
        <Select value={state.layerFilter} onValueChange={(value) => updateState({ layerFilter: value })}>
          <SelectTrigger>
            <SelectValue placeholder="Filter by layer" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Layers</SelectItem>
            {Array.from(new Set((Array.isArray(state.features) ? state.features : []).map(f => f.layer))).map(layer => (
              <SelectItem key={layer} value={layer.toString()}>Layer {layer}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex-1 overflow-auto">
        <div className="space-y-2">
          {filteredFeatures.map((feature) => (
            <Card 
              key={feature.id} 
              className={`p-3 cursor-pointer transition-colors hover:bg-muted/50 ${
                state.selectedFeature?.id === feature.id ? 'ring-2 ring-blue-500' : ''
              }`}
              onClick={() => updateState({ selectedFeature: feature })}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="text-xs">L{feature.layer}</Badge>
                  <span className="text-sm font-medium">{(feature.activation * 100).toFixed(1)}%</span>
                </div>
                <div className="flex gap-1">
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={(e) => {
                      e.stopPropagation();
                      updateState({ selectedFeature: feature });
                    }}
                  >
                    <Eye className="w-3 h-3" />
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={(e) => {
                      e.stopPropagation();
                      createPatch(feature);
                    }}
                  >
                    <Plus className="w-3 h-3" />
                  </Button>
                </div>
              </div>
              <p className="text-xs text-muted-foreground line-clamp-2">{feature.description}</p>
              <div className="flex flex-wrap gap-1 mt-2">
                {feature.topTokens.slice(0, 3).map((tokenData, idx) => (
                  <Badge key={idx} variant="outline" className="text-xs">{tokenData.token}</Badge>
                ))}
                {feature.topTokens.length > 3 && (
                  <Badge variant="outline" className="text-xs">+{feature.topTokens.length - 3}</Badge>
                )}
              </div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
};

// Feature Detail Component
const FeatureDetail: React.FC<{ store: ReturnType<typeof useAppStore> }> = ({ store }) => {
  const { state, createPatch } = store;
  const feature = state.selectedFeature;
  const featureDetail = state.selectedFeatureDetail;

  if (!feature) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted-foreground">
        <div className="text-center">
          <Layers className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>Select a feature to view details</p>
        </div>
      </div>
    );
  }

  const handleCreatePatch = async () => {
    if (!feature) return;
    
    try {
      const name = `Patch ${feature.description.slice(0, 20)}...`;
      const description = `Patch for ${feature.description}`;
      await createPatch(feature.id, name, 1.0, description);
    } catch (error) {
      // Error handling is done in the store method
    }
  };

  const handleExportFeature = async (feature?: UIFeature) => {
    if (!feature) return;
    
    try {
      const blob = await apiClient.exportFeatures([feature.id]);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `feature_${feature.id}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success('Feature exported successfully');
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Failed to export feature';
      toast.error(`Failed to export feature: ${errorMsg}`);
    }
  };

  return (
    <div className="flex-1 p-6">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-2xl font-semibold mb-2">{feature.description}</h2>
            <div className="flex items-center gap-4">
              <Badge>Layer {feature.layer}</Badge>
              <span className="text-sm text-muted-foreground">
                Activation: {(feature.activation * 100).toFixed(1)}%
              </span>
              <span className="text-sm text-muted-foreground">
                Frequency: {(feature.frequency * 100).toFixed(1)}%
              </span>
            </div>
          </div>
          <div className="flex gap-2">
            <Button onClick={handleCreatePatch} disabled={!feature}>
              <Plus className="w-4 h-4 mr-2" />
              Create Patch
            </Button>
            <Button variant="outline" onClick={() => handleExportFeature(feature)}>
              Export Feature
            </Button>
          </div>
        </div>
      </div>

      <Tabs defaultValue="tokens" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="tokens" className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            Top Tokens
          </TabsTrigger>
          <TabsTrigger value="cloud" className="flex items-center gap-2">
            <Cloud className="w-4 h-4" />
            Token Cloud
          </TabsTrigger>
          <TabsTrigger value="context" className="flex items-center gap-2">
            <FileText className="w-4 h-4" />
            Context
          </TabsTrigger>
        </TabsList>

        <TabsContent value="tokens" className="mt-6">
          <Card className="p-6">
            <h3 className="text-lg font-medium mb-4">Top Activating Tokens</h3>
            {featureDetail ? (
              <div className="space-y-3">
                {featureDetail.topTokens.map((tokenData, idx) => (
                  <div key={idx} className="flex items-center gap-3">
                    <div className="w-16 text-sm text-muted-foreground">#{idx + 1}</div>
                    <div className="flex-1 bg-muted rounded-lg p-3">
                      <code className="text-sm font-mono">{tokenData.token}</code>
                    </div>
                    <div className="w-20 text-sm text-right">
                      {tokenData.strength.toFixed(3)}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin mr-2" />
                <span>Loading token details...</span>
              </div>
            )}
          </Card>
        </TabsContent>

        <TabsContent value="cloud" className="mt-6">
          <Card className="p-6">
            <h3 className="text-lg font-medium mb-4">Token Cloud Visualization</h3>
            {featureDetail ? (
              <div className="flex flex-wrap gap-2">
                {featureDetail.topTokens.map((tokenData, idx) => {
                  const normalizedStrength = Math.min(Math.max(tokenData.strength, 0.3), 1.0);
                  return (
                    <Badge 
                      key={idx} 
                      variant="secondary" 
                      className="text-sm px-3 py-1"
                      style={{ 
                        fontSize: `${0.8 + normalizedStrength * 0.6}rem`,
                        opacity: 0.6 + normalizedStrength * 0.4
                      }}
                    >
                      {tokenData.token}
                    </Badge>
                  );
                })}
              </div>
            ) : (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin mr-2" />
                <span>Loading token cloud...</span>
              </div>
            )}
          </Card>
        </TabsContent>

        <TabsContent value="context" className="mt-6">
          <Card className="p-6">
            <h3 className="text-lg font-medium mb-4">Context Examples</h3>
            {featureDetail && featureDetail.contextExamples ? (
              <div className="space-y-4">
                {featureDetail.contextExamples.map((example, idx) => (
                  <div key={idx} className="p-4 bg-muted rounded-lg">
                    <p className="text-sm" dangerouslySetInnerHTML={{ __html: example }} />
                  </div>
                ))}
              </div>
            ) : (
              <div className="space-y-4">
                <div className="p-4 bg-muted rounded-lg">
                  <div className="flex items-center justify-center py-4">
                    <span className="text-sm text-muted-foreground">No context examples available</span>
                  </div>
                </div>
              </div>
            )}
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

// Patch Console Component
const PatchConsole: React.FC<{ store: ReturnType<typeof useAppStore> }> = ({ store }) => {
  const { state, updatePatch, removePatch } = store;
  const [inferenceText, setInferenceText] = useState('The quick brown fox jumps over the lazy dog.');
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<Array<{ token: string; original: number; patched: number }>>([]);

  const togglePatch = (id: string) => {
    const patches = Array.isArray(state.patches) ? state.patches : [];
    const patch = patches.find(p => p.id === id);
    if (patch) {
      updatePatch(id, { isEnabled: !patch.isEnabled });
      toast.success(`Patch ${patch.isEnabled ? 'disabled' : 'enabled'}`);
    }
  };

  const runInference = async () => {
    setIsRunning(true);
    try {
      const response = await fetch('/api/inference', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: inferenceText,
          max_length: 50
        })
      });

      if (!response.ok) {
        throw new Error(`Inference failed: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Transform the data to match our expected format
      const transformedResults = data.original.map((item: any, index: number) => ({
        token: item.token,
        original: item.probability,
        patched: data.patched[index]?.probability || 0
      }));
      
      setResults(transformedResults);
      toast.success('Inference completed');
    } catch (error) {
      console.error('Inference error:', error);
      toast.error(`Inference failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="w-80 border-l border-border bg-background p-4 flex flex-col h-full">
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-4">Active Patches</h3>
        <div className="space-y-3 max-h-60 overflow-auto">
          {!Array.isArray(state.patches) || state.patches.length === 0 ? (
            <p className="text-sm text-muted-foreground">No patches created yet</p>
          ) : (
            state.patches.map((patch) => (
              <Card key={patch.id} className="p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium truncate">{patch.name}</span>
                  <div className="flex items-center gap-2">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => togglePatch(patch.id)}
                    >
                      {patch.isEnabled ? 
                        <ToggleRight className="w-4 h-4 text-green-500" /> : 
                        <ToggleLeft className="w-4 h-4 text-gray-400" />
                      }
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => removePatch(patch.id)}
                    >
                      <Trash2 className="w-4 h-4 text-red-500" />
                    </Button>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-xs">Strength: {patch.strength.toFixed(2)}</Label>
                  <Slider
                    value={[patch.strength]}
                    onValueChange={([value]) => updatePatch(patch.id, { strength: value })}
                    min={0}
                    max={2}
                    step={0.1}
                    disabled={!patch.isEnabled}
                  />
                </div>
              </Card>
            ))
          )}
        </div>
      </div>

      <Separator className="my-4" />

      <div className="flex-1">
        <h3 className="text-lg font-semibold mb-4">Inference Tester</h3>
        <div className="space-y-4">
          <div>
            <Label className="text-sm mb-2 block">Input Text</Label>
            <Textarea
              value={inferenceText}
              onChange={(e) => setInferenceText(e.target.value)}
              placeholder="Enter text to analyze..."
              rows={3}
            />
          </div>
          
          <Button 
            onClick={runInference} 
            disabled={isRunning || !inferenceText.trim()}
            className="w-full"
          >
            {isRunning ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Play className="w-4 h-4 mr-2" />
            )}
            Run Inference
          </Button>

          {results.length > 0 && (
            <Card className="p-4">
              <h4 className="text-sm font-medium mb-3">Logit Differences</h4>
              <div className="space-y-2">
                {results.map((result, idx) => (
                  <div key={idx} className="flex items-center gap-2 text-xs">
                    <div className="w-16 truncate">{result.token}</div>
                    <div className="flex-1 flex items-center gap-1">
                      <div 
                        className="bg-blue-200 dark:bg-blue-800 h-2 rounded"
                        style={{ width: `${result.original * 100}%` }}
                      />
                      <div 
                        className="bg-red-200 dark:bg-red-800 h-2 rounded"
                        style={{ width: `${Math.abs(result.patched - result.original) * 100}%` }}
                      />
                    </div>
                    <div className="w-12 text-right">
                      {((result.patched - result.original) * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

// Graph Drawer Component
const GraphDrawer: React.FC<{ store: ReturnType<typeof useAppStore> }> = ({ store }) => {
  const { state, updateUI } = store;
  const [zoom, setZoom] = useState(1);

  const [graphData, setGraphData] = useState<{
    nodes: Array<{id: string, x: number, y: number, label: string}>;
    edges: Array<{from: string, to: string, strength: number}>;
  }>({
    nodes: [],
    edges: []
  });

  // Load real graph data when component mounts or when features change
  useEffect(() => {
    const loadGraphData = async () => {
      if (state.features.length > 0 && state.selectedFeature) {
        try {
          // Get provenance data for selected feature
          const response = await fetch(`/api/feature/${state.selectedFeature.id}/provenance`);
          if (response.ok) {
            const provenanceData = await response.json();
            
            // Transform provenance data to graph format
            const nodes = provenanceData.nodes || [];
            const edges = provenanceData.edges || [];
            
            setGraphData({ nodes, edges });
          } else {
            // Fallback to simple feature graph
            const nodes = state.features.slice(0, 10).map((feature, index) => ({
              id: feature.id,
              x: 100 + (index % 3) * 150,
              y: 100 + Math.floor(index / 3) * 150,
              label: `Feature ${feature.id.split('_').pop()}`,
              layer: 0, // Default layer for fallback
              type: 'feature'
            }));
            
            const edges = [];
            for (let i = 0; i < nodes.length - 1; i++) {
              if (Math.random() > 0.5) { // Random connections for demo
                edges.push({
                  from: nodes[i].id,
                  to: nodes[i + 1].id,
                  strength: Math.random() * 0.8 + 0.2,
                  type: 'random'
                });
              }
            }
            
            setGraphData({ nodes, edges });
          }
        } catch (error) {
          console.error('Failed to load graph data:', error);
          // Fallback to empty graph
          setGraphData({ nodes: [], edges: [] });
        }
      } else {
        // No selected feature, show empty graph
        setGraphData({ nodes: [], edges: [] });
      }
    };
    
    loadGraphData();
  }, [state.features, state.selectedFeature]);

  return (
    <Drawer open={state.ui.drawerOpen} onOpenChange={(open) => updateUI({ drawerOpen: open })}>
      <DrawerTrigger asChild>
        <Button variant="outline" className="fixed bottom-4 right-4">
          <BarChart3 className="w-4 h-4 mr-2" />
          Provenance Graph
        </Button>
      </DrawerTrigger>
      <DrawerContent className="h-[80vh]">
        <DrawerHeader>
          <DrawerTitle>Feature Provenance Graph</DrawerTitle>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" onClick={() => setZoom(z => Math.min(z + 0.1, 2))}>
              <ZoomIn className="w-4 h-4" />
            </Button>
            <Button size="sm" variant="outline" onClick={() => setZoom(z => Math.max(z - 0.1, 0.5))}>
              <ZoomOut className="w-4 h-4" />
            </Button>
            <Button size="sm" variant="outline" onClick={() => setZoom(1)}>
              <RotateCcw className="w-4 h-4" />
            </Button>
          </div>
        </DrawerHeader>
        <div className="flex-1 p-4">
          <div 
            className="w-full h-full bg-muted rounded-lg relative overflow-hidden"
            style={{ transform: `scale(${zoom})` }}
          >
            <svg className="w-full h-full">
              {graphData.edges.map((edge, idx) => {
                const fromNode = graphData.nodes.find(n => n.id === edge.from);
                const toNode = graphData.nodes.find(n => n.id === edge.to);
                if (!fromNode || !toNode) return null;
                
                return (
                  <line
                    key={idx}
                    x1={fromNode.x}
                    y1={fromNode.y}
                    x2={toNode.x}
                    y2={toNode.y}
                    stroke="currentColor"
                    strokeWidth={edge.strength * 3}
                    opacity={0.6}
                  />
                );
              })}
              
              {graphData.nodes.map((node) => (
                <g key={node.id}>
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={20}
                    fill="currentColor"
                    opacity={0.8}
                  />
                  <text
                    x={node.x}
                    y={node.y + 35}
                    textAnchor="middle"
                    className="text-xs fill-current"
                  >
                    {node.label}
                  </text>
                </g>
              ))}
            </svg>
          </div>
        </div>
      </DrawerContent>
    </Drawer>
  );
};

// Progress Tracking Component
const ProgressTracker: React.FC<{ store: ReturnType<typeof useAppStore> }> = ({ store }) => {
  const { state } = store;

  if (state.progress.length === 0) {
    return null;
  }

  return (
    <div className="fixed top-20 right-4 w-80 space-y-2 z-50">
      {state.progress.map((progress) => {
        const getProgressColor = (status: ProgressInfo['status']) => {
          switch (status) {
            case 'starting': return 'bg-blue-500';
            case 'in_progress': return 'bg-yellow-500';
            case 'completed': return 'bg-green-500';
            case 'failed': return 'bg-red-500';
            default: return 'bg-gray-500';
          }
        };

        const getProgressIcon = (type: ProgressInfo['type']) => {
          switch (type) {
            case 'model_loading': return <Cpu className="w-4 h-4" />;
            case 'sae_loading': return <Layers className="w-4 h-4" />;
            case 'sae_training': return <Activity className="w-4 h-4" />;
            case 'feature_analysis': return <Eye className="w-4 h-4" />;
            case 'patch_creation': return <Plus className="w-4 h-4" />;
            default: return <Activity className="w-4 h-4" />;
          }
        };

        return (
          <Card key={progress.id} className="p-4 shadow-lg bg-background border">
            <div className="flex items-center gap-3 mb-2">
              <div className={`w-2 h-2 rounded-full ${getProgressColor(progress.status)} ${progress.status === 'in_progress' ? 'animate-pulse' : ''}`} />
              {getProgressIcon(progress.type)}
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <h4 className="text-sm font-medium">{progress.title}</h4>
                  {progress.type === 'sae_training' && progress.details?.trainingMode && (
                    <Badge variant={progress.details.trainingMode === 'fast' ? "secondary" : "outline"} className="text-xs">
                      {progress.details.trainingMode === 'fast' ? "⚡ Fast" : "⚙️ Standard"}
                    </Badge>
                  )}
                </div>
                <p className="text-xs text-muted-foreground">{progress.description}</p>
                {progress.type === 'sae_training' && progress.details?.speedMultiplier && (
                  <p className="text-xs text-green-600 dark:text-green-400 mt-1">
                    ~{progress.details.speedMultiplier}x faster than baseline
                  </p>
                )}
              </div>
              {progress.status === 'in_progress' && (
                <Loader2 className="w-4 h-4 animate-spin" />
              )}
            </div>
            
            {progress.status === 'in_progress' && (
              <div className="mb-2">
                <div className="flex items-center justify-between text-xs mb-1">
                  <span>{progress.progress.toFixed(0)}%</span>
                  {progress.details?.estimatedTime && (
                    <span>ETA: {progress.details.estimatedTime}</span>
                  )}
                </div>
                <div className="w-full bg-muted rounded-full h-1.5">
                  <div 
                    className="bg-primary h-1.5 rounded-full transition-all duration-300"
                    style={{ width: `${progress.progress}%` }}
                  />
                </div>
              </div>
            )}

            {progress.details?.currentStep && (
              <p className="text-xs text-muted-foreground mt-1">
                {progress.details.currentStep}
              </p>
            )}

            {progress.error && (
              <div className="mt-2 p-2 bg-red-50 dark:bg-red-900/20 rounded text-xs text-red-600 dark:text-red-400">
                {progress.error}
              </div>
            )}

            {progress.details?.metrics && Object.keys(progress.details.metrics).length > 0 && (
              <div className="mt-2 space-y-1">
                {Object.entries(progress.details.metrics).map(([key, value]) => (
                  <div key={key} className="flex justify-between text-xs">
                    <span className="text-muted-foreground">{key}:</span>
                    <span>{typeof value === 'number' ? value.toFixed(4) : value}</span>
                  </div>
                ))}
              </div>
            )}
          </Card>
        );
      })}
    </div>
  );
};

// Main App Component
const InterpretabilityWorkbench: React.FC = () => {
  const store = useAppStore();

  useEffect(() => {
    // Initialize theme
    document.body.className = store.state.ui.theme === 'dark' ? 'dark' : '';
  }, [store.state.ui.theme]);

  return (
    <div className="h-screen flex flex-col bg-background text-foreground">
      <Header store={store} />
      
      <div className="flex-1 flex overflow-hidden">
        <FeatureTable store={store} />
        <FeatureDetail store={store} />
        <PatchConsole store={store} />
      </div>
      
      {/* Debug: Manual feature loading button */}
      {store.state.sae.status === 'ready' && store.state.features.length === 0 && (
        <div className="fixed bottom-4 right-4 z-50">
          <Button 
            onClick={() => store.loadFeatures()}
            className="bg-blue-600 hover:bg-blue-700"
          >
            Load Features ({store.state.sae.featureCount || 0})
          </Button>
        </div>
      )}
      
      <GraphDrawer store={store} />
      <ProgressTracker store={store} />
    </div>
  );
};

export default InterpretabilityWorkbench; 