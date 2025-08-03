import { InferenceResult } from './api';

// WebSocket Message Types
export interface WebSocketMessage {
  type: string;
  data?: any;
  requestId?: string;
  timestamp?: number;
}

// Client to Server Messages
export interface InferenceRequestMessage extends WebSocketMessage {
  type: 'inference_request';
  data: {
    text: string;
    maxTokens?: number;
  };
  requestId: string;
}

export interface SubscriptionMessage extends WebSocketMessage {
  type: 'subscription';
  data: {
    subscriptionType: 'patch_updates' | 'model_status' | 'inference_results';
    action: 'subscribe' | 'unsubscribe';
  };
}

export interface PingMessage extends WebSocketMessage {
  type: 'ping';
}

// Server to Client Messages
export interface InferenceResultMessage extends WebSocketMessage {
  type: 'inference_result';
  data: InferenceResult;
  requestId: string;
}

export interface PatchToggledMessage extends WebSocketMessage {
  type: 'patch_toggled';
  data: {
    patchId: string;
    action: 'enabled' | 'disabled';
    strength?: number;
  };
}

export interface ModelStatusMessage extends WebSocketMessage {
  type: 'model_status';
  data: {
    status: 'loading' | 'ready' | 'error';
    modelName?: string;
    error?: string;
  };
}

export interface PongMessage extends WebSocketMessage {
  type: 'pong';
}

export type ServerMessage = 
  | InferenceResultMessage 
  | PatchToggledMessage 
  | ModelStatusMessage 
  | PongMessage;

export type ClientMessage = 
  | InferenceRequestMessage 
  | SubscriptionMessage 
  | PingMessage;

// WebSocket Client Class
export class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000; // Start with 1 second
  private pingInterval: NodeJS.Timeout | null = null;
  private isConnecting = false;

  // Event handlers
  private onOpenHandlers: Array<() => void> = [];
  private onCloseHandlers: Array<(event: CloseEvent) => void> = [];
  private onErrorHandlers: Array<(error: Event) => void> = [];
  private onMessageHandlers: Array<(message: ServerMessage) => void> = [];

  // Pending requests (for request-response pattern)
  private pendingRequests = new Map<string, {
    resolve: (data: any) => void;
    reject: (error: any) => void;
    timeout: NodeJS.Timeout;
  }>();

  constructor(url: string) {
    // Handle relative URLs by converting to WebSocket URL
    if (url && !url.startsWith('ws://') && !url.startsWith('wss://')) {
      if (typeof window !== 'undefined') {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // Use window.location.origin to avoid IP address issues
        const origin = window.location.origin;
        this.url = `${origin.replace(/^http/, 'ws')}${url.startsWith('/') ? url : '/' + url}`;
      } else {
        // Fallback for SSR
        this.url = `ws://localhost:8000${url.startsWith('/') ? url : '/' + url}`;
      }
    } else {
      this.url = url;
    }
  }

  // Connection Management
  connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return Promise.resolve();
    }

    if (this.isConnecting) {
      return new Promise((resolve, reject) => {
        const onOpen = () => {
          this.off('open', onOpen);
          this.off('error', onError);
          resolve();
        };
        const onError = (error: Event) => {
          this.off('open', onOpen);
          this.off('error', onError);
          reject(error);
        };
        this.on('open', onOpen);
        this.on('error', onError);
      });
    }

    return new Promise((resolve, reject) => {
      try {
        this.isConnecting = true;
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.reconnectDelay = 1000;
          this.startPing();
          this.onOpenHandlers.forEach(handler => handler());
          resolve();
        };

        this.ws.onclose = (event) => {
          console.log('WebSocket disconnected:', event.code, event.reason);
          this.isConnecting = false;
          this.stopPing();
          this.onCloseHandlers.forEach(handler => handler(event));
          
          // Auto-reconnect unless it was a clean close
          if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.isConnecting = false;
          this.onErrorHandlers.forEach(handler => handler(error));
          reject(error);
        };

        this.ws.onmessage = (event) => {
          try {
            const message: ServerMessage = JSON.parse(event.data);
            console.log('WebSocket message received:', message);
            
            // Handle pending requests
            if (message.requestId && this.pendingRequests.has(message.requestId)) {
              const pending = this.pendingRequests.get(message.requestId)!;
              clearTimeout(pending.timeout);
              this.pendingRequests.delete(message.requestId);
              pending.resolve(message.data);
            }

            // Handle pong responses
            if (message.type === 'pong') {
              return; // Don't forward pong messages to handlers
            }

            this.onMessageHandlers.forEach(handler => handler(message));
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error, event.data);
          }
        };

      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnecting');
      this.ws = null;
    }
    this.stopPing();
    this.clearPendingRequests();
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    
    setTimeout(() => {
      if (this.ws?.readyState !== WebSocket.OPEN) {
        this.connect().catch(error => {
          console.error('Reconnection failed:', error);
        });
      }
    }, delay);
  }

  // Ping/Pong for connection health
  private startPing(): void {
    this.pingInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' });
      }
    }, 30000); // Ping every 30 seconds
  }

  private stopPing(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  // Message Sending
  send(message: ClientMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.log('Sending WebSocket message:', message);
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, message not sent:', message);
      throw new Error('WebSocket not connected');
    }
  }

  // Request-Response Pattern
  sendRequest<T>(message: ClientMessage, timeoutMs = 10000): Promise<T> {
    return new Promise((resolve, reject) => {
      const requestId = message.requestId || this.generateRequestId();
      message.requestId = requestId;

      const timeout = setTimeout(() => {
        this.pendingRequests.delete(requestId);
        reject(new Error(`Request timeout after ${timeoutMs}ms`));
      }, timeoutMs);

      this.pendingRequests.set(requestId, {
        resolve,
        reject,
        timeout
      });

      try {
        this.send(message);
      } catch (error) {
        clearTimeout(timeout);
        this.pendingRequests.delete(requestId);
        reject(error);
      }
    });
  }

  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private clearPendingRequests(): void {
    this.pendingRequests.forEach(pending => {
      clearTimeout(pending.timeout);
      pending.reject(new Error('WebSocket disconnected'));
    });
    this.pendingRequests.clear();
  }

  // Event Handlers
  on(event: 'open', handler: () => void): void;
  on(event: 'close', handler: (event: CloseEvent) => void): void;
  on(event: 'error', handler: (error: Event) => void): void;
  on(event: 'message', handler: (message: ServerMessage) => void): void;
  on(event: string, handler: any): void {
    switch (event) {
      case 'open':
        this.onOpenHandlers.push(handler);
        break;
      case 'close':
        this.onCloseHandlers.push(handler);
        break;
      case 'error':
        this.onErrorHandlers.push(handler);
        break;
      case 'message':
        this.onMessageHandlers.push(handler);
        break;
    }
  }

  off(event: 'open', handler: () => void): void;
  off(event: 'close', handler: (event: CloseEvent) => void): void;
  off(event: 'error', handler: (error: Event) => void): void;
  off(event: 'message', handler: (message: ServerMessage) => void): void;
  off(event: string, handler: any): void {
    switch (event) {
      case 'open':
        this.onOpenHandlers = this.onOpenHandlers.filter(h => h !== handler);
        break;
      case 'close':
        this.onCloseHandlers = this.onCloseHandlers.filter(h => h !== handler);
        break;
      case 'error':
        this.onErrorHandlers = this.onErrorHandlers.filter(h => h !== handler);
        break;
      case 'message':
        this.onMessageHandlers = this.onMessageHandlers.filter(h => h !== handler);
        break;
    }
  }

  // Connection State
  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  get readyState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED;
  }

  // High-level API methods
  async requestInference(text: string, maxTokens?: number): Promise<InferenceResult> {
    return this.sendRequest<InferenceResult>({
      type: 'inference_request',
      data: { text, maxTokens },
      requestId: this.generateRequestId()
    });
  }

  subscribe(subscriptionType: 'patch_updates' | 'model_status' | 'inference_results'): void {
    this.send({
      type: 'subscription',
      data: {
        subscriptionType,
        action: 'subscribe'
      }
    });
  }

  unsubscribe(subscriptionType: 'patch_updates' | 'model_status' | 'inference_results'): void {
    this.send({
      type: 'subscription',
      data: {
        subscriptionType,
        action: 'unsubscribe'
      }
    });
  }
}

// Create default instance
import apiClient from './api';
const wsClient = new WebSocketClient(
  import.meta.env.VITE_WS_URL || apiClient.getWebSocketURL()
);

export default wsClient;