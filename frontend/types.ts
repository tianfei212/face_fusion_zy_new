
export enum OperationMode {
  FACE_SWAP = 'FACE_SWAP',
  MODEL_SWAP = 'MODEL_SWAP'
}

export interface ThumbnailItem {
  id: string;
  name: string;
  url: string;
}

export interface SystemConfig {
  system: {
    name: string;
    subName: string;
    logoUrl: string;
    backgroundUrl: string;
    placeholderUrl: string;
    standbyImageUrl?: string;
    cameraOfflineUrl?: string;
    footerText: string;
  };
  directories: {
    portraitWorkDir: string;
    portraitDfmDir: string;
    sceneDir: string;
  };
  Is_from_blank_get_config: boolean;
  blank_url: string;
  stream_url?: string;
  stream_send_url?: string;
  stream_recv_url?: string;
  stream_send_fps?: number;
  directories_blank: {
    portraitWorkList: string;
    portraitDfmList: string;
    sceneList: string;
  };
  obsConfig: {
    enabled: boolean;
    streamUrl: string;
    username: string;
    secret: string;
  };
}

export interface AppState {
  mode: OperationMode;
  similarity: number;
  sharpness: number;
  isProcessing: boolean;
  selectedPortraitId: string | null;
  selectedBackgroundId: string | null;
  portraits: ThumbnailItem[];
  models: ThumbnailItem[];
  scenes: ThumbnailItem[];
  isCameraOn: boolean;
  isAutoMatting: boolean;
  serverStatus: 'idle' | 'testing' | 'success' | 'error';
  latency: number | null;
  isConfigOpen: boolean;
}

export type LogLevel = 'info' | 'warn' | 'error';

export interface LogEntry {
  id: string;
  time: string;
  level: LogLevel;
  message: string;
}

export type OrtModule = typeof import('onnxruntime-web/webgpu');
