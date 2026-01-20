
import { useState, useRef, useEffect, useCallback } from 'react';
import type React from 'react';
import { LogLevel, OrtModule } from '../types';
import { 
  resolveModelUrl, 
  resolveOrtWasmPrefix, 
  pickInputName, 
  pickOutput, 
  resolveDims, 
  buildInputTensor, 
  ensureOutputImage 
} from '../utils/onnxUtils';

const CHECKPOINT_ROOT = '/home/ubuntu/codes/ai_face_change/checkpoints';

export const useRealTimeMatting = (
  videoRef: React.RefObject<HTMLVideoElement>,
  outputCanvasRefMain: React.RefObject<HTMLCanvasElement>,
  outputCanvasRefRaw: React.RefObject<HTMLCanvasElement>,
  isCameraOn: boolean,
  isAutoMatting: boolean,
  log: (level: LogLevel, message: string) => void,
  onRvmReady?: () => void
) => {
  const [mattingStatus, setMattingStatus] = useState('初始化中...');
  const [mattingMode, setMattingMode] = useState('等待加载');
  const [rvmReady, setRvmReady] = useState(false);
  const [depthReady, setDepthReady] = useState(false);
  const [isUpgrading, setIsUpgrading] = useState(false);

  const ortRef = useRef<OrtModule | null>(null);
  const ortConfiguredRef = useRef(false);
  const sessionRvmRef = useRef<import('onnxruntime-web').InferenceSession | null>(null);
  const sessionDepthRef = useRef<import('onnxruntime-web').InferenceSession | null>(null);
  const rvmInputDtypeRef = useRef<'float16' | 'float32'>('float32');
  const depthInputDtypeRef = useRef<'float16' | 'float32'>('float32');
  
  const processingCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rvmStateRef = useRef<Record<string, import('onnxruntime-web').Tensor>>({});
  const rvmSizeRef = useRef<{ width: number; height: number } | null>(null);
  const inputBufferRef = useRef<Float32Array | Uint16Array | null>(null);
  const outputImageRef = useRef<ImageData | null>(null);
  
  const animationFrameRef = useRef<number | null>(null);
  const frameBusyRef = useRef(false);
  const depthFusionLoggedRef = useRef(false);
  
  const isCameraOnRef = useRef(isCameraOn);
  const isAutoMattingRef = useRef(isAutoMatting);

  useEffect(() => { isCameraOnRef.current = isCameraOn; }, [isCameraOn]);
  useEffect(() => { isAutoMattingRef.current = isAutoMatting; }, [isAutoMatting]);

  const ensureProcessingCanvas = () => {
    if (!processingCanvasRef.current) processingCanvasRef.current = document.createElement('canvas');
    return processingCanvasRef.current;
  };

  const runMattingFrame = useCallback(async () => {
    const video = videoRef.current;
    const outputCanvas = outputCanvasRefMain.current;
    if (!video || !outputCanvas) return;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    const outputCtx = outputCanvas.getContext('2d');
    if (!outputCtx) return;

    const rvmSession = sessionRvmRef.current;
    const ort = ortRef.current;

    const inputWidth = video.videoWidth;
    const inputHeight = video.videoHeight;

    if (!rvmSession || !ort || !isAutoMattingRef.current) {
      // Pass-through if models not ready
      outputCanvas.width = inputWidth;
      outputCanvas.height = inputHeight;
      outputCtx.drawImage(video, 0, 0, inputWidth, inputHeight);
      const rawCanvas = outputCanvasRefRaw.current;
      if (rawCanvas) {
        const rawCtx = rawCanvas.getContext('2d');
        if (rawCtx) {
          rawCanvas.width = inputWidth;
          rawCanvas.height = inputHeight;
          rawCtx.drawImage(video, 0, 0, inputWidth, inputHeight);
        }
      }
      return;
    }

    const inputName = pickInputName(rvmSession.inputNames);
    const inputMeta = rvmSession.inputMetadata[inputName];
    const resolvedInputDims = inputMeta && Array.isArray(inputMeta.dimensions)
      ? resolveDims(inputMeta.dimensions, inputWidth, inputHeight)
      : [1, 3, inputHeight, inputWidth];
    const modelHeight = resolvedInputDims[2];
    const modelWidth = resolvedInputDims[3];

    const processingCanvas = ensureProcessingCanvas();
    processingCanvas.width = modelWidth;
    processingCanvas.height = modelHeight;
    const processingCtx = processingCanvas.getContext('2d', { willReadFrequently: true });
    if (!processingCtx) return;
    processingCtx.drawImage(video, 0, 0, modelWidth, modelHeight);

    if (!rvmSizeRef.current || rvmSizeRef.current.width !== modelWidth || rvmSizeRef.current.height !== modelHeight) {
      rvmStateRef.current = {};
      rvmSizeRef.current = { width: modelWidth, height: modelHeight };
    }

    const { tensor: rvmInput, imageData: rvmImageData } = buildInputTensor(ort, processingCtx, modelWidth, modelHeight, inputBufferRef, rvmInputDtypeRef.current);

    const rvmFeeds: Record<string, import('onnxruntime-web').Tensor> = {
      [inputName]: rvmInput
    };

    const stateInputs = rvmSession.inputNames.filter(name => name !== inputName);
    for (const name of stateInputs) {
      if (name.toLowerCase().includes('downsample')) {
        // RVM 需要标量 downsample_ratio，推荐 0.25-0.4
        rvmFeeds[name] = new ort.Tensor(
          'float32',
          new Float32Array([0.25]),
          [1]
        );
        continue;
      }
      if (!rvmStateRef.current[name]) {
        const meta = rvmSession.inputMetadata[name];
        const dims = meta && Array.isArray(meta.dimensions) ? resolveDims(meta.dimensions, modelWidth, modelHeight) : [1, 1, modelHeight, modelWidth];
        const size = dims.reduce((acc, v) => acc * v, 1);
        // RVM state 建议使用 float32 零张量
        rvmStateRef.current[name] = new ort.Tensor('float32', new Float32Array(size), dims);
      }
      rvmFeeds[name] = rvmStateRef.current[name];
    }

    const rvmOutputs = await rvmSession.run(rvmFeeds);
    for (const name of stateInputs) {
      if (rvmOutputs[name]) rvmStateRef.current[name] = rvmOutputs[name];
    }

    const alphaTensor = pickOutput(rvmOutputs, ['pha', 'alpha', 'mask']);
    const alphaData = alphaTensor.data as Float32Array;
    const fgrTensor = rvmOutputs['fgr'] || rvmOutputs['foreground'] || undefined;
    const fgrData: Float32Array | null = fgrTensor ? (fgrTensor.data as Float32Array) : null;

    let depthMask: Float32Array | null = null;
    let depthWidth = modelWidth;
    let depthHeight = modelHeight;

    const depthSession = sessionDepthRef.current;
    if (depthSession) {
      if (!depthFusionLoggedRef.current) {
        depthFusionLoggedRef.current = true;
        log('info', '深度融合已启用');
      }
      const depthInputName = pickInputName(depthSession.inputNames);
      const depthMeta = depthSession.inputMetadata[depthInputName];
      const depthDims = depthMeta && Array.isArray(depthMeta.dimensions)
        ? resolveDims(depthMeta.dimensions, modelWidth, modelHeight)
        : [1, 3, modelHeight, modelWidth];
      depthHeight = depthDims[2];
      depthWidth = depthDims[3];
      
      processingCanvas.width = depthWidth;
      processingCanvas.height = depthHeight;
      processingCtx.drawImage(video, 0, 0, depthWidth, depthHeight);
      
      const { tensor: depthInput } = buildInputTensor(ort, processingCtx, depthWidth, depthHeight, inputBufferRef, depthInputDtypeRef.current);
      const depthOutputs = await depthSession.run({ [depthInputName]: depthInput });
      const depthTensor = pickOutput(depthOutputs, ['depth', 'pred', 'output', 'disp']);
      const depthData = depthTensor.data as Float32Array;
      
      let min = Infinity;
      let max = -Infinity;
      for (let i = 0; i < depthData.length; i += 1) {
        const value = depthData[i];
        if (value < min) min = value;
        if (value > max) max = value;
      }
      const range = max - min;
      if (range < 1e-6) {
        log('warn', 'Depth 掩码无效：范围过小，跳过融合');
      } else {
        depthMask = new Float32Array(depthData.length);
        let ones = 0;
        for (let i = 0; i < depthData.length; i += 1) {
          const normalized = (depthData[i] - min) / range;
          const v = normalized > 0.45 ? 1 : 0;
          depthMask[i] = v;
          ones += v;
        }
        const ratio = ones / depthData.length;
        if (ratio < 0.05 || ratio > 0.95) {
          log('warn', `Depth 掩码分布异常(${Math.round(ratio * 100)}%)，跳过融合`);
          depthMask = null;
        }
      }
    }

    const outputImage = ensureOutputImage(modelWidth, modelHeight, outputImageRef);
    const outputData = outputImage.data;
    const pixelCount = modelWidth * modelHeight;

    // Alpha有效性检测，避免整帧透明导致黑屏
    let alphaMax = 0;
    for (let i = 0; i < Math.min(alphaData.length, pixelCount); i += Math.ceil(pixelCount / 256)) {
      const v = alphaData[i] ?? 0;
      if (v > alphaMax) alphaMax = v;
    }
    if (alphaMax < 0.01) {
      log('warn', `RVM alpha 输出接近全零，dims=${modelWidth}x${modelHeight}，回退透传`);
      outputCanvas.width = inputWidth;
      outputCanvas.height = inputHeight;
      outputCtx.drawImage(video, 0, 0, inputWidth, inputHeight);
      const rawCanvas = outputCanvasRefRaw.current;
      if (rawCanvas) {
        const rawCtx = rawCanvas.getContext('2d');
        if (rawCtx) {
          rawCanvas.width = inputWidth;
          rawCanvas.height = inputHeight;
          rawCtx.drawImage(video, 0, 0, inputWidth, inputHeight);
        }
      }
      return;
    }

    for (let i = 0; i < pixelCount; i += 1) {
      const alphaBase = alphaData[i] ?? 0;
      let alpha = Math.min(1, Math.max(0, alphaBase));
      if (depthMask) {
        const x = i % modelWidth;
        const y = Math.floor(i / modelWidth);
        const dx = Math.min(depthWidth - 1, Math.floor((x / modelWidth) * depthWidth));
        const dy = Math.min(depthHeight - 1, Math.floor((y / modelHeight) * depthHeight));
        alpha *= depthMask[dy * depthWidth + dx] ?? 0;
      }
      const srcBase = i * 4;
      if (fgrData && fgrData.length >= pixelCount * 3) {
        const r = fgrData[i];
        const g = fgrData[i + pixelCount];
        const b = fgrData[i + pixelCount * 2];
        outputData[srcBase] = Math.round((r ?? 0) * 255);
        outputData[srcBase + 1] = Math.round((g ?? 0) * 255);
        outputData[srcBase + 2] = Math.round((b ?? 0) * 255);
      } else {
        outputData[srcBase] = rvmImageData.data[srcBase];
        outputData[srcBase + 1] = rvmImageData.data[srcBase + 1];
        outputData[srcBase + 2] = rvmImageData.data[srcBase + 2];
      }
      if (depthMask) {
        const x = i % modelWidth;
        const y = Math.floor(i / modelWidth);
        const dx = Math.min(depthWidth - 1, Math.floor((x / modelWidth) * depthWidth));
        const dy = Math.min(depthHeight - 1, Math.floor((y / modelHeight) * depthHeight));
        const dm = depthMask[dy * depthWidth + dx] ?? 0;
        alpha = Math.min(1, Math.max(0, alpha * (0.5 + 0.5 * dm)));
      }
      outputData[srcBase + 3] = Math.round(alpha * 255);
    }

    outputCanvas.width = modelWidth;
    outputCanvas.height = modelHeight;
    outputCtx.putImageData(outputImage, 0, 0);

    // Mirror to RAW panel canvas if provided
    const rawCanvas = outputCanvasRefRaw.current;
    if (rawCanvas) {
      const rawCtx = rawCanvas.getContext('2d');
      if (rawCtx) {
        rawCanvas.width = modelWidth;
        rawCanvas.height = modelHeight;
        rawCtx.putImageData(outputImage, 0, 0);
      }
    }
  }, [log]); // Removed dependencies that are refs or stable

  const startMattingLoop = useCallback(() => {
    if (animationFrameRef.current !== null) return;
    log('info', '抠像循环已启动');

    const tick = async () => {
      if (!isCameraOnRef.current) {
        animationFrameRef.current = requestAnimationFrame(tick);
        return;
      }

      if (!frameBusyRef.current) {
        frameBusyRef.current = true;
        try {
          await runMattingFrame();
        } finally {
          frameBusyRef.current = false;
        }
      }

      animationFrameRef.current = requestAnimationFrame(tick);
    };

    animationFrameRef.current = requestAnimationFrame(tick);
  }, [runMattingFrame, log]);

  useEffect(() => {
    startMattingLoop();
    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [startMattingLoop]);

  useEffect(() => {
    let cancelled = false;

    const loadModels = async () => {
      try {
        setMattingStatus('正在加载人像模型 (1/2)...');
        log('info', '开始加载 RVM 模型');
        const supportsWebGpu = typeof navigator !== 'undefined' && 'gpu' in navigator;
        if (!ortRef.current) {
          ortRef.current = supportsWebGpu 
            ? await import('onnxruntime-web/webgpu')
            : await import('onnxruntime-web');
        }
        const ort = ortRef.current;
        if (!ort) return;
        if (!ortConfiguredRef.current) {
          const wasmPrefix = resolveOrtWasmPrefix();
          if (wasmPrefix) {
            ort.env.wasm.wasmPaths = wasmPrefix;
            log('info', `WASM运行时前缀已配置: ${wasmPrefix}`);
          }
          ort.env.wasm.numThreads = 1;
          ortConfiguredRef.current = true;
        }
        
        log('info', supportsWebGpu ? 'WebGPU 可用，启用 WebGPU' : 'WebGPU 不可用，回退到 WASM');
        
        const createSession = async (url: string, _outType: 'float16' | 'float32') => {
          const ep = supportsWebGpu ? 'webgpu' : 'wasm';
          return await ort.InferenceSession.create(url, {
            executionProviders: [ep],
            graphOptimizationLevel: 'all',
            preferredOutputType: 'float32'
          });
        };

        const rvmUrlFp16 = resolveModelUrl('rvm_mobilenetv3_fp16.onnx', CHECKPOINT_ROOT);
        const rvmUrlFp32 = resolveModelUrl('rvm_mobilenetv3_fp32.onnx', CHECKPOINT_ROOT);
        log('info', `RVM 模型路径: ${supportsWebGpu ? rvmUrlFp16 : rvmUrlFp32}`);
        let rvmSession: import('onnxruntime-web').InferenceSession;
        try {
          rvmSession = await createSession(supportsWebGpu ? rvmUrlFp16 : rvmUrlFp32, supportsWebGpu ? 'float16' : 'float32');
          rvmInputDtypeRef.current = supportsWebGpu ? 'float16' : 'float32';
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          if (msg.includes('Concat') && msg.includes('tensor(float16)') && msg.includes('tensor(float)')) {
            log('warn', 'RVM fp16 模型存在混精度，自动回退到 fp32');
            rvmSession = await createSession(rvmUrlFp32, 'float32');
            rvmInputDtypeRef.current = 'float32';
          } else {
            throw e;
          }
        }
        if (cancelled) return;
        
        sessionRvmRef.current = rvmSession;
        // dtype 已按 EP 设置
        setRvmReady(true);
        setMattingMode('基础模式 (高性能)');
        setMattingStatus('人像模型就绪，正在后台加载深度模型...');
        log('info', 'RVM 模型已就绪');
        if (onRvmReady) onRvmReady();

        setIsUpgrading(true);
        log('info', '开始后台加载 Depth 模型');
        await new Promise(resolve => setTimeout(resolve, 100));

        try {
          const depthUrlFp16 = resolveModelUrl('depth_anything_v2_vits_fp16.onnx', CHECKPOINT_ROOT);
          const depthUrlFp32 = resolveModelUrl('depth_anything_v2_vits_fp32.onnx', CHECKPOINT_ROOT);
          log('info', `Depth 模型路径: ${supportsWebGpu ? depthUrlFp16 : depthUrlFp32}`);
          let depthSession: import('onnxruntime-web').InferenceSession;
          try {
            depthSession = await createSession(supportsWebGpu ? depthUrlFp16 : depthUrlFp32, supportsWebGpu ? 'float16' : 'float32');
            depthInputDtypeRef.current = supportsWebGpu ? 'float16' : 'float32';
          } catch (de) {
            const dmsg = de instanceof Error ? de.message : String(de);
            if (dmsg.includes('Concat') && dmsg.includes('tensor(float16)') && dmsg.includes('tensor(float)')) {
              log('warn', 'Depth fp16 模型存在混精度，自动回退到 fp32');
              depthSession = await createSession(depthUrlFp32, 'float32');
              depthInputDtypeRef.current = 'float32';
            } else {
              throw de;
            }
          }
          if (cancelled) return;
          sessionDepthRef.current = depthSession;
          setDepthReady(true);
          setMattingMode('专业模式 (人像分离+发丝级)');
          setMattingStatus('深度模型就绪');
          setIsUpgrading(false);
          log('info', 'Depth 模型已就绪');
        } catch (depthError) {
          const depthMessage = depthError instanceof Error ? depthError.message : '未知错误';
          setDepthReady(false);
          setIsUpgrading(false);
          setMattingStatus(`深度模型加载失败，保持基础模式: ${depthMessage}`);
          log('warn', `Depth 模型加载失败: ${depthMessage}`);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : '未知错误';
        setMattingStatus(`模型加载出错: ${message}`);
        setIsUpgrading(false);
        log('error', `模型加载失败: ${message}`);
      }
    };

    loadModels();
    return () => { cancelled = true; };
  }, [log, onRvmReady]);

  return {
    mattingStatus,
    mattingMode,
    rvmReady,
    depthReady,
    isUpgrading
  };
};
