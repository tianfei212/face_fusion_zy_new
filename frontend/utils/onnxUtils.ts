
import { OrtModule } from '../types';
import type { MutableRefObject } from 'react';

export const resolveAssetUrl = (url: string) => {
  if (!url) return '';
  if (url.startsWith('http') || url.startsWith('data:') || url.startsWith('blob:')) return url;
  const cleanUrl = url.replace(/^public\//, '').replace(/^\//, '');
  return `./${cleanUrl}`;
};

export const resolveModelUrl = (filename: string, checkpointRoot: string) => {
  if (import.meta.env.DEV) return `/@fs${checkpointRoot}/${filename}`;
  return `/models/${filename}`;
};

export const resolveOrtWasmPrefix = () => {
  if (import.meta.env.DEV) {
    // This path depends on where node_modules is. Assuming standard structure.
    return '/@fs/home/ubuntu/codes/ai_face_change/frontend/node_modules/onnxruntime-web/dist/';
  }
  return undefined;
};

export const pickInputName = (names: string[]) => {
  const bySrc = names.find(name => name.toLowerCase().includes('src'));
  if (bySrc) return bySrc;
  const byInput = names.find(name => name.toLowerCase().includes('input'));
  return byInput || names[0];
};

export const pickOutput = (outputs: Record<string, import('onnxruntime-web').Tensor>, keys: string[]) => {
  for (const key of keys) {
    if (outputs[key]) return outputs[key];
  }
  const firstKey = Object.keys(outputs)[0];
  return outputs[firstKey];
};

export const resolveDims = (dims: Array<number | string>, width: number, height: number) => {
  return dims.map((dim, index) => {
    if (typeof dim === 'number' && dim > 0) return dim;
    if (index === 2) return height;
    if (index === 3) return width;
    return 1;
  }) as number[];
};

// Convert float32 number to IEEE754 half precision bits (Uint16)
const float32ToHalfBits = (val: number) => {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = val;
  const x = u32[0];
  const sign = (x >>> 31) & 0x1;
  let exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  let half;
  if (exp === 0xff) {
    half = mant ? 0x7e00 : 0x7c00; // NaN or Inf
  } else {
    exp = exp - 127 + 15;
    if (exp <= 0) {
      if (exp < -10) half = 0; // underflow
      else {
        mant |= 0x800000;
        const shift = 1 - exp;
        half = mant >> (shift + 13);
      }
    } else if (exp >= 31) {
      half = 0x7c00; // overflow to Inf
    } else {
      half = (exp << 10) | (mant >> 13);
    }
  }
  return (sign << 15) | half;
};

export const buildInputTensor = (
  ort: OrtModule,
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  inputBufferRef: MutableRefObject<Float32Array | Uint16Array | null>,
  dtype: 'float32' | 'float16'
) => {
  const imageData = ctx.getImageData(0, 0, width, height);
  const totalPixels = width * height;
  const requiredSize = totalPixels * 3;

  if (dtype === 'float16') {
    if (!inputBufferRef.current || !(inputBufferRef.current instanceof Uint16Array) || inputBufferRef.current.length !== requiredSize) {
      inputBufferRef.current = new Uint16Array(requiredSize);
    }
    const buf = inputBufferRef.current as Uint16Array;
    for (let i = 0; i < totalPixels; i += 1) {
      const base = i * 4;
      const r = imageData.data[base] / 255;
      const g = imageData.data[base + 1] / 255;
      const b = imageData.data[base + 2] / 255;
      buf[i] = float32ToHalfBits(r);
      buf[i + totalPixels] = float32ToHalfBits(g);
      buf[i + totalPixels * 2] = float32ToHalfBits(b);
    }
    return {
      tensor: new ort.Tensor('float16', buf, [1, 3, height, width]),
      imageData
    };
  } else {
    if (!inputBufferRef.current || !(inputBufferRef.current instanceof Float32Array) || inputBufferRef.current.length !== requiredSize) {
      inputBufferRef.current = new Float32Array(requiredSize);
    }
    const buf = inputBufferRef.current as Float32Array;
    for (let i = 0; i < totalPixels; i += 1) {
      const base = i * 4;
      const r = imageData.data[base] / 255;
      const g = imageData.data[base + 1] / 255;
      const b = imageData.data[base + 2] / 255;
      buf[i] = r;
      buf[i + totalPixels] = g;
      buf[i + totalPixels * 2] = b;
    }
    return {
      tensor: new ort.Tensor('float32', buf, [1, 3, height, width]),
      imageData
    };
  }
};

export const ensureOutputImage = (
  width: number, 
  height: number, 
  outputImageRef: MutableRefObject<ImageData | null>
) => {
  if (!outputImageRef.current || outputImageRef.current.width !== width || outputImageRef.current.height !== height) {
    outputImageRef.current = new ImageData(width, height);
  }
  return outputImageRef.current;
};
