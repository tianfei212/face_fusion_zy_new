
import { ThumbnailItem } from './types';

export const PORTRAITS: ThumbnailItem[] = Array.from({ length: 6 }).map((_, i) => ({
  id: `portrait-${i + 1}`,
  name: `人像 ${i + 1}`,
  url: `https://picsum.photos/seed/p${i + 1}/200/200`
}));

export const BACKGROUNDS: ThumbnailItem[] = [
  {
    id: 'bg-primary',
    name: '中影官方背景',
    url: 'zy-blank.png'
  },
  ...Array.from({ length: 2 }).map((_, i) => ({
    id: `bg-${i + 1}`,
    name: `示例背景 ${i + 1}`,
    url: `https://picsum.photos/seed/b${i + 1}/400/225`
  }))
];
