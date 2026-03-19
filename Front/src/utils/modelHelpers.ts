// src/utils/modelHelpers.ts
import * as React from 'react';

export function renderConfig(config: any): React.ReactNode {
  if (typeof config !== 'object' || config === null) {
    return React.createElement('span', null, String(config));
  }
  if (Array.isArray(config)) {
    return React.createElement('ul', { className: 'ml-4 list-disc' }, 
      config.map((item, idx) => 
        React.createElement('li', { key: idx }, renderConfig(item))
      )
    );
  }
  return React.createElement('ul', { className: 'ml-4 list-disc' },
    Object.entries(config).map(([key, value]) => 
      React.createElement('li', { key }, 
        `${key}: `, typeof value === 'object' && value !== null ? renderConfig(value) : String(value)
      )
    )
  );
}

export function formatNumber(value: any, precision = 5) {
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n)) return 'N/A';
  try {
    return n.toPrecision(precision);
  } catch {
    return String(n);
  }
}

export function getStats(arr: any) {
  if (arr == null) return null;
  const nums: number[] = [];
  const collect = (x: any) => {
    if (x == null) return;
    if (Array.isArray(x)) {
      x.forEach(collect);
      return;
    }
    const n = Number(x);
    if (Number.isFinite(n)) nums.push(n);
  };
  collect(arr);
  if (nums.length === 0) return null;
  nums.sort((a, b) => a - b);
  const min = nums[0];
  const max = nums[nums.length - 1];
  const mean = nums.reduce((s, v) => s + v, 0) / nums.length;
  const median =
    nums.length % 2 === 0
      ? (nums[nums.length / 2 - 1] + nums[nums.length / 2]) / 2
      : nums[Math.floor(nums.length / 2)];
  const variance = nums.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / nums.length;
  return { min, max, mean, median, variance };
}
