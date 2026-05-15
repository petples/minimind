#!/usr/bin/env python3
"""Merge multiple LoRA weights into one (additive, for multi-stage stacking).
Usage: python merge_lora.py lora1.pth lora2.pth ... -o merged.pth
"""
import torch
import argparse

def merge_lora(paths, output):
    if not paths:
        print("No input files")
        return
    
    merged = torch.load(paths[0], map_location='cpu')
    print(f"Base: {paths[0]} ({sum(p.numel() for p in merged.values())} params)")
    
    for path in paths[1:]:
        w = torch.load(path, map_location='cpu')
        n_params = sum(p.numel() for p in w.values())
        for key, val in w.items():
            if key in merged:
                merged[key] = merged[key] + val
            else:
                merged[key] = val
        print(f"Added: {path} ({n_params} params)")
    
    torch.save(merged, output)
    print(f"Saved merged ({sum(p.numel() for p in merged.values())} params) → {output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', help='LoRA weight files')
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()
    merge_lora(args.inputs, args.output)
