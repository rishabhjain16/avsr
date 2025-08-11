#!/usr/bin/env python3
"""
Simple script to print detokenized samples from CSV file

Usage:
    python check_samples.py /path/to/your/labels/train.csv
    usage : python check_samples.py /data/ssd2/data_rishabh/lrs2_rf/labels/lrs2_train_transcript_lengths_seg16s.csv 
"""

import sys
import torch
from datamodule.transforms import TextTransform

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_samples.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Load TextTransform
    text_transform = TextTransform()
    
    print(f"Reading samples from: {csv_file}")
    print(f"Vocabulary size: {len(text_transform.token_list)}")
    print("=" * 80)
    
    # Read and process first 10 lines
    with open(csv_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:  # Only first 10 samples
                break
                
            parts = line.strip().split(',')
            if len(parts) < 4:
                print(f"Sample {i+1}: Invalid format")
                continue
                
            dataset_name = parts[0]
            rel_path = parts[1] 
            input_length = parts[2]
            token_ids_str = parts[3]
            
            # Convert token IDs string to tensor
            token_ids = [int(x) for x in token_ids_str.split()]
            token_tensor = torch.tensor(token_ids)
            
            # Detokenize
            text = text_transform.post_process(token_tensor)
            
            print(f"Sample {i+1}:")
            print(f"  Dataset: {dataset_name}")
            print(f"  File: {rel_path}")
            print(f"  Length: {input_length}")
            print(f"  Tokens: {token_ids[:15]}{'...' if len(token_ids) > 15 else ''}")
            print(f"  Text: '{text}'")
            print()

if __name__ == "__main__":
    main()
