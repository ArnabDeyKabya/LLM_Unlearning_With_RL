"""
Load and Explore TOFU Dataset
Task Of Fictitious Unlearning - Maini et al., 2024

Dataset Details:
- 200 fictitious authors
- ~20 QA samples per author
- Test set: 100 real people, 117 world knowledge items
- Purpose: Explore fictitious personal information unlearning and test generalization
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd

def load_tofu_data():
    """
    Load TOFU dataset from HuggingFace
    """
    print("=" * 80)
    print("Loading TOFU Dataset")
    print("=" * 80)
    
    try:
        # Load the TOFU dataset from HuggingFace
        # The dataset is available at: locuslab/TOFU
        print("\n📥 Downloading TOFU dataset from HuggingFace...")
        dataset = load_dataset("locuslab/TOFU")
        
        print("\n✓ Dataset loaded successfully!")
        print(f"\n📊 Dataset splits: {list(dataset.keys())}")
        
        # Explore the dataset structure
        for split_name, split_data in dataset.items():
            print(f"\n{split_name.upper()}:")
            print(f"  - Number of samples: {len(split_data)}")
            print(f"  - Features: {list(split_data.features.keys())}")
            
            # Show a sample
            if len(split_data) > 0:
                print(f"\n  Sample from {split_name}:")
                sample = split_data[0]
                for key, value in sample.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"    {key}: {value[:100]}...")
                    else:
                        print(f"    {key}: {value}")
        
        return dataset
    
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("\nTrying alternative approach...")
        
        # Alternative: Load specific subsets
        try:
            print("\n📥 Loading TOFU forget set...")
            forget_set = load_dataset("locuslab/TOFU", "forget10")
            print(f"✓ Loaded forget set: {len(forget_set['train'])} samples")
            
            print("\n📥 Loading TOFU retain set...")
            retain_set = load_dataset("locuslab/TOFU", "retain90")
            print(f"✓ Loaded retain set: {len(retain_set['train'])} samples")
            
            return {"forget": forget_set, "retain": retain_set}
        except Exception as e2:
            print(f"\n❌ Error with alternative approach: {e2}")
            return None

def save_tofu_data(dataset, output_dir="TOFU_Datasets"):
    """
    Save TOFU dataset to local files for easier access
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving dataset to {output_dir}/")
    
    if dataset is None:
        print("❌ No dataset to save")
        return
    
    try:
        for split_name, split_data in dataset.items():
            if hasattr(split_data, 'to_pandas'):
                # If it's a HF dataset
                df = split_data.to_pandas()
            elif hasattr(split_data, 'train'):
                # If it's a DatasetDict
                df = split_data['train'].to_pandas()
            else:
                continue
            
            # Save as CSV
            csv_path = output_path / f"{split_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  ✓ Saved {csv_path}")
            
            # Save as JSON
            json_path = output_path / f"{split_name}.json"
            df.to_json(json_path, orient='records', indent=2)
            print(f"  ✓ Saved {json_path}")
        
        print(f"\n✓ All data saved to {output_dir}/")
    
    except Exception as e:
        print(f"\n❌ Error saving data: {e}")

def analyze_tofu_structure(dataset):
    """
    Analyze the structure of TOFU dataset
    """
    print("\n" + "=" * 80)
    print("TOFU Dataset Analysis")
    print("=" * 80)
    
    if dataset is None:
        print("❌ No dataset to analyze")
        return
    
    for split_name, split_data in dataset.items():
        print(f"\n📊 Analyzing {split_name}...")
        
        # Get the actual data
        if hasattr(split_data, 'to_pandas'):
            df = split_data.to_pandas()
        elif hasattr(split_data, 'train'):
            df = split_data['train'].to_pandas()
        else:
            continue
        
        print(f"  Total samples: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        # Analyze content
        if 'question' in df.columns:
            print(f"\n  Sample questions:")
            for i, q in enumerate(df['question'].head(3)):
                print(f"    {i+1}. {q[:100]}...")
        
        if 'answer' in df.columns:
            print(f"\n  Sample answers:")
            for i, a in enumerate(df['answer'].head(3)):
                print(f"    {i+1}. {a[:100]}...")
        
        # Check for author information
        if 'author' in df.columns:
            unique_authors = df['author'].nunique()
            print(f"\n  Unique authors: {unique_authors}")

if __name__ == "__main__":
    # Load the dataset
    dataset = load_tofu_data()
    
    # Analyze structure
    if dataset:
        analyze_tofu_structure(dataset)
        
        # Save to local files
        save_tofu_data(dataset)
    
    print("\n" + "=" * 80)
    print("✓ TOFU Dataset Loading Complete!")
    print("=" * 80)
