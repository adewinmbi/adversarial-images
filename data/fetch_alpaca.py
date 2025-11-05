#!/usr/bin/env python3
"""
Download the Stanford Alpaca dataset if it doesn't already exist.
"""
import os
from pathlib import Path
import urllib.request
import sys


def download_alpaca_dataset():
    """Download alpaca_data.json if it doesn't exist"""
    
    # Define paths
    script_dir = Path(__file__).parent
    output_file = script_dir / "alpaca.json"
    url = "https://github.com/tatsu-lab/stanford_alpaca/raw/main/alpaca_data.json"
    
    # Check if file already exists
    if output_file.exists():
        print(f"Dataset already exists at {output_file}")
        print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        return True
    
    # Create directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Alpaca dataset from {url}...")
    print(f"Saving to {output_file}...")
    
    try:
        # Download with progress indication
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.2f}/{mb_total:.2f} MB)", end="")
        
        urllib.request.urlretrieve(url, output_file, reporthook=report_progress)
        print()  # New line after progress
        
        # Verify download
        if output_file.exists():
            file_size = output_file.stat().st_size
            print(f"Download complete! Size: {file_size / 1024 / 1024:.2f} MB")
            
            # Basic validation - check if it's valid JSON
            import json
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
                print(f"Validation successful! Found {len(data)} examples")
                return True
            except json.JSONDecodeError as e:
                print(f"Error: Downloaded file is not valid JSON: {e}")
                output_file.unlink()  # Delete invalid file
                return False
        else:
            print("Error: Download failed - file does not exist")
            return False
            
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        if output_file.exists():
            output_file.unlink()  # Clean up partial download
        return False


if __name__ == "__main__":
    success = download_alpaca_dataset()
    sys.exit(0 if success else 1)

