"""
File: mmrs1m_structure.py

Description:
Streams the MMRS-1M dataset archives (split tar.gz files) directly from 
HuggingFace to map the internal directory structure.

Notes:
This script does NOT save files to disk. It pipes the download stream directly 
into 'tar' to list filenames, allowing for rapid structure analysis of large datasets.
"""

import io
import threading
import subprocess
import requests
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_URL = "https://huggingface.co/datasets/initiacms/MMRS-1M/resolve/main"
# The dataset is split into 7 parts (0-6)
PARTS = [f"data.tar.gz.{i}" for i in range(7)] 

# -----------------------------------------------------------------------------
# Stream Reader Logic
# -----------------------------------------------------------------------------

def read_tar_output(proc, found_dirs):
    """
    Background worker that reads the stdout of the running 'tar' process.
    It parses file paths in real-time to identify unique dataset directories.
    
    Args:
        proc (subprocess.Popen): The active tar process.
        found_dirs (set): A shared set to track discovered directories.
    """
    # Wrap the binary stdout in a text wrapper to read lines comfortably
    text_stdout = io.TextIOWrapper(proc.stdout, encoding='utf-8', errors='replace')
    
    try:
        for line in text_stdout:
            path = line.strip()
            parts = path.split("/")
            
            # We filter for depth 3 to find dataset roots: data/task/dataset_name
            # Example: data/detection/AIR-SARShip-2.0/images/file.jpg
            if len(parts) >= 3:
                dataset_key = "/".join(parts[:3])
                
                if dataset_key not in found_dirs:
                    found_dirs.add(dataset_key)
                    
                    # Format the notification message
                    msg = f"Found: {dataset_key}"
                    
                    # Apply heuristics to highlight specific modalities (SAR/IR)
                    lower_key = dataset_key.lower()
                    if any(x in lower_key for x in ['sar', 'radar', 'ssdd', 'mstar', 'hrs', 'fusar', 'air-sar']):
                        msg += "  🟢 [SAR DETECTED]"
                    if any(x in lower_key for x in ['ir', 'infrared', 'thermal', 'flir', 'kaist', 'hit-uav']):
                        msg += "  🔴 [IR DETECTED]"
                        
                    # Use tqdm.write to print safely without breaking the progress bar
                    tqdm.write(msg)

    except Exception as e:
        tqdm.write(f"[ERROR] Reader thread encountered an issue: {e}")

# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------

def main():
    print("[INFO] Starting MMRS-1M structure mapping (Streaming Mode)...")
    print("[INFO] NOTE: Headers will be downloaded and piped to memory; no disk space will be used.")
    
    # 1. Initialize the tar subprocess
    # 'tar -tzf -': 
    #   -t: list contents
    #   -z: filter through gzip
    #   -f -: read from stdin (which we will feed via Python)
    tar_cmd = ["tar", "-tzf", "-"]
    
    proc = subprocess.Popen(
        tar_cmd, 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.DEVNULL, # Suppress tar warnings/errors
        bufsize=1024*1024          # 1MB buffer for efficiency
    )

    found_dirs = set()
    
    # 2. Start the output reader thread
    # This runs in parallel, listening to tar while the main thread handles the download.
    reader_thread = threading.Thread(target=read_tar_output, args=(proc, found_dirs))
    reader_thread.daemon = True
    reader_thread.start()

    try:
        # 3. Iterate through all dataset parts
        global_bar = tqdm(PARTS, desc="Overall Progress", position=0)
        
        for part_name in global_bar:
            url = f"{BASE_URL}/{part_name}"
            
            # Initiate streaming request
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                # 4. Feed data to subprocess
                with tqdm(
                    total=total_size, 
                    desc=f"Streaming {part_name}", 
                    unit='B', 
                    unit_scale=True, 
                    unit_divisor=1024,
                    leave=False, # Clean up bar after file finishes
                    position=1
                ) as file_bar:
                    
                    # Read chunks from HTTP and write to tar process stdin
                    for chunk in r.iter_content(chunk_size=64*1024):
                        if chunk:
                            try:
                                if proc.stdin is not None:
                                    proc.stdin.write(chunk)
                            except BrokenPipeError:
                                tqdm.write("[WARN] Tar process closed unexpectedly (possibly finished).")
                                return
                            
                            file_bar.update(len(chunk))
            
            # Flush stdin buffer after each file to ensure data is processed
            try:
                if proc.stdin:
                    proc.stdin.flush()
            except: 
                pass

    except KeyboardInterrupt:
        tqdm.write("\n\n[INFO] Process stopped by user.")
    except Exception as e:
        tqdm.write(f"\n[ERROR] Pipeline failed: {e}")
    finally:
        # 5. Cleanup Resources
        if proc.stdin:
            try: 
                proc.stdin.close()
            except: 
                pass
        proc.terminate()
        
        # Allow the reader thread to finish printing pending lines
        reader_thread.join(timeout=1)
        print("\n[INFO] Mapping complete.")

if __name__ == "__main__":
    main()