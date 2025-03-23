#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import torch
import torchaudio
from snac import SNAC
from tqdm import tqdm
import concurrent.futures

################################################################################
# Configuration constants at the top
################################################################################
INPUT_FOLDER = "/mnt/weka/home/laion/talent-raw/"
SAMPLE_RATE = 24000
PROCESSES_PER_GPU = 8  # e.g. 5 processes per GPU

################################################################################
# Flattening function (unchanged)
################################################################################
def flatten_tensors_adjusted(tensors):
    """Flatten SNAC codes (3 or 4 levels) into a single list of ints,
    weaving them in a special pattern with '#' separators."""
    flattened_list = []
    num_levels = len(tensors)

    if num_levels == 3:
        for i in range(tensors[0].size(1)):
            flattened_list.append("#")
            flattened_list.append(tensors[0][0][i].item())   # from level0
            for j in range(2):
                flattened_list.append(tensors[1][0][j + i*2].item())  # from level1
                for k in range(2):
                    flattened_list.append(tensors[2][0][k + j*2 + i*4].item())  # from level2

    elif num_levels == 4:
        for i in range(tensors[0].size(1)):
            flattened_list.append("#")
            flattened_list.append(tensors[0][0][i].item())   # from level0
            for j in range(2):
                flattened_list.append(tensors[1][0][j + i*2].item())  # from level1
                for k in range(2):
                    flattened_list.append(tensors[2][0][k + j*2 + i*4].item())  # from level2
                    for l in range(2):
                        flattened_list.append(tensors[3][0][l + k*2 + j*4 + i*8].item())
    else:
        raise ValueError(f"Expected 3 or 4 code levels, got {num_levels}.")

    return flattened_list

################################################################################
# find_pairs - unchanged
################################################################################
def find_pairs(root_dir):
    """Scans `root_dir` recursively to find matching .mp3/.json pairs."""
    pairs = []
    for dp, _, files in os.walk(root_dir):
        mp3s  = [f for f in files if f.lower().endswith(".mp3")]
        jsons = [f for f in files if f.lower().endswith(".json")]
        used  = set()
        for mp3 in mp3s:
            base = mp3[:-4]  # remove .mp3
            matches = [(j, len(os.path.commonprefix([base, j[:-5]])))
                       for j in jsons if j not in used]
            if matches:
                matches.sort(key=lambda x: x[1], reverse=True)
                chosen = matches[0][0]
                used.add(chosen)
                pairs.append({
                    "mp3":  os.path.join(dp, mp3),
                    "json": os.path.join(dp, chosen)
                })
    print(f"[INFO] Found {len(pairs)} pairs in {root_dir}.")
    return pairs

################################################################################
# Worker function
################################################################################
def encode_on_gpu(gpu_id, chunk, pbar, sample_rate=24000):
    """
    Each worker uses a single GPU device (gpu_id),
    processes its subset of (mp3,json) pairs,
    updates the global pbar after each file.
    """
    # If gpu_id is an integer, build device string
    if isinstance(gpu_id, int):
        device = torch.device(f"cuda:{gpu_id}")
    else:
        # e.g. 'cpu' fallback
        device = torch.device("cpu")

    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

    for pair in chunk:
        mp3_path  = pair["mp3"]
        json_path = pair["json"]

        # Load audio
        audio, sr = torchaudio.load(mp3_path)
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            audio = torchaudio.functional.resample(audio, sr, sample_rate)

        audio = audio.to(device).unsqueeze(0)  # shape [1,1,T]

        with torch.inference_mode():
            _, codes = model(audio)

        flattened_codes = flatten_tensors_adjusted(codes)

        # Try loading existing JSON
        json_data = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
            except json.JSONDecodeError:
                # Overwrite if malformed
                json_data = {}

        duration_sec = audio.shape[-1] / sample_rate
        json_data["playtime_sec"]  = duration_sec
        json_data["SNAC 24 kHz"]   = flattened_codes

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        # Global progress bar update
        pbar.update(1)

    del model
    torch.cuda.empty_cache()

################################################################################
# Main
################################################################################
def main():
    # 1) Find all pairs
    pairs = find_pairs(INPUT_FOLDER)
    if not pairs:
        print("[WARN] No .mp3/.json pairs found. Exiting.")
        return

    # 2) Check how many GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("[WARN] No GPU found! Using CPU only.")
        num_gpus = 1  # We'll run on CPU only
    print(f"[INFO] Found {num_gpus} GPU(s) available.")

    # 3) We want 5 processes per GPU => total processes
    total_processes = num_gpus * PROCESSES_PER_GPU
    print(f"[INFO] Launching {total_processes} worker(s) total "
          f"({PROCESSES_PER_GPU} per GPU).")

    # 4) We chunk the file list into total_processes slices
    chunk_size = math.ceil(len(pairs) / total_processes)
    chunks = []
    for i in range(total_processes):
        start = i * chunk_size
        end   = start + chunk_size
        chunk = pairs[start:end]
        if not chunk:
            break
        chunks.append(chunk)

    # 5) We create a global progress bar over all pairs
    with tqdm(total=len(pairs), desc="Encoding SNAC", ncols=80) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=total_processes) as executor:
            futs = []
            for i, chunk in enumerate(chunks):
                # The GPU ID is determined by i // PROCESSES_PER_GPU
                # e.g. i=0..4 => GPU0, i=5..9 => GPU1, etc.
                if torch.cuda.is_available():
                    gpu_id = i // PROCESSES_PER_GPU
                else:
                    gpu_id = "cpu"
                futs.append(executor.submit(encode_on_gpu, gpu_id, chunk, pbar, SAMPLE_RATE))

            # Wait for all
            concurrent.futures.wait(futs)

    print("✅ Done flattening and saving all SNAC codes.")


if __name__ == "__main__":
    main()
