import os
import json
import random
import traceback
from tqdm import tqdm

INPUT_FOLDER = "/mnt/weka/home/laion/talent-raw/de_alloy_intense_awe_wonder_and_an_awestruck_feeling_vocalbursts_fixed"
OUTPUT_FOLDER = "/mnt/weka/home/laion/talent-snac-pairs"

MAX_SNAC_TOKENS_INPUT = 800  # Only include up to 800 tokens from the reference SNAC array in the <snac> tag.
FILES_PER_SUBFOLDER = 1000   # Each subfolder will hold up to 1000 files.

def ensure_folder_exists(folder):
    """Create the folder if it does not exist."""
    if not os.path.exists(folder):
        os.makedirs(folder)

def collect_all_folders(base_folder):
    """
    Recursively collect all folder paths (including the base_folder itself).
    Returns a list of absolute paths.
    """
    folders = []
    for root, dirs, files in os.walk(base_folder):
        folders.append(root)
    return folders

def load_and_validate_json(filepath):
    """
    Load JSON from `filepath`.
    Returns a dict if successful and valid, otherwise None.
    Valid means:
      - JSON is parseable.
      - Contains "whisper_transcription" (>= 3 chars).
      - Contains "SNAC 24 kHz" (non-empty list).
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        print(f"[ERROR] Failed to parse JSON from {filepath}.\n{traceback.format_exc()}")
        return None

    if "whisper_transcription" not in data or "SNAC 24 kHz" not in data:
        return None

    whisper_text = data["whisper_transcription"].strip()
    if len(whisper_text) < 3:
        return None

    snac_24khz = data["SNAC 24 kHz"]
    if not isinstance(snac_24khz, list) or len(snac_24khz) == 0:
        return None

    playtime_sec = data.get("playtime_sec", 0.0)
    if not isinstance(playtime_sec, (float, int)):
        playtime_sec = 0.0

    return {
        "filepath": filepath,
        "data": data,
    }

def snac_array_to_string(snac_array, max_len=None):
    """
    Convert a list of tokens into a space-separated string.
    If `max_len` is provided, only include up to that many tokens.
    """
    if max_len is not None:
        truncated = snac_array[:max_len]
    else:
        truncated = snac_array
    return " ".join(str(x) for x in truncated)

def compute_chars_per_second(text, playtime):
    """
    Return the integer-rounded (chars / seconds).
    Avoid division by zero by using playtime=1.0 if <= 0.
    """
    if playtime <= 0:
        playtime = 1.0
    return round(len(text) / playtime)

def main():
    ensure_folder_exists(OUTPUT_FOLDER)
    
    # 1. Recursively collect all subfolders from the input folder.
    all_folders = collect_all_folders(INPUT_FOLDER)

    # 2. Gather valid JSON files, folder by folder.
    folder_to_records = {}
    print("Scanning folders and collecting valid JSON files...")
    for folder in all_folders:
        valid_records = []
        for fname in os.listdir(folder):
            if fname.lower().endswith(".json"):
                abs_path = os.path.join(folder, fname)
                if os.path.isfile(abs_path):
                    record = load_and_validate_json(abs_path)
                    if record is not None:
                        valid_records.append(record)
        if valid_records:
            folder_to_records[folder] = valid_records

    total_targets = sum(len(records) for records in folder_to_records.values())
    print(f"Found {len(all_folders)} folders and a total of {total_targets} valid target JSON files.")
    print("Now generating output pairs...")

    global_counter = 1  # Global counter for output file naming.

    with tqdm(total=total_targets, desc="Processing", unit="file") as pbar:
        for folder, records in folder_to_records.items():
            if len(records) < 2:
                for _ in records:
                    pbar.update(1)
                continue

            for target_record in records:
                pbar.update(1)
                target_data = target_record["data"]
                target_snac = target_data["SNAC 24 kHz"]
                target_whisper = target_data["whisper_transcription"].strip()
                target_playtime = target_data.get("playtime_sec", 0.0)
                
                # Select a different valid reference from the same folder.
                reference_record = None
                for _ in range(10):
                    ref_candidate = random.choice(records)
                    if ref_candidate["filepath"] != target_record["filepath"]:
                        reference_record = ref_candidate
                        break
                if reference_record is None:
                    continue

                ref_data = reference_record["data"]
                ref_whisper = ref_data["whisper_transcription"].strip()
                ref_snac_array = ref_data["SNAC 24 kHz"]
                ref_playtime = ref_data.get("playtime_sec", 0.0)

                ref_snac_str = snac_array_to_string(ref_snac_array, max_len=MAX_SNAC_TOKENS_INPUT)
                target_snac_str = snac_array_to_string(target_snac, max_len=None)
                # Wrap target SNAC tokens with <snac2> tags.
                target_snac_str = f"<snac2>{target_snac_str}</snac2>"

                # Compute speeds.
                ref_speed = compute_chars_per_second(ref_whisper, ref_playtime)
                target_speed = compute_chars_per_second(target_whisper, target_playtime)

                # Build the input XML.
                input_str = (
                    f"<speed>{ref_speed}</speed><text1>{ref_whisper}</text1>"
                    f"<snac> {ref_snac_str} </snac>"
                    f"<speed>{target_speed}</speed><text2>{target_whisper}</text2>"
                )

                output_dict = {
                    "input": input_str,
                    "output": target_snac_str
                }

                # Determine subfolder based on global_counter.
                subfolder_index = (global_counter - 1) // FILES_PER_SUBFOLDER + 1
                subfolder_path = os.path.join(OUTPUT_FOLDER, str(subfolder_index))
                ensure_folder_exists(subfolder_path)

                out_name = f"{global_counter}.json"
                out_path = os.path.join(subfolder_path, out_name)

                try:
                    with open(out_path, "w", encoding="utf-8") as f_out:
                        json.dump(output_dict, f_out, ensure_ascii=False, indent=2)
                except Exception:
                    print(f"[ERROR] Writing output JSON to {out_path} failed.\n{traceback.format_exc()}")
                    continue

                global_counter += 1

    print("All done. The paired output JSON files should be in subfolders of:", OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
