"""Targeted downloader for UK-DALE 2017 disaggregated dataset from CEDA Archive.

Uses HTTP Range requests to read the remote zip central directory and extract only
the required channels (mains aggregate + 4 target appliances) and metadata for Houses 1-5,
avoiding the need to download the full 3.58 GB archive.
"""

import argparse
import concurrent.futures
import os
from pathlib import Path
import struct
import sys
import time
import urllib.request
import zlib
from typing import Dict, List, Tuple


CEDA_UKDALE_ZIP_URL = (
    "https://dap.ceda.ac.uk/edc/efficiency/residential/EnergyConsumption/"
    "Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated/ukdale.zip"
)

TARGET_CHANNELS = {
    "house_1": ["channel_1.dat", "channel_5.dat", "channel_6.dat", "channel_12.dat", "channel_13.dat"],
    "house_2": ["channel_1.dat", "channel_12.dat", "channel_13.dat", "channel_14.dat", "channel_15.dat"],
    "house_3": ["channel_1.dat"],
    "house_4": ["channel_1.dat", "channel_5.dat", "channel_6.dat"],
    "house_5": ["channel_1.dat", "channel_19.dat", "channel_22.dat", "channel_23.dat", "channel_24.dat"],
}


def parse_remote_zip_central_dir(url: str) -> Tuple[int, Dict[str, Tuple[int, int, int, int]]]:
    """Reads the EOCD and Central Directory of a remote zip file via HTTP Range requests."""
    print(f"Connecting to CEDA UK-DALE repository: {url}")
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        content_length = int(resp.headers.get("Content-Length", 0))
    if content_length == 0:
        raise RuntimeError("Failed to determine remote archive size.")
    print(f"Total remote archive size: {content_length:,} bytes ({content_length / 1e9:.2f} GB)")

    # Fetch last 128 KB to locate End of Central Directory (EOCD)
    tail_size = min(131072, content_length)
    req = urllib.request.Request(
        url,
        headers={"Range": f"bytes={content_length - tail_size}-{content_length - 1}", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        tail = resp.read()

    eocd_pos = tail.rfind(b"PK\x05\x06")
    if eocd_pos == -1:
        raise RuntimeError("Could not find EOCD marker in remote zip.")

    disk_num, cd_disk, disk_entries, total_entries, cd_size, cd_offset = struct.unpack(
        "<HHHHII", tail[eocd_pos + 4 : eocd_pos + 20]
    )
    print(f"Found {total_entries} total zip entries in central directory (cd_size={cd_size:,} bytes).")

    # Fetch the central directory block
    req_cd = urllib.request.Request(
        url,
        headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req_cd, timeout=20) as resp_cd:
        cd_data = resp_cd.read()

    idx = 0
    entries = {}
    while idx < len(cd_data):
        if cd_data[idx : idx + 4] != b"PK\x01\x02":
            break
        method = struct.unpack("<H", cd_data[idx + 10 : idx + 12])[0]
        comp_size = struct.unpack("<I", cd_data[idx + 20 : idx + 24])[0]
        uncomp_size = struct.unpack("<I", cd_data[idx + 24 : idx + 28])[0]
        fn_len = struct.unpack("<H", cd_data[idx + 28 : idx + 30])[0]
        extra_len = struct.unpack("<H", cd_data[idx + 30 : idx + 32])[0]
        comment_len = struct.unpack("<H", cd_data[idx + 32 : idx + 34])[0]
        local_offset = struct.unpack("<I", cd_data[idx + 42 : idx + 46])[0]
        fn = cd_data[idx + 46 : idx + 46 + fn_len].decode("utf-8", errors="ignore")
        entries[fn] = (local_offset, comp_size, uncomp_size, method)
        idx += 46 + fn_len + extra_len + comment_len

    return content_length, entries


def stream_extract_file(url: str, entry_info: Tuple[int, int, int, int], dest_path: Path):
    """Streams and extracts a single file from remote zip using HTTP Range."""
    local_offset, comp_size, uncomp_size, method = entry_info
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch local file header (30 bytes + filename + extra)
    req_lh = urllib.request.Request(
        url,
        headers={"Range": f"bytes={local_offset}-{local_offset + 127}", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req_lh, timeout=15) as resp:
        lh_data = resp.read()

    if lh_data[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Invalid local file header signature at offset {local_offset}")

    fn_len = struct.unpack("<H", lh_data[26:28])[0]
    extra_len = struct.unpack("<H", lh_data[28:30])[0]
    payload_start = local_offset + 30 + fn_len + extra_len
    payload_end = payload_start + comp_size - 1

    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    req_payload = urllib.request.Request(
        url,
        headers={"Range": f"bytes={payload_start}-{payload_end}", "User-Agent": "Mozilla/5.0"},
    )

    CHUNK_SIZE = 1024 * 1024  # 1 MB chunk streaming
    with urllib.request.urlopen(req_payload, timeout=30) as resp:
        if method == 8:  # Deflate
            decompressor = zlib.decompressobj(-15)
            with open(temp_path, "wb") as f_out:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    decompressed = decompressor.decompress(chunk)
                    if decompressed:
                        f_out.write(decompressed)
                remaining = decompressor.flush()
                if remaining:
                    f_out.write(remaining)
        elif method == 0:  # Stored
            with open(temp_path, "wb") as f_out:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f_out.write(chunk)
        else:
            raise NotImplementedError(f"Unsupported compression method: {method}")

    temp_path.replace(dest_path)


def extract_single_entry(
    url: str,
    fn: str,
    entry_info: Tuple[int, int, int, int],
    dest_root: Path,
    max_retries: int = 4,
) -> Tuple[str, bool]:
    """Extracts a single entry with automatic retry on network errors."""
    target_file = dest_root / fn
    comp_size = entry_info[1]
    uncomp_size = entry_info[2]

    if target_file.exists() and target_file.stat().st_size == uncomp_size:
        print(f"Skipped (already exists): {fn} ({uncomp_size / 1e6:.2f} MB)")
        return fn, True

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            stream_extract_file(url, entry_info, target_file)
            dur = time.time() - t0
            rate = (comp_size / 1e6) / max(0.01, dur)
            print(f"Done: {fn} ({comp_size / 1e6:.2f} MB comp -> {uncomp_size / 1e6:.2f} MB uncomp) in {dur:.1f}s ({rate:.2f} MB/s)")
            return fn, True
        except Exception as e:
            print(f"Attempt {attempt}/{max_retries} failed for {fn}: {e}")
            if target_file.with_suffix(target_file.suffix + ".tmp").exists():
                try:
                    target_file.with_suffix(target_file.suffix + ".tmp").unlink()
                except Exception:
                    pass
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                raise e


def download_ukdale(
    dest_dir: str = "data/raw/ukdale",
    target_only: bool = True,
    url: str = CEDA_UKDALE_ZIP_URL,
    workers: int = 8,
):
    """Downloads and extracts target channels and labels for UK-DALE Houses 1-5 concurrently."""
    dest_root = Path(dest_dir)
    dest_root.mkdir(parents=True, exist_ok=True)

    _, entries = parse_remote_zip_central_dir(url)

    # 1. Identify files to extract
    files_to_extract: List[str] = []
    # Always extract labels and metadata
    for fn in entries:
        if "labels.dat" in fn or fn.startswith("metadata/"):
            files_to_extract.append(fn)

    if target_only:
        for house, ch_list in TARGET_CHANNELS.items():
            for ch in ch_list:
                fn = f"{house}/{ch}"
                if fn in entries:
                    files_to_extract.append(fn)
                else:
                    print(f"Warning: {fn} not found in remote archive entries.")
    else:
        files_to_extract = list(entries.keys())

    total_comp_bytes = sum(entries[fn][1] for fn in files_to_extract)
    total_uncomp_bytes = sum(entries[fn][2] for fn in files_to_extract)
    print(f"\nTarget extraction list: {len(files_to_extract)} files (using {workers} concurrent workers)")
    print(f"Total download size: {total_comp_bytes / 1e6:.2f} MB ({total_comp_bytes / 1e9:.2f} GB)")
    print(f"Total uncompressed size: {total_uncomp_bytes / 1e6:.2f} MB ({total_uncomp_bytes / 1e9:.2f} GB)\n")

    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extract_single_entry, url, fn, entries[fn], dest_root): fn
            for fn in files_to_extract
        }
        for future in concurrent.futures.as_completed(futures):
            fn = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"ERROR: {fn} failed: {exc}")
                raise exc

    total_elapsed = time.time() - t_start
    print(f"\n✅ All {len(files_to_extract)} files successfully verified in {dest_root}")
    print(f"Total time: {total_elapsed / 60:.2f} minutes.")


def download_ukdale_metadata(raw_dir: str = "data/raw/ukdale") -> None:
    """Helper alias for metadata extraction."""
    download_ukdale(dest_dir=raw_dir, target_only=True, workers=4)


def download_ukdale_targets(raw_dir: str = "data/raw/ukdale", max_workers: int = 6) -> None:
    """Helper alias for target channel extraction."""
    download_ukdale(dest_dir=raw_dir, target_only=True, workers=max_workers)


def main():
    parser = argparse.ArgumentParser(description="Download & extract UK-DALE dataset channels via HTTP Range")
    parser.add_argument("--dest_dir", type=str, default="data/raw/ukdale", help="Output directory")
    parser.add_argument("--full", action="store_true", help="Extract all files rather than only target channels")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent download threads")
    args = parser.parse_args()

    download_ukdale(dest_dir=args.dest_dir, target_only=not args.full, workers=args.workers)


if __name__ == "__main__":
    main()
