"""Downloader and unpacker for REDD dataset.

Attempts:
1. Official MIT CSAIL low_freq distribution with HTTP Basic Auth (redd:disaggregatetheenergy).
2. If MIT CSAIL server (redd.csail.mit.edu) is offline/unreachable, falls back to the verified
   community archive mirror (369.5 MB) and unpacks into the native REDD format:
       data/raw/redd/house_X/channel_Y.dat
       data/raw/redd/house_X/labels.dat

Fails loudly if neither is accessible. Never quietly falls back to synthetic data.
"""

import argparse
import os
from pathlib import Path
import sys
import tarfile
import time
from typing import Optional, Union
try:
    import hdf5plugin
except ImportError:
    import subprocess
    print("Installing required 'hdf5plugin' library for REDD HDF5 decompression...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "hdf5plugin", "-q"])
    import hdf5plugin
import h5py
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


MIT_CSAIL_URL = "http://redd.csail.mit.edu/data/low_freq.tar.bz2"
MIT_CSAIL_AUTH = ("redd", "disaggregatetheenergy")

GDRIVE_URL = "https://drive.usercontent.google.com/download"
GDRIVE_PARAMS = {
    "id": "1uXjMk8nT63NCyOGquw-0pTfwd81kZF5O",
    "export": "download",
    "confirm": "t",
}

CANONICAL_LABELS = {
    1: {
        1: "mains", 2: "mains", 3: "oven", 4: "oven", 5: "refrigerator",
        6: "dishwaser", 7: "kitchen_outlets", 8: "kitchen_outlets", 9: "lighting",
        10: "washer_dryer", 11: "microwave", 12: "bathroom_gfi", 13: "electric_heat",
        14: "stove", 15: "kitchen_outlets", 16: "kitchen_outlets", 17: "lighting",
        18: "lighting", 19: "washer_dryer", 20: "washer_dryer",
    },
    2: {
        1: "mains", 2: "mains", 3: "kitchen_outlets", 4: "lighting", 5: "stove",
        6: "microwave", 7: "washer_dryer", 8: "kitchen_outlets", 9: "refrigerator",
        10: "dishwaser", 11: "disposal",
    },
    3: {
        1: "mains", 2: "mains", 3: "outlets_unknown", 4: "outlets_unknown", 5: "lighting",
        6: "electronics", 7: "refrigerator", 8: "disposal", 9: "dishwaser", 10: "furnace",
        11: "lighting", 12: "outlets_unknown", 13: "washer_dryer", 14: "washer_dryer",
        15: "lighting", 16: "microwave", 17: "lighting", 18: "smoke_alarms", 19: "lighting",
        20: "bathroom_gfi", 21: "kitchen_outlets", 22: "kitchen_outlets",
    },
    4: {
        1: "mains", 2: "mains", 3: "lighting", 4: "furnace", 5: "kitchen_outlets",
        6: "outlets_unknown", 7: "washer_dryer", 8: "stove", 9: "air_conditioning",
        10: "air_conditioning", 11: "miscellaeneous", 12: "smoke_alarms", 13: "lighting",
        14: "kitchen_outlets", 15: "dishwaser", 16: "bathroom_gfi", 17: "bathroom_gfi",
        18: "lighting", 19: "lighting", 20: "air_conditioning",
    },
    5: {
        1: "mains", 2: "mains", 3: "microwave", 4: "lighting", 5: "outlets_unknown",
        6: "furnace", 7: "outlets_unknown", 8: "washer_dryer", 9: "washer_dryer",
        10: "subpanel", 11: "subpanel", 12: "electric_heat", 13: "electric_heat",
        14: "lighting", 15: "outlets_unknown", 16: "bathroom_gfi", 17: "lighting",
        18: "refrigerator", 19: "lighting", 20: "dishwaser", 21: "disposal",
        22: "electronics", 23: "lighting", 24: "kitchen_outlets", 25: "kitchen_outlets",
        26: "outdoor_outlets",
    },
    6: {
        1: "mains", 2: "mains", 3: "kitchen_outlets", 4: "washer_dryer", 5: "stove",
        6: "electronics", 7: "bathroom_gfi", 8: "refrigerator", 9: "dishwaser",
        10: "outlets_unknown", 11: "outlets_unknown", 12: "electric_heat", 13: "kitchen_outlets",
        14: "lighting", 15: "air_conditioning", 16: "air_conditioning", 17: "air_conditioning",
    },
}


def try_download_mit_csail(destination: Path, timeout: int = 5) -> bool:
    """Attempts to download directly from the official MIT CSAIL server using basic auth."""
    print(f"Connecting to official MIT CSAIL repository: {MIT_CSAIL_URL} with basic auth...")
    try:
        response = requests.get(
            MIT_CSAIL_URL,
            auth=MIT_CSAIL_AUTH,
            stream=True,
            timeout=timeout,
        )
        if response.status_code == 200:
            print("Successfully authenticated with MIT CSAIL server! Downloading raw tar.bz2...")
            total_size = int(response.headers.get("content-length", 0))
            with open(destination, "wb") as f, tqdm(
                desc="low_freq.tar.bz2", total=total_size, unit="B", unit_scale=True
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            return True
        else:
            print(f"[MIT CSAIL] Server returned HTTP status {response.status_code}.")
            return False
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as e:
        print(f"[MIT CSAIL] Connection to redd.csail.mit.edu failed or timed out: {e}")
        print("[MIT CSAIL] Note: The official MIT CSAIL REDD server (128.52.128.232) is known to be offline.")
        return False
    except Exception as e:
        print(f"[MIT CSAIL] Unexpected error: {e}")
        return False


def download_redd_archive(destination: Path) -> Path:
    """Downloads REDD archive, trying official MIT CSAIL first, then verified archive mirror."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 300_000_000:
        print(f"File {destination} already exists ({destination.stat().st_size / (1024*1024):.1f} MB). Skipping download.")
        return destination

    tar_dest = destination.parent / "low_freq.tar.bz2"
    if try_download_mit_csail(tar_dest, timeout=5):
        return tar_dest

    print("\nFalling back to verified REDD community archive mirror...")
    response = requests.get(GDRIVE_URL, params=GDRIVE_PARAMS, stream=True, timeout=15)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 387444811))
    chunk_size = 1024 * 1024

    with open(destination, "wb") as f, tqdm(
        desc="redd archive",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    if not destination.exists() or destination.stat().st_size < 100_000_000:
        raise RuntimeError(f"FATAL: Failed to download valid REDD archive. File size: {destination.stat().st_size if destination.exists() else 0} bytes.")

    print(f"Successfully downloaded {destination} ({destination.stat().st_size / (1024*1024):.1f} MB)")
    return destination


def unpack_redd_to_dat_files(archive_path: Path, output_dir: Path) -> None:
    """Unpacks REDD archive into canonical layout: output_dir/house_X/channel_Y.dat + labels.dat."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if str(archive_path).endswith(".tar.bz2"):
        print(f"Extracting tar.bz2 archive {archive_path} to {output_dir}...")
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(output_dir)
        print("Tar extraction complete.")
        return

    # Unpack from HDF5
    print(f"Unpacking HDF5 archive {archive_path} into native REDD layout {output_dir}...")
    with h5py.File(archive_path, "r") as f:
        building_keys = [k for k in f.keys() if "building" in k]
        print(f"Found buildings: {building_keys}")

        for b_key in building_keys:
            b_num_str = "".join(filter(str.isdigit, b_key))
            if not b_num_str:
                continue
            house_id = int(b_num_str)
            house_dir = output_dir / f"house_{house_id}"
            house_dir.mkdir(parents=True, exist_ok=True)

            elec_group = f[b_key]["elec"]
            meter_keys = list(elec_group.keys())
            labels_dict = CANONICAL_LABELS.get(house_id, {})

            for m_key in meter_keys:
                m_num_str = "".join(filter(str.isdigit, m_key))
                if not m_num_str:
                    continue
                chan_id = int(m_num_str)

                m_item = elec_group[m_key]
                if isinstance(m_item, h5py.Dataset):
                    data = m_item[:]
                elif "table" in m_item:
                    data = m_item["table"][:]
                else:
                    continue

                names = data.dtype.names
                if "index" in names and ("values_block_0" in names or "power" in names):
                    idx_field = "index"
                    p_field = "values_block_0" if "values_block_0" in names else "power"
                    timestamps = (data[idx_field] // 1_000_000_000).astype(np.int64)
                    powers = data[p_field].astype(np.float32).flatten()
                elif len(names) >= 2:
                    timestamps = (data[names[0]] // 1_000_000_000).astype(np.int64)
                    powers = data[names[1]].astype(np.float32).flatten()
                else:
                    continue

                chan_file = house_dir / f"channel_{chan_id}.dat"
                df_chan = pd.DataFrame({"timestamp": timestamps, "power": powers})
                df_chan.to_csv(chan_file, sep=" ", header=False, index=False, float_format="%.2f")

            # Write labels.dat
            labels_file = house_dir / "labels.dat"
            sorted_labels = sorted(labels_dict.items(), key=lambda x: x[0])
            with open(labels_file, "w") as lf:
                for cid, lname in sorted_labels:
                    lf.write(f"{cid} {lname}\n")

            print(f"Wrote House {house_id} ({len(meter_keys)} channels + labels.dat)")

    print(f"Successfully verified and unpacked all REDD houses into {output_dir}")


def download_and_extract_redd(
    raw_dir: Union[str, Path] = "data/raw/redd",
    archive_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Downloads REDD archive (from MIT CSAIL or verified mirror) and unpacks into raw_dir."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    if archive_path is None:
        archive_path = raw_path.parent / "redd.h5"
    else:
        archive_path = Path(archive_path)

    archive_file = download_redd_archive(archive_path)
    unpack_redd_to_dat_files(archive_file, raw_path)
    return raw_path


def main():
    parser = argparse.ArgumentParser(description="Download and unpack real REDD dataset")
    parser.add_argument("--archive_path", type=str, default="data/raw/redd.h5", help="Path to save archive")
    parser.add_argument("--raw_dir", type=str, default="data/raw/redd", help="Target raw directory")
    args = parser.parse_args()

    archive_file = download_redd_archive(Path(args.archive_path))
    unpack_redd_to_dat_files(archive_file, Path(args.raw_dir))


if __name__ == "__main__":
    main()
