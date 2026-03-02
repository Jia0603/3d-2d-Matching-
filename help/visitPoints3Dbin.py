import argparse
import numpy as np
from pathlib import Path
from hloc.utils.read_write_model import read_model

def inspect_tracks(model_path):
    print(f"Loading model from: {model_path}")
    
    # 1. Read the binary file
    # 'read_model' returns three dictionaries: cameras, images, points3D
    cameras, images, points3D = read_model(model_path, ext=".bin")
    
    print(f"Successfully loaded {len(points3D)} 3D points.")
    
    # 2. Collect Track Lengths
    # Every point3D object has an attribute .image_ids which is the list of images seeing it.
    track_lengths = []
    for p_id, p in points3D.items():
        length = len(p.image_ids)
        track_lengths.append(length)
    
    track_lengths = np.array(track_lengths)
    
    # 3. Print Statistics
    print("-" * 30)
    print("TRACK LENGTH STATISTICS")
    print("-" * 30)
    print(f"Minimum Track Length: {np.min(track_lengths)}")
    print(f"Maximum Track Length: {np.max(track_lengths)}")
    print(f"Mean Track Length:    {np.mean(track_lengths):.2f}")
    print(f"Median Track Length:  {np.median(track_lengths)}")
    print("-" * 30)
    
    # 4. Distribution (The Histogram)
    print("DISTRIBUTION:")
    print(f"Points with 2 views:       {np.sum(track_lengths == 2)} ({np.sum(track_lengths == 2)/len(track_lengths)*100:.1f}%)")
    print(f"Points with 3-9 views:     {np.sum((track_lengths >= 3) & (track_lengths < 10))}")
    print(f"Points with 10-20 views:   {np.sum((track_lengths >= 10) & (track_lengths <= 20))}")
    print(f"Points with > 20 views:    {np.sum(track_lengths > 20)}  <-- PROOF of chaining")
    print(f"Points with > 50 views:    {np.sum(track_lengths > 50)}")
    
    # 5. Show a specific example of a long track
    # Find the point with the maximum length
    max_idx = np.argmax(track_lengths)
    # The dictionary keys are not necessarily 0..N, so we find the ID manually
    # (Just taking a shortcut for the example)
    max_len = np.max(track_lengths)
    
    print("-" * 30)
    print(f"EXAMPLE: A point with {max_len} views")
    
    for p_id, p in points3D.items():
        if len(p.image_ids) == max_len:
            print(f"Point ID: {p_id}")
            print(f"XYZ: {p.xyz}")
            print(f"Observed by {len(p.image_ids)} images. First 10 image IDs:")
            print(f"{p.image_ids[:10]} ...")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Point this to the folder containing cameras.bin, images.bin, points3D.bin
    parser.add_argument('--path', type=Path, required=True, help="Path to the sparse model folder")
    args = parser.parse_args()
    
    inspect_tracks(args.path)