import os
import glob
import csv

# Paths
kitti_folder = "KITTI_DATA/Canton"  # folder with KITTI txt files
mot_output_file = "MOT_DATA/gt.txt"

# Map KITTI classes to integer IDs
class_mapping = {
    "person": 1,
}

mot_rows = []

# Sort files to ensure frame order
kitti_files = sorted(glob.glob(os.path.join(kitti_folder, "*.txt")))

for frame_idx, file_path in enumerate(kitti_files, start=1):
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:  # need at least up to bbox bottom
            continue
        
        obj_label = parts[0]
        track_id = int(parts[1])
        bbox_left = float(parts[5])
        bbox_top = float(parts[6])
        bbox_right = float(parts[7])
        bbox_bottom = float(parts[8])
        
        x = int(round(bbox_left))
        y = int(round(bbox_top))
        w = int(round(bbox_right - bbox_left))
        h = int(round(bbox_bottom - bbox_top))

        included_for_eval = 1
        class_id = class_mapping.get(obj_label, 0)  # default 0 if class unknown
        
        # Use visibility if present, otherwise default 1
        # visibility_ratio = float(parts[16]) if len(parts) > 16 else 1.0
        visibility_ratio = 1.0
        
        mot_row = [frame_idx, track_id, x, y, w, h, included_for_eval, class_id, visibility_ratio]
        mot_rows.append(mot_row)

# Write MOT CSV
with open(mot_output_file, "w", newline="") as f:
    writer = csv.writer(f)
    for row in mot_rows:
        writer.writerow(row)

print(f"Conversion completed! Saved to {mot_output_file}")
