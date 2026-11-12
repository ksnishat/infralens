import os
import cv2
import numpy as np
import albumentations as A

# --- CONFIGURATION ---
DATA_DIR = "/app/Rust-Detection-1"
SETS = ["train", "valid"] 

# Define transformations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.SafeRotate(limit=15, p=0.5) 
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def clip_bbox(box):
    """
    Sanitize the box: Ensure all values are strictly between 0.0 and 1.0
    YOLO format is: [x_center, y_center, width, height]
    """
    return [min(max(x, 0.0), 1.0) for x in box]

def augment_dataset():
    print("Starting Data Augmentation (With Sanitization)...")
    
    for split in SETS:
        img_dir = os.path.join(DATA_DIR, split, "images")
        lbl_dir = os.path.join(DATA_DIR, split, "labels")
        
        images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        print(f"Processing {split}: Found {len(images)} original images.")
        
        count = 0
        for img_name in images:
            base_name = os.path.splitext(img_name)[0]
            lbl_name = base_name + ".txt"
            
            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(lbl_dir, lbl_name)
            
            # Read Image
            image = cv2.imread(img_path)
            if image is None: continue
            
            # Read and CLEAN Label
            bboxes = []
            class_labels = []
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        cls = int(parts[0])
                        raw_bbox = [float(x) for x in parts[1:]]
                        
                        # --- THE FIX: SANITIZE BEFORE ADDING ---
                        clean_box = clip_bbox(raw_bbox)
                        
                        # Extra Safety: Ensure width/height > 0
                        if clean_box[2] > 0 and clean_box[3] > 0:
                            bboxes.append(clean_box)
                            class_labels.append(cls)
            
            # Skip images with broken/empty labels
            if not bboxes:
                continue

            # --- GENERATE 2 AUGMENTED VERSIONS ---
            for i in range(2):
                try:
                    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_img = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    
                    if len(aug_bboxes) == 0: continue

                    # Save New Image
                    new_name = f"{base_name}_aug_{i}"
                    cv2.imwrite(os.path.join(img_dir, new_name + ".jpg"), aug_img)
                    
                    # Save New Label
                    with open(os.path.join(lbl_dir, new_name + ".txt"), 'w') as f:
                        for cls, box in zip(class_labels, aug_bboxes):
                            # Final clip to be safe
                            box = clip_bbox(box)
                            f.write(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
                    
                    count += 1
                except Exception as e:
                    # If a specific box is mathematically impossible, skip it but print why
                    pass 

        print(f"Finished {split}: Created {count} new images.")

if __name__ == "__main__":
    augment_dataset()