import os
import shutil
import json
from pathlib import Path
import numpy as np

def convert_to_yolo_dataset(
    root_dir,
    output_dir='yolo_dataset'
):
    # load anotation
    annotations_path = os.path.join(root_dir, 'annotations.json')
    with open(annotations_path) as f:
        anns = json.load(f)
    
    # yolo folder firectory
    output_dir = Path(output_dir)
    (output_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
    (output_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
    (output_dir / 'test' / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'test' / 'labels').mkdir(parents=True, exist_ok=True)
    
    # check images id
    image_info = {img['id']: img for img in anns['images']}
    
    # get splits from original dataset
    splits = anns['splits']['chessred2k']
    train_ids = set(splits['train']['image_ids'])
    val_ids = set(splits['val']['image_ids'])
    test_ids = set(splits['test']['image_ids'])
    
    # process each image
    for img_id, img_data in image_info.items():
        # determine split
        if img_id in train_ids:
            split = 'train'
        elif img_id in val_ids:
            split = 'val'
        elif img_id in test_ids:
            split = 'test'
        else:
            continue
        
        # origin and destination paths
        src_img_path = os.path.join(root_dir, img_data['path'])
        dst_img_dir = output_dir / split / 'images'
        dst_label_dir = output_dir / split / 'labels'
        
        # copy image to the appropriate directory
        shutil.copy(src_img_path, dst_img_dir)
        
        # generate label file name and path
        img_width = img_data['width']
        img_height = img_data['height']
        txt_filename = Path(img_data['path']).stem + '.txt'
        txt_path = dst_label_dir / txt_filename
        
        # annotations for the current image
        img_annotations = [a for a in anns['annotations']['pieces'] if a['image_id'] == img_id]
        
        with open(txt_path, 'w') as f:
            for ann in img_annotations:
                if 'bbox' not in ann or 'category_id' not in ann:
                    continue  # Invalid annotations
                
                yolo_class = ann['category_id']
                
                # normalizing bounding box coordinates
                x_min, y_min, width, height = ann['bbox']
                x_center = (x_min + width / 2) / img_width
                y_center = (y_min + height / 2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height
                
                f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    yaml_content = f"""path: {output_dir.absolute()}
train: train/images
val: val/images
test: test/images

names:
0: white-pawn
1: white-rook
2: white-knight
3: white-bishop
4: white-queen
5: white-king
6: black-pawn
7: black-rook
8: black-knight
9: black-bishop
10: black-queen
11: black-king
12: empty
"""

    with open(output_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)

convert_to_yolo_dataset('')