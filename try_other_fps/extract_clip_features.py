import torch
from PIL import Image
import open_clip
import os
from tqdm import tqdm
import glob
import json
import pdb

BATCH_SIZE = 32

save_dir = "/home/rqshen/spatial_mllm/clip_features/didemo"
os.makedirs(save_dir, exist_ok=True)

model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
model = model.cuda()
model.eval()

frames_path = "/tmp/didemo/video/frames"
video_ids = sorted(os.listdir(frames_path))

for video_id in tqdm(video_ids):
    try:
        video_folder = os.path.join(frames_path, video_id)
        if not os.path.exists(video_folder):
            print(f"Skip {video_id}: folder not found")
            continue
            
        frame_paths = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
        
        if not frame_paths:
            print(f"Skip {video_id}: no frames found")
            continue
            
        video_features = []
        batches = [frame_paths[i:i + BATCH_SIZE] for i in range(0, len(frame_paths), BATCH_SIZE)]
        
        with torch.no_grad(), torch.autocast("cuda"):
            for batch_paths in batches:
                batch_images = torch.stack([
                    preprocess(Image.open(path)) for path in batch_paths
                ]).cuda()
                
                features = model.encode_image(batch_images)
                features = features / features.norm(dim=-1, keepdim=True)
                video_features.append(features.cpu())
        
        video_features = torch.cat(video_features, dim=0)
        
        save_path = os.path.join(save_dir, f"{video_id}.pt")
        torch.save(video_features, save_path)
        
    except Exception as e:
        print(f"Error processing {video_id}: {str(e)}")
        continue