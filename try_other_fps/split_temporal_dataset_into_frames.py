import os
import subprocess
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

def extract_frames_from_video(video_path, output_folder, target_fps=10):
    try:
        os.makedirs(output_folder, exist_ok=True)
 
        cmd = [
            'ffmpeg',
            '-i', video_path,           
            '-vf', f'fps={target_fps}', 
            '-frame_pts', '0',          
            '-vsync', '0',              
            '-f', 'image2',             
            '-q:v', '2',                
            os.path.join(output_folder, '%05d.jpg')  
        ]
        
        with open(os.devnull, 'w') as devnull:
            subprocess.run(cmd, check=True, stderr=devnull)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error for {video_path}: {str(e)}")
        return False
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return False

def process_video(args):
    try:
        filename, input_folder, output_base = args
        if Path(filename).suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}:
            video_id = Path(filename).stem
            video_path = os.path.join(input_folder, filename)
            output_folder = os.path.join(output_base, video_id)
            return extract_frames_from_video(video_path, output_folder, target_fps=10)  # 在这里修改fps
        return False
    except Exception as e:
        print(f"Error in process_video for {filename}: {str(e)}")
        return False

def extract_all_videos():
    input_folder = "/tmp/didemo/video/train"
    output_base = "/tmp/didemo/video/frames"
    
    os.makedirs(output_base, exist_ok=True)
    
    video_files = [f for f in sorted(os.listdir(input_folder)) 
                  if Path(f).suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}]
    
    num_processes = 16
    print(f"Starting processing with {num_processes} processes...")
    
    args_list = [(f, input_folder, output_base) for f in video_files]
    
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_video, args_list),
            total=len(args_list),
            desc="Processing videos"
        ))
    
    successful = sum(1 for r in results if r)
    failed = len(video_files) - successful
    print(f"Processing completed. Successful: {successful}, Failed: {failed}")

if __name__ == "__main__":
    extract_all_videos()