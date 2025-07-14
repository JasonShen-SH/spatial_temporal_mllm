import subprocess
import os

def download_videos(video_file):
    # 打开videos.txt文件
    with open(video_file, 'r') as file:
        # 逐行读取视频链接
        for line in file:
            # 去除每行的空格和换行符
            video_id = line.strip()
            if not video_id:
                continue
                
            # 检查视频是否已存在（检查常见视频格式）
            output_dir = "E:/other_videos/"
            existing_files = [
                f"{output_dir}{video_id}.mp4",
                f"{output_dir}{video_id}.webm",
                f"{output_dir}{video_id}.mkv"
            ]
            
            # 如果任一格式的文件已存在，则跳过
            if any(os.path.exists(f) for f in existing_files):
                print(f"视频 {video_id} 已存在，跳过下载")
                continue
                
            # 生成完整的 YouTube URL
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            print(f"正在下载: {video_url}")
            
            # 执行 yt-dlp 下载命令
            output_path = f"{output_dir}{video_id}.%(ext)s"
            subprocess.run([
                "C:/Users/13301/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/Scripts/yt-dlp.exe", 
                video_url, "-f", "134", "-o", output_path
            ])
            print(f"下载完成: {video_url}\n")

if __name__ == "__main__":
    video_file = "E:/video_names.txt"
    download_videos(video_file)