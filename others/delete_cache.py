import os
import shutil

def remove_pycache_dirs(root_dir):
    """
    递归遍历 root_dir 目录，并删除所有 __pycache__ 目录
    """
    for current_path, dirs, _ in os.walk(root_dir):
        for d in dirs:
            if d == "__pycache__":
                full_dir_path = os.path.join(current_path, d)
                print(f"Deleting {full_dir_path}")
                shutil.rmtree(full_dir_path, ignore_errors=True)

if __name__ == "__main__":
    # 列出你需要清除 __pycache__ 目录的所有根目录
    # if os.path.exists("/mnt/data_hdd/rqshen/Sa2VA/debug4"):
    #     shutil.rmtree("/mnt/data_hdd/rqshen/Sa2VA/debug4")
    #     print("debug4 removed")
        
    # if os.path.exists("/mnt/data_hdd/rqshen/Sa2VA/debug5"):
    #     shutil.rmtree("/mnt/data_hdd/rqshen/Sa2VA/debug5")
    #     print("debug5 removed")
        
    root_dirs = ["third_parts", "projects"]

    for root in root_dirs:
        if os.path.exists(root):
            print(f"Processing {root} ...")
            remove_pycache_dirs(root)
        else:
            print(f"Directory not found: {root}")