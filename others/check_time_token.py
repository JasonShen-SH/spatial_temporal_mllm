import json
import re
import pdb
import numpy as np

########################################################
# 检查数量
# file_path = '/home/rqshen/spatial_temporal_mllm/special_jsons/temporal/Charades/time_token.json'
# file_path = '/home/rqshen/VTG-IT/VTG-IT/moment_retrieval/VTG-IT-MR/Charades/time_token.json'

# with open(file_path, 'r') as f:
#     data = json.load(f)
# unique_videos = {item['video'] for item in data}
# print(f"JSON文件总条目数: {len(data)}")
# print(f"不同video的数量: {len(unique_videos)}")
########################################################

########################################################
# # 检查时间顺序: start 是否在 end 前面
# def parse_time(time_str):
#     number_map = {
#         '<TIME_ZERO>': '0',
#         '<TIME_ONE>': '1',
#         '<TIME_TWO>': '2',
#         '<TIME_THREE>': '3',
#         '<TIME_FOUR>': '4',
#         '<TIME_FIVE>': '5',
#         '<TIME_SIX>': '6',
#         '<TIME_SEVEN>': '7',
#         '<TIME_EIGHT>': '8',
#         '<TIME_NINE>': '9',
#         '<TIME_DOT>': '.'
#     }
    
#     for token, digit in number_map.items():
#         time_str = time_str.replace(token, digit)
    
#     return float(time_str)

# def check_time_order(data):
#     errors = []
#     for idx, item in enumerate(data):
#         try:
#             for qa in item['QA']:
#                 answer = qa['a']
#                 time_part = answer.split('seconds,')[0].strip()
#                 start_str, end_str = time_part.split(' - ')
                
#                 start_time = parse_time(start_str)
#                 end_time = parse_time(end_str)
                
#                 if start_time >= end_time:
#                     errors.append({
#                         'index': idx,
#                         'video': item['video'],
#                         'start_time': start_time,
#                         'end_time': end_time,
#                         'original_answer': answer
#                     })
#             # print(f"成功处理{idx}")
#         except Exception as e:
#             print(f"处理{idx}时出错: {e}")
#             errors.append({
#                 'index': idx,
#                 'video': item['video'],
#                 'error': str(e)
#             })
    
#     return errors

# file_path = '/home/rqshen/spatial_temporal_mllm/special_jsons/temporal/Charades/time_token.json'
# with open(file_path, 'r') as f:
#     data = json.load(f)

# errors = check_time_order(data)

# print(f"总条目数: {len(data)}")
# if not errors:
#     print("所有时间戳都符合要求（开始时间严格小于结束时间）")
# else:
#     print(f"发现 {len(errors)} 个时间顺序错误:")
#     for error in errors:
#         print(f"\n索引: {error['index']}")
#         print(f"视频: {error['video']}")
#         print(f"开始时间: {error['start_time']}")
#         print(f"结束时间: {error['end_time']}")
#         print(f"原始答案: {error['original_answer']}")
########################################################


########################################################
# 计算每个video平均帧数

def parse_time(time_str):
    number_map = {
        '<TIME_ZERO>': '0',
        '<TIME_ONE>': '1',
        '<TIME_TWO>': '2',
        '<TIME_THREE>': '3',
        '<TIME_FOUR>': '4',
        '<TIME_FIVE>': '5',
        '<TIME_SIX>': '6',
        '<TIME_SEVEN>': '7',
        '<TIME_EIGHT>': '8',
        '<TIME_NINE>': '9',
        '<TIME_DOT>': '.'
    }
    
    for token, digit in number_map.items():
        time_str = time_str.replace(token, digit)
    
    return float(time_str)

def analyze_time_intervals(data, fps=10):
    intervals = []
    total_count = 0
    less_than_40_count = 0
    
    for idx, item in enumerate(data):
        try:
            for qa in item['QA']:
                answer = qa['a']
                time_part = answer.split('seconds,')[0].strip()
                start_str, end_str = time_part.split(' - ')
                
                start_time = parse_time(start_str)
                end_time = parse_time(end_str)
                
                if start_time >= end_time:
                    continue
                    
                frame_count = (end_time - start_time) * fps
                total_count += 1
                if frame_count < 30:
                    less_than_40_count += 1
                    
                intervals.append({
                    'index': idx,
                    'video': item['video'],
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'frame_count': frame_count
                })
        except Exception as e:
            print(f"处理{idx}时出错: {e}")
    
    frame_counts = [item['frame_count'] for item in intervals]
    durations = [item['duration'] for item in intervals]
    
    stats = {
        'total_samples': len(intervals),
        'mean_frames': np.mean(frame_counts),
        'median_frames': np.median(frame_counts),
        'min_frames': np.min(frame_counts),
        'max_frames': np.max(frame_counts),
        'mean_duration': np.mean(durations),
        'median_duration': np.median(durations),
        'min_duration': np.min(durations),
        'max_duration': np.max(durations),
        'less_than_40_ratio': less_than_40_count / total_count if total_count > 0 else 0
    }
    
    return intervals, stats

file_path = '/home/rqshen/spatial_temporal_mllm/special_jsons/temporal/Charades/time_token.json'
with open(file_path, 'r') as f:
    data = json.load(f)

intervals, stats = analyze_time_intervals(data, fps=10)

print("统计信息:")
print(f"总样本数: {stats['total_samples']}")
print(f"帧数<30的比例: {stats['less_than_40_ratio']:.4f}")
print(f"\n帧数统计:")
print(f"平均帧数: {stats['mean_frames']:.2f}")
print(f"中位数帧数: {stats['median_frames']:.2f}")
print(f"最小帧数: {stats['min_frames']:.2f}")
print(f"最大帧数: {stats['max_frames']:.2f}")

print(f"\n时长统计(秒):")
print(f"平均时长: {stats['mean_duration']:.2f}")
print(f"中位数时长: {stats['median_duration']:.2f}")
print(f"最小时长: {stats['min_duration']:.2f}")
print(f"最大时长: {stats['max_duration']:.2f}")

print("\n最短的5个片段:")
sorted_by_duration = sorted(intervals, key=lambda x: x['duration'])
for item in sorted_by_duration[:5]:
    print(f"视频: {item['video']}, 时长: {item['duration']:.2f}秒, 帧数: {item['frame_count']:.2f}")

print("\n最长的5个片段:")
for item in sorted_by_duration[-5:]:
    print(f"视频: {item['video']}, 时长: {item['duration']:.2f}秒, 帧数: {item['frame_count']:.2f}")