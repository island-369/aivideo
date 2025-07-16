import os
import cv2

def is_video_file(filename):
    exts = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpg', '.mpeg')
    return filename.lower().endswith(exts)

def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def main(video_dir):
    video_info = []
    for fname in sorted(os.listdir(video_dir)):
        if is_video_file(fname):
            full_path = os.path.join(video_dir, fname)
            cnt = get_video_frame_count(full_path)
            if cnt is None:
                print(f"[警告] 无法打开视频: {fname}")
                continue
            video_info.append((fname, cnt))
            print(f"{fname}: {cnt}帧")
    if not video_info:
        print("目录下没有找到任何视频。")
        return

    # 汇总统计
    counts = [c for _,c in video_info]
    print("\n--- 视频帧数统计汇总 ---")
    print(f"视频数: {len(counts)}")
    print(f"总帧数: {sum(counts)}")
    print(f"平均帧数: {sum(counts)/len(counts):.2f}")
    print(f"最小帧数: {min(counts)}")
    print(f"最大帧数: {max(counts)}")
    print(f"中位数: {sorted(counts)[len(counts)//2]}")
    # 可选：把统计结果保存到csv
    import pandas as pd
    pd.DataFrame(video_info, columns=["filename", "frame_count"]).to_csv("video_frame_count.csv", index=False)
    print("已保存video_frame_count.csv")

if __name__ == '__main__':
    import sys
    if len(sys.argv)!=2:
        print(f"用法：python {sys.argv[0]} <视频目录>")
        exit(1)
    main(sys.argv[1])