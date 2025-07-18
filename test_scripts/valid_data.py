# precheck_videos.py

import os
import argparse
import decord
from tqdm import tqdm

def check_videos(directory, output_file=None):
    """
    扫描一个目录下的所有视频文件，并检查它们是否可以被decord正常打开和读取。

    参数:
        directory (str): 需要检查的视频文件所在的目录路径。
        output_file (str, optional): 用于保存损坏文件ID列表的输出文件名。默认为 None。
    """
    print(f"\n开始扫描目录: {directory}")

    # 检查目录是否存在
    if not os.path.isdir(directory):
        print(f"错误：目录 '{directory}' 不存在或不是一个有效的目录。")
        return

    # 找到所有常见格式的视频文件
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    try:
        video_files = [f for f in os.listdir(directory) if f.lower().endswith(video_extensions)]
    except Exception as e:
        print(f"错误：无法读取目录 '{directory}'。权限问题？错误详情: {e}")
        return

    if not video_files:
        print("在该目录中没有找到任何视频文件。")
        return

    good_files_count = 0
    bad_files = []

    # 使用tqdm显示进度条，方便观察
    for filename in tqdm(video_files, desc="正在检查文件"):
        filepath = os.path.join(directory, filename)
        try:
            # 核心检查：尝试用decord打开视频
            vr = decord.VideoReader(filepath, ctx=decord.cpu(0))
            
            # 更严格的检查：尝试读取视频的长度和第一帧
            _ = len(vr)
            _ = vr[0]
            
            good_files_count += 1
        except Exception as e:
            # 如果任何步骤失败，就认为是损坏文件
            bad_files.append({'file': filename, 'error': str(e).strip().replace('\n', ' ')})

    # --- 打印检查结果报告 ---
    print("\n" + "="*20 + " 检查结果汇总 " + "="*20)
    print(f"总共检查文件数: {len(video_files)}")
    print(f"✅ 正常文件数: {good_files_count}")
    print(f"❌ 损坏/异常文件数: {len(bad_files)}")
    print("="*56)

    if bad_files:
        print("\n--- 损坏/异常文件列表 ---")
        for item in bad_files:
            print(f"文件: {item['file']:<40} | 错误: {item['error']}")
        
        if output_file:
            print(f"\n正在将损坏文件的ID列表写入到: {output_file}")
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for item in bad_files:
                        # 只写入文件名（不含后缀），方便后续处理
                        video_id = os.path.splitext(item['file'])[0]
                        f.write(f"{video_id}\n")
                print("写入完成。")
            except Exception as e:
                print(f"错误：无法写入文件 {output_file}。详情: {e}")

if __name__ == "__main__":
    # 使用argparse创建命令行接口
    parser = argparse.ArgumentParser(
        description="一个用于批量检查视频文件是否损坏的脚本。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("directory", type=str, help="需要检查的视频文件所在的目录路径。\n例如: ../data/train")
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="bad_files.txt", 
        help="用于保存损坏文件ID列表的输出文件名。\n(默认: bad_files.txt)。如果不想保存，请设置为 'None'。"
    )
    
    args = parser.parse_args()
    
    # 如果用户输入'None'或'none'，则不保存文件
    output_filename = args.output if args.output.lower() != 'none' else None
    
    check_videos(args.directory, output_filename)