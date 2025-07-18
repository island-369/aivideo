# test_decord_single_video.py
import decord
import sys

def test_video_file(filepath):
    """
    一个专门用来测试单个视频文件能否被decord完整读取的函数。
    """
    print("="*50)
    print(f"正在测试文件: {filepath}")
    
    try:
        # 1. 尝试打开视频文件
        print("步骤 1: 尝试用 decord.VideoReader 打开...")
        vr = decord.VideoReader(filepath, ctx=decord.cpu(0))
        print("✅ 成功打开！")

        # 2. 尝试获取总帧数
        print("步骤 2: 尝试获取总帧数 (len)...")
        num_frames = len(vr)
        print(f"✅ 成功获取！总帧数: {num_frames}")

        # 3. 尝试获取视频尺寸 (这是在您脚本中出错的地方)
        print("步骤 3: 尝试获取视频尺寸 (.shape)...")
        # 使用 .shape 属性，它返回 (帧数, 高, 宽, 通道数)
        height, width, _ = vr.shape[1:]
        print(f"✅ 成功获取！尺寸: {width}x{height}")
        
        # 4. 尝试读取第一帧
        print("步骤 4: 尝试读取第一帧...")
        first_frame = vr[0]
        print(f"✅ 成功读取！第一帧形状: {first_frame.shape}")
        
        print("\n--- 结论: ✅ 该视频文件可以被 decord 完整、正常地读取。 ---")

    except Exception as e:
        print("\n--- 结论: ❌ 测试失败！---")
        print(f"在上述某个步骤中发生了错误，错误详情如下:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
    
    print("="*50)


if __name__ == "__main__":
    # 从命令行接收第一个参数作为文件路径
    if len(sys.argv) < 2:
        print("用法: python test_decord_single_video.py <视频文件路径>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    test_video_file(video_path)