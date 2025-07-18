deepspeed --num_gpus=1 train_deepspeed.py --deepspeed_config ds_config.json


pip install opencv-python --default-timeout=600 --resume-retries=10 -i https://pypi.tuna.tsinghua.edu.cn/simple


echo $STY

python zero_to_fp32.py .  model

deepspeed --num_gpus=4 test_distribute.py 