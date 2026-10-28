# 1. 推理
## 1.1 激活客户端终端环境
```
cd C:\Users\grg\lerobot-043
conda activate lerobot-043

lerobot-find-cameras opencv
```
## 1.2 整炒
```
python -m lerobot.async_inference.robot_client --server_address=10.61.5.8:8099 --robot.type=bi_so_follower --robot.left_arm_config.port=COM24 --robot.right_arm_config.port=COM28 --robot.id=bimanual_follower --robot.left_arm_config.cameras="{fixed: {type: intelrealsense, serial_number_or_name: '260322278758', width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" --robot.right_arm_config.cameras="{wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: MJPG}}" --task="Pick up the banknotes from the desk, straighten them neatly and hand them over to the customer." --policy_type=pi05 --pretrained_name_or_path=/nfs/lerobot/s101/models/straighten_cash_0928_to_0930_pi05/checkpoints/last/pretrained_model --policy_device=cuda:0 --actions_per_chunk=50 --debug_visualize_queue_size=True
```

# 2. 数据采集
## 2.1 整钞
```
lerobot-human-inloop-record --robot.type=bi_so_follower --robot.left_arm_config.port=COM24 --robot.right_arm_config.port=COM28 --robot.id=bimanual_follower  --robot.left_arm_config.cameras="{fixed: {type: intelrealsense, serial_number_or_name: '260322278758', width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: MJPG}}" --robot.right_arm_config.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}}" --teleop.type=bi_so_leader --teleop.left_arm_config.port=COM22 --teleop.right_arm_config.port=COM26 --teleop.id=bimanual_leader --dataset.push_to_hub=false --display_data=true --dataset.repo_id=so101/straighten_cash_1010_1_nopolicy --dataset.num_episodes=50 --dataset.single_task="Pick up the banknotes from the desk, straighten them neatly and hand them over to the customer." --dataset.episode_time_s=120
```