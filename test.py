import os
import imageio
import numpy as np

from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import ReachTarget

# -----------------------------
# 1. 配置 observation，打开一个相机
# -----------------------------
obs_config = ObservationConfig()
obs_config.set_all(False)

# 打开 front 相机 RGB
obs_config.front_camera.set_all(True)
obs_config.front_camera.image_size = [256, 256]

# 如果你想换 wrist / left_shoulder / overhead，也可以改
# obs_config.wrist_camera.set_all(True)
# obs_config.wrist_camera.image_size = [256, 256]

# -----------------------------
# 2. 定义 action mode
# -----------------------------
action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(),
    gripper_action_mode=Discrete()
)

# -----------------------------
# 3. 创建环境
# -----------------------------
env = Environment(
    action_mode=action_mode,
    obs_config=obs_config,
    headless=True
)

env.launch()
print("RLBench launch OK")

# -----------------------------
# 4. 选择任务
# -----------------------------
task = env.get_task(ReachTarget)
descriptions, obs = task.reset()
print("Task descriptions:", descriptions)

frames = []

# 先存一张初始图
if obs.front_rgb is not None:
    frames.append(obs.front_rgb)

# -----------------------------
# 5. 随机 rollout
# -----------------------------
# 对于 JointVelocity + Discrete gripper，
# action 通常是 [7维关节速度 + 1维夹爪开合]
# 这里我们动态根据 env.action_shape 来构造
action_dim = env.action_shape[0] if isinstance(env.action_shape, tuple) else env.action_shape
print("Action dim:", action_dim)

num_steps = 50

for step in range(num_steps):
    # 前 7 维：随机关节速度，范围不要太大
    arm_action = np.random.uniform(-0.1, 0.1, size=action_dim - 1)

    # 最后一维：夹爪动作，0 或 1
    gripper_action = np.random.choice([0.0, 1.0], size=1)

    action = np.concatenate([arm_action, gripper_action]).astype(np.float32)

    try:
        obs, reward, terminate = task.step(action)
    except Exception as e:
        print(f"Step {step} failed: {e}")
        break

    if obs.front_rgb is not None:
        frames.append(obs.front_rgb)

    print(f"step={step}, reward={reward}, terminate={terminate}")

    if terminate:
        print("Episode terminated early.")
        break

# -----------------------------
# 6. 保存 GIF
# -----------------------------
save_path = "random_reachtarget.gif"
imageio.mimsave(save_path, frames, fps=10)
print(f"GIF saved to: {os.path.abspath(save_path)}")

env.shutdown()
print("Environment shutdown.")