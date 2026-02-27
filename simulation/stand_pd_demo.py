from quadruped_ctrl.controllers.controller_factory import ControllerFactory
from quadruped_ctrl.quadruped_env import QuadrupedEnv
import mujoco
import mujoco.viewer
import time
import numpy as np

if __name__ == '__main__':
    env = QuadrupedEnv(robot_config='robot/go1.yaml',
                       model_path='quadruped_ctrl/assets/robot/go1/scene_terrain.xml',
                       sim_config_path='sim_config.yaml',)
    obs, _ = env.reset()
    pd_controller = ControllerFactory.create_controller("pd", env)
    print("启动 MuJoCo 可视化查看器...")
    
    # 使用被动查看器（passive viewer）
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step = 0
        while viewer.is_running():
            step_start = time.time()
            state = env.get_state()
            # 随机action控制
            # action = env.action_space.sample()
            pos = np.array([0.0, 0.7, -1.4] * 4)
            # print(state.base.pos[2])  # 0.32
            action = pd_controller.get_action(state, pos)
            obs, _, terminated, truncated, info = env.step(action)
            
            # 重置逻辑
            # if terminated or truncated:
            #     obs, _ = env.reset()
            
            # 同步可视化
            viewer.sync()
            
            # 每50步打印一次
            if step % 50 == 0:
                print(f"步数: {step}, 仿真时间: {env.data.time:.2f}s")
            
            step += 1
            
            # 控制帧率（可选）
            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
