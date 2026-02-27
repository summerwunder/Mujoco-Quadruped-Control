import numpy as np
import gymnasium as gym

from quadruped_ctrl.quadruped_env import QuadrupedEnv


class RLActionWrapper(gym.Wrapper):
    def step(self, action):
        tau = self.env.map_rl_action_to_torque(action)
        return self.env.step(tau)


def main():
    env = QuadrupedEnv(
        robot_config='robot/go1.yaml',
        model_path='quadruped_ctrl/assets/robot/go1/scene_terrain.xml',
        sim_config_path='sim_config.yaml',
        rl_config_path='rl_config.yaml',
    )
    env = RLActionWrapper(env)
    obs, info = env.reset()
    print('obs shape:', obs.shape)

    for step in range(2000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step={step:02d} reward={reward:.4f} "
            f"r_vel={info.get('r_vel', 0.0):.4f} "
            f"p_tilt={info.get('p_tilt', 0.0):.4f}"
        )
        if terminated or truncated:
            obs, info = env.reset()
        env.render()


if __name__ == '__main__':
    main()
