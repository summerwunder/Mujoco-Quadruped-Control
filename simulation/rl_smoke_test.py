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
        r_vel = info.get('r_vel', 0.0)
        r_swing = info.get('r_swing', 0.0)
        p_contact = info.get('p_contact', 0.0)
        p_tilt = info.get('p_tilt', 0.0)
        p_action_delta = info.get('p_action_delta', 0.0)
        print(
            f"step={step:04d} R={reward:7.4f} r_vel={r_vel:.4f} r_swing={r_swing:.4f} "
            f"p_contact={p_contact:.4f} p_tilt={p_tilt:.6f} p_delta={p_action_delta:.4f}"
        )
        if terminated or truncated:
            obs, info = env.reset()
        env.render()


if __name__ == '__main__':
    main()
