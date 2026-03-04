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
        r_lin  = info.get('r_tracking_lin_vel', 0.0)
        r_ang  = info.get('r_tracking_ang_vel', 0.0)
        p_swing = info.get('p_swing_phase',     0.0)
        p_fh   = info.get('p_footswing_height', 0.0)
        p_vz   = info.get('p_lin_vel_z',        0.0)
        p_rate = info.get('p_action_rate',      0.0)
        p_slip = info.get('p_feet_slip',      0.0)

        print(
            f"step={step:04d} R={reward:7.4f} "
            f"r_lin={r_lin:.3f} r_ang={r_ang:.3f} "
            f"p_swing={p_swing:.3f} p_fh={p_fh:.4f} "
            f"p_vz={p_vz:.4f} p_slip={p_slip:.4f} p_rate={p_rate:.4f}"
        )

        if terminated:
            print(f"  ✗ 跌倒终止 at step={step}")
        if terminated or truncated:
            obs, info = env.reset()


if __name__ == '__main__':
    main()
