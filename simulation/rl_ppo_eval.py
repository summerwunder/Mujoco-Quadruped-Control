import argparse
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from quadruped_ctrl.quadruped_env import QuadrupedEnv


class RLActionWrapper(gym.Wrapper):
    def step(self, action):
        tau = self.env.map_rl_action_to_torque(action)
        return self.env.step(tau)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--command-vx', type=float, default=0.4)
    parser.add_argument('--command-vy', type=float, default=0.0)
    parser.add_argument('--command-yaw', type=float, default=0.0)
    args = parser.parse_args()

    command = np.array([args.command_vx, args.command_vy, args.command_yaw], dtype=np.float32)

    env = QuadrupedEnv(
        robot_config='robot/go1.yaml',
        model_path='quadruped_ctrl/assets/robot/go1/scene_terrain.xml',
        sim_config_path='sim_config.yaml',
        rl_config_path='rl_config.yaml',
        ref_base_lin_vel=command[:3],
        ref_base_ang_vel=np.array([0.0, 0.0, command[2]], dtype=np.float32),
    )
    env = RLActionWrapper(env)

    model = PPO.load(args.model, env=env)
    obs, info = env.reset()

    for step in range(args.steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if step % 100 == 0:
            print(
                f"step={step:04d} reward={reward:.4f} "
                f"r_vel={info.get('r_vel', 0.0):.4f} "
                f"p_tilt={info.get('p_tilt', 0.0):.4f} "
                f"p_energy={info.get('p_energy', 0.0):.4f}"
            )
        if terminated or truncated:
            obs, info = env.reset()
        env.render()

if __name__ == '__main__':
    main()
