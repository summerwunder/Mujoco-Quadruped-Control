import argparse
import os
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from quadruped_ctrl.quadruped_env import QuadrupedEnv


class RLActionWrapper(gym.Wrapper):
    def step(self, action):
        tau = self.env.map_rl_action_to_torque(action)
        return self.env.step(tau)


def make_env(seed: int, command: np.ndarray):
    def _init():
        env = QuadrupedEnv(
            robot_config='robot/go1.yaml',
            model_path='quadruped_ctrl/assets/robot/go1/scene.xml',
            sim_config_path='sim_config.yaml',
            rl_config_path='rl_config.yaml',
            ref_base_lin_vel=command[:3],
            ref_base_ang_vel=np.array([0.0, 0.0, command[2]], dtype=np.float32),
        )
        env = RLActionWrapper(env)
        env.reset(seed=seed)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-timesteps', type=int, default=10000000)
    parser.add_argument('--n-envs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-dir', type=str, default='runs/ppo')
    parser.add_argument('--model-dir', type=str, default='runs/ppo/models')
    parser.add_argument('--command-vx', type=float, default=0.5)
    parser.add_argument('--command-vy', type=float, default=0.0)
    parser.add_argument('--command-yaw', type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    command = np.array([args.command_vx, args.command_vy, args.command_yaw], dtype=np.float32)

    env_fns = [make_env(args.seed + i, command) for i in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, filename=os.path.join(args.log_dir, 'monitor.csv'))

    eval_env = DummyVecEnv([make_env(args.seed + 1000, command)])
    eval_env = VecMonitor(eval_env, filename=os.path.join(args.log_dir, 'eval_monitor.csv'))

    checkpoint_cb = CheckpointCallback(
        save_freq=100000,
        save_path=args.model_dir,
        name_prefix='ppo_quadruped',
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.model_dir,
        log_path=args.log_dir,
        eval_freq=10000,
        deterministic=True,
    )

    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=args.log_dir,
        verbose=1,
        seed=args.seed,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, eval_cb],
    )

    model.save(os.path.join(args.model_dir, 'ppo_quadruped_final'))
    vec_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
