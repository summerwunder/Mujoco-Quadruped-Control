# RL PPO (SB3)

## Train

```bash
python simulation/rl_ppo_train.py \
  --total-timesteps 200000 \
  --n-envs 4 \
  --command-vx 0.4 \
  --command-vy 0.0 \
  --command-yaw 0.0 
```

## Evaluate

```bash
python simulation/rl_ppo_eval.py --model runs/ppo/models/ppo_quadruped_final
```

## Smoke Test

```bash
python simulation/rl_smoke_test.py
```
