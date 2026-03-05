# Quadruped Controller

基于 MuJoCo 的四足机器人控制框架，支持 NMPC 步态控制与强化学习（PPO）训练。

## Demo

| NMPC 平地行走 | NMPC坡度行走 | NMPC 地形行走 |
|:---:|:---:|:---:|
| ![walk_mpc](docs/walk_mpc.gif) | ![walk_slope_mpc](docs/walk_slope_mpc.gif) | ![walk_terrain_mpc](docs/walk_terrain_mpc.gif) |

## 功能特性

- NMPC 梯度优化控制器（支持平地 / 斜坡 / 崎岖地形）
- 强化学习控制器（PPO，基于 Stable-Baselines3）
- 摆动轨迹生成
- 全身控制接口
- 可视化调试工具

## 安装

```bash
pip install -e .
```

## 运行示例

```bash
# NMPC 控制
python simulation/trot_ground_mpc_demo.py
python simulation/trot_terrain_mpc_demo.py

# RL 训练
python simulation/rl_ppo_train.py

# RL 评估
python simulation/rl_ppo_eval.py
```

## 项目结构

```
quadruped_ctrl/
├── assets/          # 机器人模型文件
├── config/          # 配置文件
├── controllers/     # 控制器实现（NMPC / PD）
├── interface/       # 参考接口
├── planning/        # 步态规划
└── utils/           # 工具函数
simulation/
├── trot_ground_mpc_demo.py   # NMPC 平地演示
├── trot_terrain_mpc_demo.py  # NMPC 地形演示
├── rl_ppo_train.py           # PPO 训练
└── rl_ppo_eval.py            # PPO 评估
```

## 许可证

Apache2.0 License
