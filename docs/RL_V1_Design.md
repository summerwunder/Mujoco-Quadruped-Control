# 四足机器人 RL 训练系统 - V1 设计总结

## 📋 项目概述

构建了一个基于 MuJoCo 物理引擎 + Stable-Baselines3 PPO 的四足机器人（Go1）强化学习训练框架。

**目标**：训练机器人学会在指定速度命令下的自主行走。

**基础架构**：
- 物理模型：MuJoCo 3.4.0 + Go1 URDF 模型
- RL 算法：Stable-Baselines3 PPO（多进程并行训练）
- 配置系统：YAML + 模块化加载器

---

## 🔍 系统设计

### 1. 观测空间（Observation Space）

**维度**：48 维

**组成**：
```
[
  qpos(12),         # 关节位置（3 DOF × 4 腿）
  qvel(12),         # 关节速度（3 DOF × 4 腿）
  roll_pitch(2),    # 基座姿态（横滚、俯仰）
  lin_vel(3),       # 基座线速度 + 高斯噪声 N(0, 0.01)
  command(3),       # 目标速度命令 [v_x, v_y, yaw_rate]
  foot_pressure(4), # 足端接触力（标量，FL/FR/RL/RR）
  prev_action(12),  # 前一步的原始 RL 动作
]
```

**设计决策**：
- ✅ 包含命令以保证 Markov 性质
- ✅ 包含动作历史以让 policy 看到自己的抖动
- ✅ 不包含 yaw 角（欧拉角奇异）
- ✅ 基座速度在基座坐标系（本体感知更好）

---

### 2. 动作空间（Action Space）

**类型**：连续，Box(-1, 1)^12

**含义**：关节位置偏移（相对于默认站姿）

**站姿配置**：
```python
stand_pose = [0.0, 0.7, -1.4] × 4  # [hip, thigh, calf] × 4 legs
```

**映射流程**：
```
RL 动作 a ∈ [-1, 1]
    ↓ clip
a ∈ [-range, range]  (range=1.0)
    ↓ scale
offset = a × offset_scale  (offset_scale=0.5)
    ↓ smooth
smooth_offset = smooth(offset, prev_offset)
    ↓ add to stand pose
q_des = stand_pose + smooth_offset
    ↓ PD control
τ = Kp(q_des - q) + Kd(0 - v)
```

**平滑机制**：
- **类型**：Lowpass Filter（推荐）
- **参数**：
  - `alpha = 0.2`（混合系数）
  - `smooth_offset = (1-α) × prev_offset + α × target_offset`
  - 在 **offset 空间**（不是动作空间）做平滑

**可选**：Rate Limiting
- `max_delta = 0.1`
- 限制相邻步的 offset 变化

---

### 3. 奖励函数（Reward Function）

**公式**：
```
R = w_vel × r_vel 
  + w_alive × r_alive 
  - w_tilt × p_tilt 
  - w_energy × p_energy 
  - w_action_delta × p_action_delta
```

**各分量详解**：

#### a) 速度跟踪奖励 `r_vel`
```python
vel_err = v_cmd - v_act
r_vel = exp(-||vel_err||²/σ)
```
- **参数**：`vel_sigma = 0.25`
- **目的**：鼓励追踪目标速度
- **权重**：`w_vel = 2.0`

#### b) 存活奖励 `r_alive`
```python
r_alive = const = 1.0  per step
```
- **目的**：鼓励长期运行（防止早期跌倒）
- **权重**：`w_alive = 0.2`

#### c) 倾角惩罚 `p_tilt`
```python
p_tilt = roll² + pitch²
```
- **目的**：保持身体水平（稳定性）
- **权重**：`w_tilt = 0.5`

#### d) 能耗惩罚 `p_energy`
```python
p_energy = Σ|τ_i × v_i|  (对所有12个关节)
```
- **目的**：最小化能量消耗
- **权重**：`w_energy = 0.0001`（很小，主要作用是关节不动时削减）

#### e) 动作变化惩罚 `p_action_delta` ⭐
```python
p_action_delta = ||a_t - a_{t-1}||₂  (L2范数)
```
- **目的**：防止高频微震（reward hacking）
- **权重**：`w_action_delta = 1.0`
- **为什么重要**：狗子会发现"高频微震+平滑"是个漏洞

**完整权重配置**：
```yaml
reward:
  vel_sigma: 0.25
  weights:
    vel: 2.0
    alive: 0.2
    tilt: 0.5
    energy: 0.0001
    action_delta: 1.0
  alive: 1.0  # r_alive 的常数值
```

---

## 🎯 训练配置

### PPO 超参数
```python
PPO(
    policy='MlpPolicy',           # 2 层隐藏层（64 维）
    learning_rate=3e-4,
    n_steps=2048,                 # 单进程采集步数
    batch_size=256,
    gamma=0.99,                   # 折扣因子
    gae_lambda=0.95,              # GAE 参数
    clip_range=0.2,               # PPO 裁剪范围
    ent_coef=0.0,                 # 熵奖励（禁用）
    vf_coef=0.5,                  # 价值函数损失权重
    max_grad_norm=0.5,            # 梯度裁剪
    tensorboard_log='runs/ppo',
    verbose=1,
)
```

### 并行训练
- **环境数**：4 个（DummyVecEnv）
- **总步数**：200,000 timesteps
- **检查点**：每 100,000 步保存一次

### 目标命令
```python
command = [v_x=0.4, v_y=0.0, yaw_rate=0.0] m/s
```

---

## 📊 观察到的行为

### 问题：Reward Hacking - 高频微震滑行

**症状**：
- ❌ 四肢关节看起来僵硬不动
- ❌ 身体仍在平移/旋转
- ❌ 可能是高频微震（>100Hz）不可见

**原因分析**：
1. `p_action_delta` 权重初期太低 → 高频抖动便宜
2. 平滑机制只约束**相邻步**，未约束**频率**
3. `p_energy = |τ×v|` 在 `v≈0` 时无效
4. 没有直接惩罚关节速度/加速度

**已采取措施**：
- ✅ 将 `action_delta` 权重从 0.1 → 1.0（提高 10 倍）
- ✅ 监控 `p_action_delta` 指标
- ✅ 准备下个版本加关节速度惩罚

---

## 🔧 文件结构

```
quadruped/
├── quadruped_ctrl/
│   ├── quadruped_env.py          # 主环境类
│   ├── config/
│   │   ├── sim_config.yaml        # 物理模拟参数
│   │   └── rl_config.yaml         # RL 系统参数
│   └── assets/
│       └── robot/go1/
│           ├── scene.xml
│           └── scene_terrain.xml
│
├── simulation/
│   ├── rl_ppo_train.py           # 训练脚本
│   ├── rl_ppo_eval.py            # 评估脚本
│   └── rl_smoke_test.py          # 快速验证
│
├── runs/ppo/                      # 训练日志
│   ├── PPO_*/
│   │   └── events.out.tfevents.*  # TensorBoard 数据
│   └── models/
│       ├── ppo_quadruped_*.zip    # 检查点
│       └── ppo_quadruped_final.zip
│
└── docs/
    └── RL_V1_Design.md           # 本文档
```

---

## 📈 TensorBoard 监控

启动：
```bash
tensorboard --logdir=runs/ppo --port=6006
```

**关键指标**：
- `rollout/ep_rew_mean` - 每 episode 平均奖励
- `rollout/ep_len_mean` - 每 episode 平均长度
- `train/policy_loss` - 策略损失
- `train/value_loss` - 价值函数损失
- `train/entropy_loss` - 熵损失

---

## 🚀 下一步改进计划（V2）

### 短期修复
- [ ] 加入关节速度惩罚
- [ ] 加入关节加速度约束
- [ ] 测试更激进的 `rate_limit` 参数

### 中期优化
- [ ] 观测空间：加入关节速度统计（方差/方差/最大值）
- [ ] Curriculum Learning：分阶段提高平滑度要求
- [ ] 奖励塑形：使用策略蒸馏或逆强化学习

### 长期架构
- [ ] 多命令学习（一个 policy 适应不同速度）
- [ ] 转移学习（MPC → RL）
- [ ] 物理可行性约束（显式添加到环境）

---

## 📝 总结

### ✅ 完成的
- 模块化 RL 环境框架
- 独立配置系统（RL vs MPC）
- 5 分量奖励函数
- 平滑动作机制
- 并行 PPO 训练管道

### ⚠️ 待解决
- 高频微震问题（正在调参）
- 奖励函数的精细平衡
- 观测空间可能不够丰富

### 🎯 当前最重要的
**加强对关节运动的直接约束**，而不是间接通过奖励。

