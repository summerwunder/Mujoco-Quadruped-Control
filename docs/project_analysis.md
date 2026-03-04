# 四足机器人控制系统项目分析文档

> **分析日期:** 2026年2月26日
> **项目路径:** `/home/mingrui/Documents/Project/mujoco_project/quadruped`

---

## 1. 项目概述

### 1.1 项目目标

本项目是一个基于 MuJoCo 仿真环境的四足机器人运动控制框架，旨在实现四足机器人的**平衡控制**、**步态规划**和**轨迹跟踪**。项目采用模块化设计，支持多种控制策略（PD控制器、NMPC控制器）和多种步态（Trot、Pace、Crawl等）。

### 1.2 核心特性

- **物理仿真**: 基于 MuJoCo 高性能物理引擎
- **模型支持**: 支持 Unitree Go1、Go2 四足机器人模型
- **控制算法**: 
  - 基于非线性模型预测控制（NMPC）的全身控制
  - 传统 PD 关节控制器
- **步态规划**: 支持多种周期性步态（Trot、Pace、Crawl、Full Stance）
- **轨迹生成**: 五阶贝塞尔曲线摆动腿轨迹规划
- **地形适应**: 地形估计与自适应落脚点规划

### 1.3 技术栈

| 类别 | 技术 |
|------|------|
| 仿真引擎 | MuJoCo 3.4.0 |
| 编程语言 | Python 3.x |
| 优化求解器 | Acados (CasADi) |
| 数值计算 | NumPy, SciPy |
| 强化学习接口 | Gymnasium |
| 配置管理 | YAML |

---

## 2. 项目架构

### 2.1 目录结构

```
quadruped/
├── quadruped_ctrl/                 # 核心控制库
│   ├── __init__.py
│   ├── datatypes.py               # 数据类型定义
│   ├── quadruped_env.py           # Gymnasium 仿真环境
│   ├── assets/                    # MuJoCo 模型资源
│   │   └── robot/
│   │       ├── go1/               # Go1 机器人模型
│   │       │   ├── go1.xml        # 机器人本体定义
│   │       │   ├── scene.xml      # 仿真场景
│   │       │   └── scene_terrain.xml
│   │       └── go2/               # Go2 机器人模型
│   ├── config/                    # 配置文件
│   │   ├── sim_config.yaml        # 仿真参数
│   │   ├── robot/
│   │   │   ├── go1.yaml           # Go1 物理参数
│   │   │   └── go2.yaml           # Go2 物理参数
│   │   └── mpc/
│   │       └── go1_mpc_config.yaml # MPC 控制器参数
│   ├── controllers/               # 控制器模块
│   │   ├── controller_base.py     # 控制器基类
│   │   ├── controller_factory.py  # 控制器工厂
│   │   ├── pd/                    # PD 控制器
│   │   └── nmpc_gradient/         # NMPC 控制器
│   │       ├── quadruped_model.py      # 动力学模型
│   │       ├── controller_handler.py   # MPC 求解器
│   │       └── controller_constraint.py # 约束定义
│   ├── planning/                  # 规划模块
│   │   ├── periodic_gait_generator.py   # 步态生成器
│   │   ├── swing_trajectory_generator.py # 摆动轨迹生成
│   │   ├── foothold_reference_generator.py # 落脚点规划
│   │   └── terrain_estimator.py   # 地形估计
│   ├── interface/                 # 接口模块
│   │   ├── wb_interface.py        # 全身控制接口
│   │   └── reference_interface.py # 参考状态接口
│   └── utils/                     # 工具模块
│       ├── inverse_kinematics.py  # 逆运动学求解
│       ├── config_loader.py       # 配置加载器
│       ├── visual.py              # 可视化工具
│       └── terrain_generator.py   # 地形生成
├── simulation/                    # 仿真演示
│   ├── stand_demo.py              # 站立演示
│   └── stay_demo.py               # 保持姿态演示
├── setup.py                       # 包安装配置
└── docs/                          # 文档目录
```

### 2.2 模块依赖关系

```
┌─────────────────────────────────────────────────────────────┐
│                     QuadrupedEnv (环境层)                    │
│  - 继承 Gymnasium.Env                                        │
│  - 管理仿真循环、状态观测、渲染                               │
└───────────────────────────┬─────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Controllers   │   │ Planning      │   │ Interface     │
│ (控制层)      │   │ (规划层)      │   │ (接口层)      │
├───────────────┤   ├───────────────┤   ├───────────────┤
│ - PD          │   │ - GaitGen     │   │ - WBInterface │
│ - NMPC        │   │ - SwingGen    │   │ - RefInterface│
│               │   │ - FootholdGen │   │               │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                ┌───────────────────────┐
                │   Datatypes (数据层)   │
                │   - QuadrupedState    │
                │   - LegJointMap       │
                │   - BaseState         │
                │   - ReferenceState    │
                └───────────────────────┘
```

---

## 3. 核心数据结构

### 3.1 QuadrupedState (四足机器人状态)

```python
@dataclass
class QuadrupedState:
    # 四条腿的状态
    FL: LegJointMap  # 前左
    FR: LegJointMap  # 前右
    RL: LegJointMap  # 后左
    RR: LegJointMap  # 后右
    
    # 基座状态
    base: BaseState
    
    # 时间戳
    time: float
    step_num: int
    
    # 全局状态数组
    qpos: np.ndarray  # 关节位置
    qvel: np.ndarray  # 关节速度
    tau_ctrl: np.ndarray  # 控制力矩
```

### 3.2 LegJointMap (单腿状态)

```python
@dataclass
class LegJointMap:
    leg_name: str
    
    # 索引信息
    qpos_idxs: np.ndarray  # 关节角度索引
    qvel_idxs: np.ndarray  # 关节速度索引
    tau_idxs: np.ndarray   # 控制力矩索引
    
    # 关节空间状态
    qpos: np.ndarray   # [hip, thigh, calf] 关节角度
    qvel: np.ndarray   # 关节角速度
    tau: np.ndarray    # 控制力矩
    
    # 笛卡尔空间状态 (基座坐标系)
    foot_pos: np.ndarray  # 足端位置
    foot_vel: np.ndarray  # 足端速度
    hip_pos: np.ndarray   # 髋关节位置
    
    # 接触信息
    contact_state: bool          # 是否接触地面
    contact_force: np.ndarray    # 接触力 (GRF)
    
    # Jacobian 矩阵
    jac_pos_world: np.ndarray   # 足端位置雅可比
    jac_pos_base: np.ndarray    # 基座坐标系雅可比
```

### 3.3 BaseState (基座状态)

```python
@dataclass
class BaseState:
    # 位置和姿态 (世界坐标系)
    pos: np.ndarray       # [x, y, z]
    quat: np.ndarray      # [w, x, y, z] 四元数
    rot_mat: np.ndarray   # 旋转矩阵
    euler: np.ndarray     # [roll, pitch, yaw]
    
    # 速度 (基座坐标系)
    lin_vel: np.ndarray   # 线速度
    ang_vel: np.ndarray   # 角速度
    
    # 质心
    com: np.ndarray       # 质心位置
```

---

## 4. 控制器模块分析

### 4.1 控制器架构

项目采用**工厂模式**管理控制器，通过 `ControllerFactory` 根据名称创建控制器实例：

```python
class ControllerFactory:
    @staticmethod
    def create_controller(name, env, **kwargs):
        if name == "pd":
            return PDController(env, **kwargs)
        elif name == "mpc_gradient":
            return Quadruped_NMPC_Handler(env, **kwargs)
```

### 4.2 PD 控制器

**文件:** `controllers/pd/pd_controller.py`

**功能:** 简单的关节空间 PD 控制

**控制律:**
```
τ = Kp * (q_des - q) - Kd * q̇
```

**参数:**
- `kp = 60.0`: 比例增益
- `kd = 3.5`: 微分增益

**适用场景:** 静态站立、简单姿态保持

### 4.3 NMPC 控制器

**文件:** `controllers/nmpc_gradient/controller_handler.py`

#### 4.3.1 动力学模型

**文件:** `controllers/nmpc_gradient/quadruped_model.py`

**状态向量 (30维):**
```
x = [body_pos(3), body_vel(3), euler(3), ang_vel(3),
     foot_pos_FL(3), foot_pos_FR(3), foot_pos_RL(3), foot_pos_RR(3),
     integrals(6)]
```

**控制输入 (24维):**
```
u = [foot_vel_FL(3), foot_vel_FR(3), foot_vel_RL(3), foot_vel_RR(3),
     foot_force_FL(3), foot_force_FR(3), foot_force_RL(3), foot_force_RR(3)]
```

**前向动力学方程:**
```python
# 身体平动
ẋ = v
v̇ = g + (1/m) * Σ(F_i * stance_i)

# 身体转动
θ̇ = ω_rotation_matrix⁻¹ * ω
ω̇ = I⁻¹ * (R_w_b * τ - ω × (I * ω))

# 足端运动
ṗ_foot = v_foot * (1 - stance) * (1 - proximity)
```

#### 4.3.2 优化问题配置

**预测时域:** 12 步 (可配置)

**代价函数:**
```
J = Σ ||Q * (x - x_ref)||² + ||R * (u - u_ref)||²
```

**权重矩阵 (从配置文件加载):**
| 状态变量 | 权重 |
|---------|------|
| Z位置 | 1200 |
| XY速度 | 200 |
| Z速度 | 200 |
| Roll/Pitch | 300 |
| 足端位置 | 100 |
| 力积分 | 10-30 |

#### 4.3.3 约束定义

**文件:** `controllers/nmpc_gradient/controller_constraint.py`

**1. 摩擦锥约束:**
```
|fx| ≤ μ * fz
|fy| ≤ μ * fz
fz_min ≤ fz ≤ fz_max
```

**2. 足端位置约束:**
- 支撑相: 锁定在当前位置 ±5mm
- 摆动相: 允许在参考落点 ±15cm 范围内

**3. 稳定性约束 (可选):**
- 静态稳定性: 重心在支撑多边形内
- ZMP 稳定性: 零力矩点在支撑多边形内

#### 4.3.4 求解器特性

- **Acados OCP Solver**: 快速非线性 MPC 求解
- **Warm Start**: 支持热启动加速求解
- **Fallback 策略**: 求解失败时使用上一帧解或静态平衡力

---

## 5. 规划模块分析

### 5.1 步态生成器

**文件:** `planning/periodic_gait_generator.py`

#### 5.1.1 核心算法

**相位计算:**
```python
phase = (time * frequency + offset) % 1.0
contact = (phase < duty_factor)
```

#### 5.1.2 支持的步态

| 步态 | 占空比 | 步频 | 相位偏移 [FL, FR, RL, RR] |
|------|--------|------|---------------------------|
| Trot | 0.6 | 1.4 Hz | [0.5, 1.0, 1.0, 0.5] |
| Pace | 0.7 | 1.4 Hz | [0.8, 0.3, 0.8, 0.3] |
| Crawl | 0.8 | 0.5 Hz | [0.0, 0.5, 0.75, 0.25] |
| Full Stance | - | - | 全部站立 |

#### 5.1.3 智能启停

```python
def update_start_and_stop(...):
    # 条件检测
    is_command_zero = ||ref_vel|| < 0.01
    is_robot_static = ||base_vel|| < 0.1
    is_posture_flat = |roll| < 0.05 && |pitch| < 0.05
    is_feet_home = foot_dist < 0.06
    
    # 状态切换
    if all_conditions_met:
        is_full_stance = True  # 进入全站立
    elif command_not_zero:
        is_full_stance = False  # 恢复步态
```

### 5.2 摆动轨迹生成器

**文件:** `planning/swing_trajectory_generator.py`

#### 5.2.1 五阶贝塞尔曲线

**控制点定义:**
```
P0, P1 = lift_off     # 起点锁定
P2, P3 = peak_height  # 控制隆起高度
P4, P5 = touch_down   # 终点锁定
```

**曲线方程:**
```
B(t) = Σ C(5,i) * (1-t)^(5-i) * t^i * P_i
```

**特性:**
- C² 连续性 (位置、速度、加速度均连续)
- 边界速度为零
- 平滑过渡

#### 5.2.2 参数

```yaml
swing_height: 0.06m    # 摆动高度
swing_duration: ~0.25s # 摆动周期
```

### 5.3 落脚点规划器

**文件:** `planning/foothold_reference_generator.py`

#### 5.3.1 Raibert 启发式

```python
# 身体移动补偿
raibert_offset = (stance_time / 2) * ref_velocity

# 速度误差补偿 (Capture Point)
vel_error_offset = sqrt(h / g) * (current_vel - ref_vel)

# 总偏移
total_offset = clip(raibert_offset + vel_error_offset, -0.15, 0.15)
```

#### 5.3.2 落脚点计算

```python
target_hip = hip_position + total_offset + lateral_offset
target_world = R_wb.T @ target_hip + base_position
```

### 5.4 地形估计器

**文件:** `planning/terrain_estimator.py`

**功能:** 根据足端位置估计地形坡度

**输出:**
- 地形 Roll/Pitch
- 地形高度
- 机器人相对高度

---

## 6. 接口模块分析

### 6.1 全身控制接口 (WBInterface)

**文件:** `interface/wb_interface.py`

#### 6.1.1 控制力矩计算

**支撑腿:**
```python
τ = -J^T * F_GRF  # 雅可比转置映射地面反力
```

**摆动腿:**
```python
τ = J^T * (Kp * pos_error + Kd * vel_error)

# 可选: 反馈线性化
τ += M * J^(-1) * (acc_cmd - J̇ * q̇) + C  # M:质量矩阵, C:科氏力
```

#### 6.1.2 参数

```yaml
swing_kp: 500
swing_kd: 10
```

### 6.2 参考状态接口 (ReferenceInterface)

**文件:** `interface/reference_interface.py`

**职责:**
1. 管理步态生成器
2. 管理摆动轨迹生成器
3. 管理落脚点规划器
4. 生成 MPC 参考状态

**输出:**
- `ReferenceState`: 参考位置、速度、姿态
- `contact_sequence`: 接触序列 (4, H)
- `swing_refs`: 摆动腿参考轨迹

---

## 7. 仿真环境分析

### 7.1 QuadrupedEnv

**文件:** `quadruped_env.py`

**继承:** `gymnasium.Env`

#### 7.1.1 观测空间 (40维)

```
obs = [base_pos(3), base_quat(4), base_lin_vel(3), base_ang_vel(3),
       joint_qpos(12), joint_qvel(12)]
```

#### 7.1.2 动作空间 (12维)

```
action = joint_torques(12)  # 范围: [-30, 30] N·m
```

#### 7.1.3 核心方法

| 方法 | 功能 |
|------|------|
| `reset()` | 重置仿真，返回初始观测 |
| `step(action)` | 执行一步仿真 |
| `update_state_from_mujoco()` | 从 MuJoCo 更新状态 |
| `render()` | 渲染可视化 |
| `_compute_contact_forces()` | 计算接触力 |
| `_compute_leg_jacobian()` | 计算雅可比矩阵 |

#### 7.1.4 关节映射

```python
# Go1 关节索引
FL: qpos[7:10], qvel[6:9], ctrl[0:3]
FR: qpos[10:13], qvel[9:12], ctrl[3:6]
RL: qpos[13:16], qvel[12:15], ctrl[6:9]
RR: qpos[16:19], qvel[15:18], ctrl[9:12]
```

---

## 8. 配置系统

### 8.1 仿真配置 (sim_config.yaml)

```yaml
physics:
  dt: 0.002          # 500Hz 仿真频率
  mpc_frequency: 80  # MPC 更新频率
  gravity: 9.81
  mu: 0.5            # 摩擦系数

optimize:
  use_feedback_linearization: true
  use_foothold_constraint: true
  use_warm_start: true

gait:
  active: "trot"
  gaits:
    trot:
      duty_factor: 0.6
      step_freq: 1.4
      phase_offsets: [0.5, 1.0, 1.0, 0.5]
```

### 8.2 机器人配置 (go1.yaml)

```yaml
robot_name: "go1"
physics:
  mass: 12.019
  inertia: [0.158, ...]  # 3x3 惯性矩阵

geometry:
  hip_height: 0.32

swing_control:
  kp: 500
  kd: 10
```

### 8.3 MPC 配置 (go1_mpc_config.yaml)

```yaml
horizon: 12
grf_max: 150.0
grf_min: 0.0

weights:
  Q_position: [0, 0, 1200]
  Q_velocity: [200, 200, 200]
  Q_base_angle: [300, 300, 0]
  
solver:
  use_ddp: false
  use_rti: false
  use_integrators: true
```

---

## 9. 机器人模型

### 9.1 Unitree Go1

**文件:** `assets/robot/go1/go1.xml`

#### 9.1.1 物理参数

| 参数 | 值 |
|------|-----|
| 总质量 | ~12 kg |
| 髋关节高度 | 0.32 m |
| 大腿长度 | 0.213 m |
| 小腿长度 | 0.213 m |
| 足端半径 | 0.023 m |

#### 9.1.2 关节配置

每条腿 3 个关节:
- **Hip (髋关节)**: 外展/内收，范围 [-0.863, 0.863] rad
- **Thigh (大腿)**: 屈/伸，范围 [-0.686, 4.501] rad
- **Calf (小腿)**: 屈/伸，范围 [-2.818, -0.888] rad

#### 9.1.3 执行器

- 类型: 电机控制
- 最大力矩: ±45.43 N·m
- 阻尼: 0.6 N·m·s/rad

---

## 10. 控制流程

### 10.1 完整控制回路

```
┌─────────────────────────────────────────────────────────────────┐
│                        主控制循环                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. 获取当前状态                                                   │
│    state = env.update_state_from_mujoco()                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. 计算参考状态                                                   │
│    ref_state, contact_seq, swing_refs =                         │
│        ref_interface.get_reference_state(...)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. MPC 求解最优 GRF                                              │
│    optimal_GRF, footholds = controller.get_action(...)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. 计算关节力矩                                                   │
│    tau = wb_interface.compute_tau(state, swing_refs, GRF)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. 执行仿真步                                                     │
│    env.step(tau)                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                         下一周期
```

---

## 11. 优势与特点

### 11.1 架构优势

1. **模块化设计**: 控制器、规划器、接口层分离，易于扩展
2. **工厂模式**: 控制器可插拔，方便切换不同控制策略
3. **数据类封装**: 使用 dataclass 清晰定义状态结构
4. **配置驱动**: 参数通过 YAML 文件管理，无需修改代码

### 11.2 算法特点

1. **NMPC 实时性**: 使用 Acados 快速求解器，支持实时控制
2. **约束完备**: 摩擦锥、足端位置、稳定性约束
3. **Warm Start**: 利用上一帧解加速收敛
4. **Fallback 机制**: 求解失败时的安全回退策略
5. **反馈线性化**: 支持摆动腿精确轨迹跟踪

### 11.3 工程实践

1. **RL 兼容**: 环境继承 Gymnasium，可直接用于强化学习训练
2. **可视化**: 支持速度向量、摆动轨迹、落脚点可视化
3. **逆运动学**: 数值 IK 求解器支持足端位置控制

---

## 12. 潜在改进方向

### 12.1 控制层面

1. **模型精度**: 当前使用单刚体模型，可考虑全身动力学模型
2. **鲁棒性**: 增加对外部扰动的估计和补偿
3. **自适应**: 在线参数辨识和自适应控制

### 12.2 规划层面

1. **动态步态**: 支持步态切换和自适应步态
2. **复杂地形**: 增强地形感知和非平坦地形规划
3. **障碍物**: 集成视觉避障

### 12.3 工程层面

1. **单元测试**: 增加测试覆盖率
2. **文档**: 完善 API 文档
3. **日志**: 结构化日志系统
4. **仿真**: 支持更多机器人模型

---

## 13. 快速上手

### 13.1 安装依赖

```bash
pip install -r quadruped_ctrl/requirements.txt
pip install -e .
```

### 13.2 运行演示

```bash
# PD 控制器站立演示
python simulation/stand_demo.py
```

### 13.3 使用示例

```python
from quadruped_ctrl.quadruped_env import QuadrupedEnv
from quadruped_ctrl.controllers.controller_factory import ControllerFactory

# 创建环境
env = QuadrupedEnv(robot_config='robot/go1.yaml')

# 创建控制器
controller = ControllerFactory.create_controller("mpc_gradient", env)

# 控制循环
obs, _ = env.reset()
while True:
    state = env.get_state()
    action = controller.get_action(state, ...)
    obs, _, terminated, truncated, _ = env.step(action)
```

---

## 附录 A: 关键公式汇总

### A.1 前向动力学

```
线加速度:  a = g + (1/m) * Σ F_i
角加速度:  α = I⁻¹ * (τ - ω × (I * ω))
```

### A.2 雅可比映射

```
足端速度:  v_foot = J * q̇
足端力矩:  τ = -J^T * F
```

### A.3 贝塞尔曲线

```
B(t) = Σᵢ C(n,i) * (1-t)^(n-i) * t^i * Pᵢ
```

### A.4 Raibert 落脚点

```
p_foot = p_hip + (T_stance/2) * v_ref + k * (v - v_ref)
```

---

## 附录 B: 参考文献

1. Di Carlo, J., et al. "Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control." IROS 2018.
2. Bledt, G., et al. "MIT Cheetah 3: Design and Control of a Robust, Dynamic Quadruped Robot." IROS 2018.
3. Hutter, M., et al. "ANYmal - a highly mobile and dynamic quadrupedal robot." ICRA 2016.
4. Acados: https://docs.acados.org/

---

*文档结束*