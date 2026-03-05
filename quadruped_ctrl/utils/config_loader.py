"""
Configuration loader for robot configs from YAML files.
"""

from __future__ import annotations

import os
import yaml
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from quadruped_ctrl.datatypes import RobotConfig


class ConfigLoader:
    """配置文件加载器"""
    
    @staticmethod
    def load_robot_config(config_path: str) -> RobotConfig:
        """从YAML文件加载机器人配置
        
        Args:
            config_path: 配置文件路径 (绝对路径或相对于config目录)
        Returns:
            RobotConfig对象
        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置文件格式错误
        """
        # 处理路径
        if not os.path.isabs(config_path):
            # 相对路径：相对于本模块的config目录
            module_dir = os.path.dirname(__file__)
            config_path = os.path.join(module_dir, '..', 'config', config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            raise ValueError(f"Empty config file: {config_path}")
        
        return ConfigLoader._parse_config(data)
    
    @staticmethod
    def _parse_config(data: Dict[str, Any]) -> RobotConfig:
        """解析YAML数据到RobotConfig
        
        Args:
            data: YAML加载后的字典
        
        Returns:
            RobotConfig对象
        """
        robot_name = data.get('robot_name')
        
        # 物理参数
        physics = data.get('physics', {})
        mass = float(physics.get('mass', 12.0))
        gravity = float(physics.get('gravity', 9.81))
        inertia = np.array(physics.get('inertia', None)).flatten()
        
        # 几何参数
        geometry = data.get('geometry', {})
        hip_height = float(geometry.get('hip_height', 0.25))
        
        # 控制参数
        swing_control = data.get('swing_control', {})
        swing_kp = float(swing_control.get('kp', 60.0))
        swing_kd = float(swing_control.get('kd', 10.0))
        step_height = hip_height * 0.4
        # 创建配置对象
        config = RobotConfig(
            robot_name=robot_name,
            mass=mass,
            n_legs=4,
            dofs_per_leg=[3, 3, 3, 3],
            gravity=gravity,
            friction_coeff=1.0,
            inertia=inertia,
            hip_height=hip_height,
            foot_radius=0.01,
            swing_kp=swing_kp,
            swing_kd=swing_kd,
            step_height=step_height,
        )
        
        return config
    
    @staticmethod
    def load_builtin_config(robot_name: str) -> RobotConfig:
        """加载内置机器人配置
        
        Args:
            robot_name: 机器人名称 ('go1', 'go2', 等)
        Returns:
            RobotConfig对象       
        Raises:
            ValueError: 不支持的机器人名称
        """
        supported_robots = {
            'go1': 'robot/go1.yaml',
            'go2': 'robot/go2.yaml',
        }
        
        if robot_name not in supported_robots:
            raise ValueError(
                f"Unsupported robot: {robot_name}. "
                f"Supported: {list(supported_robots.keys())}"
            )
        
        config_file = supported_robots[robot_name]
        return ConfigLoader.load_robot_config(config_file)
    
    @staticmethod
    def load_sim_config(config_path: str = 'sim_config.yaml') -> Dict[str, Any]:
        """加载仿真配置
        
        Args:
            config_path: 仿真配置文件路径 (相对于config目录)
            
        Returns:
            仿真配置字典，包含:
            {
                'optimize': {'use_swing_tau_forward_optimization': bool, 'use_friction_compensation': bool},
                'physics': {'dt': float, 'mpc_frequency': int, 'scene': str},
                'gait': {...}
            }
        """
        if not os.path.isabs(config_path):
            module_dir = os.path.dirname(__file__)
            config_path = os.path.join(module_dir, '..', 'config', config_path)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Sim config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            sim_config = yaml.safe_load(f)
        
        if sim_config is None:
            raise ValueError(f"Empty sim config file: {config_path}")
        
        return sim_config

    @staticmethod
    def load_rl_config(config_path: str = 'rl_config.yaml') -> Dict[str, Any]:
        """加载强化学习配置

        Args:
            config_path: RL 配置文件路径（相对于 config 目录）

        Returns:
            RL 配置字典
        """
        if not os.path.isabs(config_path):
            module_dir = os.path.dirname(__file__)
            config_path = os.path.join(module_dir, '..', 'config', config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"RL config file not found: {config_path}")

        with open(config_path, 'r') as f:
            rl_config = yaml.safe_load(f)

        if rl_config is None:
            raise ValueError(f"Empty rl config file: {config_path}")

        return rl_config

    @staticmethod
    def load_gait_params(config_path: str = 'sim_config.yaml', gait_name: Optional[str] = None) -> Dict[str, Any]:
        """加载 gait 配置并返回指定 gait 或所有 gait 的参数。

        默认返回一个字典，键为 gait 名称，值为参数字典：
            { '<gait_name>': { 'type': int, 'duty_factor': float, 'step_freq': float, 'phase_offsets': np.ndarray }, ... }

        Args:
            config_path: 仿真配置文件路径（相对于 config 目录）
            gait_name: 如果指定，返回该 gait 的参数；否则返回所有 gait 的参数（不包含 `active` 字段）

        Returns:
            gait 字典或单个 gait 参数字典。
        """
        sim_cfg = ConfigLoader.load_sim_config(config_path)
        gait_section = sim_cfg.get('gait', {}) if sim_cfg is not None else {}

        gaits = gait_section.get('gaits', {})

        parsed_gaits = {}
        for name, params in gaits.items():
            if params is None:
                continue
            p_type = params.get('type', None)
            duty = float(params.get('duty_factor', 1.0))
            step_freq = float(params.get('step_freq', 0.0))
            phase = params.get('phase_offsets', None)
            margin = params.get('margin', None)
            try:
                phase_arr = np.array(phase, dtype=np.float64) if phase is not None else None
            except Exception:
                phase_arr = None

            parsed_gaits[name] = {
                'type': int(p_type) if p_type is not None else None,
                'duty_factor': duty,
                'step_freq': step_freq,
                'phase_offsets': phase_arr,
                'margin': float(margin) if margin is not None else None,
            }

        if gait_name is None:
            return parsed_gaits

        if gait_name not in parsed_gaits:
            raise KeyError(f"Gait '{gait_name}' not found in sim_config: {list(parsed_gaits.keys())}")

        return parsed_gaits[gait_name]

    @staticmethod
    def load_mpc_config(config_path: str = 'go1_mpc_config.yaml') -> Dict[str, Any]:
        """加载 MPC 配置文件（权重、代价矩阵、预测步长等）
        Args:
            config_path: MPC 配置文件路径（相对于 config 目录）
        Returns:
            字典形式的 MPC 配置，内部数组会被转换为 numpy.ndarray
        """
        if not os.path.isabs(config_path):
            module_dir = os.path.dirname(__file__)
            config_path = os.path.join(module_dir, '..', 'config', 'mpc', config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"MPC config file not found: {config_path}")

        with open(config_path, 'r') as f:
            mpc_config = yaml.safe_load(f)

        if mpc_config is None:
            raise ValueError(f"Empty mpc config file: {config_path}")

        # Convert lists to numpy arrays where appropriate
        weights = mpc_config.get('weights', {})
        converted_weights = {}
        for k, v in weights.items():
            try:
                converted_weights[k] = np.array(v, dtype=np.float64)
            except Exception:
                converted_weights[k] = v

        r_vals = mpc_config.get('R', {})
        converted_r = {}
        for k, v in r_vals.items():
            try:
                converted_r[k] = np.array(v, dtype=np.float64)
            except Exception:
                converted_r[k] = v

        mpc_config['weights'] = converted_weights
        mpc_config['R'] = converted_r

        return mpc_config
