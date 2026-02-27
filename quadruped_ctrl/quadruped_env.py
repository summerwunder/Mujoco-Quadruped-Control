"""
Quadruped robot environment.
Inherits from gym.Env for RL training.
"""

from __future__ import annotations

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
from .planning.swing_trajectory_generator import SwingTrajectoryGenerator
from .utils.visual import plot_swing_trajectory, render_vector, render_sphere
from .datatypes import QuadrupedState, RobotConfig, Trajectory, LegJointMap, BaseState
from .utils.config_loader import ConfigLoader


class QuadrupedEnv(gym.Env):
    """四足机器人环境，继承 gym.Env"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self, robot_config: Optional[str] = None, 
                 model_path: Optional[str] = None,
                 sim_config_path: Optional[str] = None,
                 rl_config_path: Optional[str] = None,
                 ref_base_lin_vel: Optional[np.ndarray] = None,
                 ref_base_ang_vel: Optional[np.ndarray] = None):
        """初始化环境
        Args:
            robot_config: 机器人配置，如果为None则使用默认配置
            model_path: MuJoCo模型XML文件路径，如果为None则使用默认路径
            sim_config_path: 仿真配置文件路径，如果为None则使用 'sim_config.yaml'
        """
        super().__init__()
        
        if robot_config is None:
            self.robot = ConfigLoader.load_builtin_config('go1')
        else:
            self.robot = ConfigLoader.load_robot_config(robot_config)

        if sim_config_path is None:
            sim_config_path = 'sim_config.yaml'
        self.sim_config_path = sim_config_path
        self.sim_config = ConfigLoader.load_sim_config(sim_config_path)   

        if rl_config_path is None:
            rl_config_path = 'rl_config.yaml'
        self.rl_config_path = rl_config_path
        self.rl_config = ConfigLoader.load_rl_config(rl_config_path)
        
        if model_path is None:
            module_dir = os.path.dirname(__file__)
            model_path = os.path.join(module_dir, 'assets', 'robot', 'go1', 'scene.xml')
        
        self.show_velocity_vector = self.sim_config.get('render', {}).get('show_velocity_vector', False)
        self.show_swing_trajectory = self.sim_config.get('render', {}).get('show_swing_trajectory', False)
        self.show_footholds = self.sim_config.get('render', {}).get('show_footholds', False)
        
        self.verbose = self.sim_config.get('verbose', False)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        
        self.state: Optional[QuadrupedState] = None
        self.ref_base_lin_vel = ref_base_lin_vel if ref_base_lin_vel is not None else np.zeros(3)
        self.ref_base_ang_vel = ref_base_ang_vel if ref_base_ang_vel is not None else np.zeros(3)
        self.base_lin_vel_noise_std = self.rl_config.get('obs_noise', {}).get(
            'base_lin_vel_std', 0.01
        )
        
        self.current_step = 0
        self.max_steps = 1000
        # self.dt = self.model.opt.timestep
        self.dt = self.sim_config.get('physics', {}).get('dt', 0.002)
        self.mu = self.sim_config.get('physics', {}).get('mu', 0.5)
        self._setup_joint_mapping()
        
        # Store the ids of visual aid geometries
        self._geom_ids = {}
        self._swing_geom_ids = None
        
        n_dof = self.robot.get_total_dof()  # 12 (3 DOF per leg * 4 legs) 
        # 观测空间（RL）：关节角度、关节速度、IMU姿态(roll,pitch)、
        # 基座线速度(带噪声)、足端压力标量、动作历史
        # [qpos(12), qvel(12), roll_pitch(2), lin_vel(3), command(3), foot_pressure(4), prev_action(12)]
        obs_dim = 12 + 12 + 2 + 3 + 3 + 4 + 12
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # 动作空间：关节位置偏移（相对于默认站姿）
        action_cfg = self.rl_config.get('action', {})
        self.action_range = float(action_cfg.get('range', 1.0))
        self.action_offset_scale = float(action_cfg.get('offset_scale', 0.5))
        self.action_max_delta = float(action_cfg.get('max_delta', 0.1))
        smooth_cfg = action_cfg.get('smooth', {})
        self.action_smooth_type = str(smooth_cfg.get('type', 'none')).lower()
        self.action_smooth_alpha = float(smooth_cfg.get('alpha', 0.2))
        self.action_smooth_max_delta = float(smooth_cfg.get('max_delta', self.action_max_delta))
        self.action_space = spaces.Box(
            low=-self.action_range,
            high=self.action_range,
            shape=(n_dof,),
            dtype=np.float32
        )
        self.stand_pose_qpos = np.array([0.0, 0.7, -1.4] * 4, dtype=np.float32)
        self.prev_action = np.zeros(n_dof, dtype=np.float32)
        self.kp = float(self.robot.swing_kp)
        self.kd = float(self.robot.swing_kd)
        reward_cfg = self.rl_config.get('reward', {})
        self.reward_vel_sigma = float(reward_cfg.get('vel_sigma', 0.25))
        weights = reward_cfg.get('weights', {})
        self.reward_w_vel = float(weights.get('vel', 1.0))
        self.reward_w_alive = float(weights.get('alive', 0.2))
        self.reward_w_tilt = float(weights.get('tilt', 0.5))
        self.reward_w_energy = float(weights.get('energy', 0.0001))
        self.reward_alive = float(reward_cfg.get('alive', 1.0))
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境
        
        Args:
            seed: 随机种子
            options: 重置选项
        
        Returns:
            初始观测和信息字典
        """
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)

        if self.model.nkey > 0:
            init_qpos = self.model.key_qpos[0].copy() 
        else:
            init_qpos = np.zeros(self.model.nq, dtype=np.float64)
            

        for leg_name in ['FL', 'FR', 'RL', 'RR']:
            qpos_idxs = self.joint_idx_map[leg_name]['qpos_idxs']
            self.data.qpos[qpos_idxs] = init_qpos[qpos_idxs]
            
        self.data.qpos[0:2] = np.array([0.0, 0.0])
        self.data.qpos[2] = self.robot.hip_height
        
        mujoco.mj_forward(self.model, self.data)

        self.update_state_from_mujoco()
        
        self.current_step = 0
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        
        obs = self.get_observation()
        info = self.get_info()
        
        return obs, info
    
    # TODO: update reference velocity
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步仿真
        
        Args:
            action: 控制动作 (n_dof,)
        
        Returns:
            观测、奖励、完成标志、信息字典
        """
        # 设置控制力矩（MPC 直接输出力矩）
        self.data.ctrl[:] = action
        
        # 执行一步仿真
        mujoco.mj_step(self.model, self.data)
        
        # 更新状态
        self.update_state_from_mujoco()
        
        # 获取观测
        obs = self.get_observation()
        
        # 计算奖励
        reward, reward_info = self._compute_reward()
        
        # 检查是否完成
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # 更新步数
        self.current_step += 1
        
        # 信息字典
        info = {'step': self.current_step}
        info.update(reward_info)
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode: str = 'human', swing_vis: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """渲染环境
        
        Args:
            mode: 渲染模式
            swing_vis: 摆动轨迹可视化数据字典
        """
        if self.viewer is None and mode == 'human':
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=True,
                show_right_ui=True,
                # key_callback=lambda x: self._key_callback(x),
            )
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)
            
        if self.viewer is None:
            return None
        
        base_pos = self.state.base.pos if self.state is not None else np.zeros(3)   
        if self.show_velocity_vector:     
            current_vel = self.state.base.lin_vel_world if self.state is not None else np.zeros(3)
            ref_vel = self.ref_base_lin_vel
            
            ref_vec_id = self._geom_ids.get('ref_vel_vec', -1)
            self._geom_ids['ref_vel_vec'] = render_vector(
                viewer=self.viewer,
                vector=ref_vel,
                pos=base_pos + np.array([0, 0, 0.1]),  
                scale=np.linalg.norm(ref_vel) + 1e-3,  
                color=np.array([1, 0, 0, 0.8]),      
                geom_id=ref_vec_id
            )
            curr_vec_id = self._geom_ids.get('curr_vel_vec', -1)
            # 4. 渲染实际速度箭头 (青色)
            curr_vec_id = self._geom_ids.get('curr_vel_vec', -1)
            self._geom_ids['curr_vel_vec'] = render_vector(
                viewer=self.viewer,
                vector=current_vel,
                pos=base_pos + np.array([0, 0, 0.2]), 
                scale=np.linalg.norm(current_vel) + 1e-3,
                color=np.array([0, 1, 1, 0.8]),       
                geom_id=curr_vec_id
            )
        
        if self.show_swing_trajectory and swing_vis is not None:
            swing_generator = swing_vis.get('swing_generator')
            swing_period = swing_vis.get('swing_period')
            swing_time = swing_vis.get('swing_time')
            lift_off_positions = swing_vis.get('lift_off_positions')
            nmpc_footholds = swing_vis.get('nmpc_footholds')
            ref_feet_pos = swing_vis.get('ref_feet_pos')
            
            if all([swing_generator, swing_period, swing_time, lift_off_positions, nmpc_footholds, ref_feet_pos]):
                self._swing_geom_ids = plot_swing_trajectory(
                    viewer=self.viewer,
                    swing_generator=swing_generator,
                    swing_period=swing_period,
                    swing_time=swing_time,
                    lift_off_positions=lift_off_positions,
                    nmpc_footholds=nmpc_footholds,
                    ref_feet_pos=ref_feet_pos,
                    geom_ids=self._swing_geom_ids,
                )
        
        # 渲染落脚点（NMPC优化结果 vs 参考落脚点）
        if self.show_footholds and swing_vis is not None:
            nmpc_footholds = swing_vis.get('nmpc_footholds')
            ref_feet_pos = swing_vis.get('ref_feet_pos')
            
            
            # 🔵 参考落脚点
            if ref_feet_pos is not None:
                for leg_name in ['FL', 'FR', 'RL', 'RR']:
                    ref_pos = ref_feet_pos.get(leg_name)
                    if ref_pos is not None:
                        geom_id_key = f'ref_foothold_{leg_name}'
                        old_geom_id = self._geom_ids.get(geom_id_key, -1)
                        self._geom_ids[geom_id_key] = render_sphere(
                            viewer=self.viewer,
                            position=ref_pos,
                            diameter=0.025,
                            color=np.array([0, 0, 1, 0.6]),  # 蓝色，半透明
                            geom_id=old_geom_id
                        )
                        
            # # 🟢 NMPC优化落脚点
            # if nmpc_footholds is not None:
            #     for leg_name in ['FL', 'FR', 'RL', 'RR']:
            #         nmpc_pos = nmpc_footholds.get(leg_name)
            #         if nmpc_pos is not None:
            #             geom_id_key = f'nmpc_foothold_{leg_name}'
            #             old_geom_id = self._geom_ids.get(geom_id_key, -1)
            #             self._geom_ids[geom_id_key] = render_sphere(
            #                 viewer=self.viewer,
            #                 position=nmpc_pos,
            #                 diameter=0.025,
            #                 color=np.array([0, 1, 0, 0.6]),  # 绿色，半透明
            #                 geom_id=old_geom_id
            #             )
        
        self.viewer.cam.lookat[:2] = base_pos[:2]
        self.viewer.cam.distance = 1.5
        self.viewer.sync()
            
        
    
    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def get_observation(self) -> np.ndarray:
        """从当前状态生成观测向量（用于RL/控制）
        
        Returns:
            观测向量 (obs_dim,)
            [joint_qpos(12), joint_qvel(12), roll_pitch(2),
             base_lin_vel(3), command(3), foot_pressure(4), prev_action(12)]
        """
        if self.state is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # 基座状态（速度使用基座坐标系，更适合本体感知）
        base_lin_vel = self.state.base.lin_vel  # (3,) 基座坐标系
        if self.base_lin_vel_noise_std > 0:
            base_lin_vel = base_lin_vel + np.random.normal(
                0.0, self.base_lin_vel_noise_std, size=3
            )
        roll_pitch = self.state.base.euler[:2].copy()
        command = np.array([
            self.ref_base_lin_vel[0],
            self.ref_base_lin_vel[1],
            self.ref_base_ang_vel[2],
        ], dtype=np.float32)

        joint_qpos = np.concatenate([
            self.state.FL.qpos,
            self.state.FR.qpos,
            self.state.RL.qpos,
            self.state.RR.qpos,
        ])  # (12,)
        
        joint_qvel = np.concatenate([
            self.state.FL.qvel,
            self.state.FR.qvel,
            self.state.RL.qvel,
            self.state.RR.qvel,
        ])  # (12,)

        foot_pressure = np.array([
            self.state.FL.contact_normal_force,
            self.state.FR.contact_normal_force,
            self.state.RL.contact_normal_force,
            self.state.RR.contact_normal_force,
        ], dtype=np.float32)
        
        # 组合观测
        obs = np.concatenate([
            joint_qpos,
            joint_qvel,
            roll_pitch,
            base_lin_vel,
            command,
            foot_pressure,
            self.prev_action,
        ], dtype=np.float32)
        
        return obs
    
    def set_state(self, state: QuadrupedState):
        """设置环境状态
        
        Args:
            state: 目标状态
        """
        self.state = state
    
    def get_state(self) -> QuadrupedState:
        """获取当前状态
        
        Returns:
            当前状态
        """
        return self.state
    
    def is_fallen(self) -> bool:
        """检查机器人是否跌倒
        
        Returns:
            是否跌倒
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """获取当前环境信息
        
        Returns:
            信息字典
        """
        return {
            'step': self.current_step,
            'time': self.current_step * self.dt,
        }

    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        if self.state is None:
            return 0.0, {}

        vel_cmd = np.array([
            self.ref_base_lin_vel[0],
            self.ref_base_lin_vel[1],
            self.ref_base_ang_vel[2],
        ], dtype=np.float32)
        vel_act = np.array([
            self.state.base.lin_vel_world[0],
            self.state.base.lin_vel_world[1],
            self.state.base.ang_vel_world[2],
        ], dtype=np.float32)

        vel_err = vel_cmd - vel_act
        sigma = max(self.reward_vel_sigma, 1e-6)
        r_vel = np.exp(-np.dot(vel_err, vel_err) / sigma)

        roll_pitch = self.state.base.euler[:2]
        p_tilt = float(np.dot(roll_pitch, roll_pitch))

        tau = self.state.tau_ctrl if self.state.tau_ctrl is not None else np.zeros(12)
        joint_qvel = np.concatenate([
            self.state.FL.qvel,
            self.state.FR.qvel,
            self.state.RL.qvel,
            self.state.RR.qvel,
        ])
        p_energy = float(np.sum(np.abs(tau * joint_qvel)))

        reward = (
            self.reward_w_vel * r_vel
            + self.reward_w_alive * self.reward_alive
            - self.reward_w_tilt * p_tilt
            - self.reward_w_energy * p_energy
        )
        info = {
            'r_vel': float(r_vel),
            'r_alive': float(self.reward_alive),
            'p_tilt': float(p_tilt),
            'p_energy': float(p_energy),
            'reward': float(reward),
        }
        return float(reward), info

    def map_rl_action_to_torque(self, rl_action: np.ndarray) -> np.ndarray:
        """将 RL 动作（关节位置偏移）映射为关节力矩，不影响 step。

        Args:
            rl_action: (12,) 关节位置偏移动作，范围建议在 [-1, 1]

        Returns:
            tau: (12,) 关节力矩
        """
        rl_action = np.clip(rl_action, -self.action_range, self.action_range)
        target_offset = rl_action * self.action_offset_scale
        if self.action_smooth_type == 'lowpass':
            alpha = np.clip(self.action_smooth_alpha, 0.0, 1.0)
            smoothed_offset = (1.0 - alpha) * self.prev_action + alpha * target_offset
        elif self.action_smooth_type == 'rate_limit':
            delta = target_offset - self.prev_action
            max_delta = self.action_smooth_max_delta
            delta = np.clip(delta, -max_delta, max_delta)
            smoothed_offset = self.prev_action + delta
        else:
            smoothed_offset = target_offset

        q_des = self.stand_pose_qpos + smoothed_offset
        qpos = np.concatenate([
            self.state.FL.qpos,
            self.state.FR.qpos,
            self.state.RL.qpos,
            self.state.RR.qpos,
        ])
        qvel = np.concatenate([
            self.state.FL.qvel,
            self.state.FR.qvel,
            self.state.RL.qvel,
            self.state.RR.qvel,
        ])
        tau = self.kp * (q_des - qpos) + self.kd * (0.0 - qvel)

        self.prev_action = smoothed_offset.astype(np.float32).copy()
        return tau.astype(np.float32)
    
    def _setup_joint_mapping(self):
        """设置关节索引映射
        - leg: FL, FR, RL, RR
        - joint: hip, thigh, calf
        """
        self.leg_joint_names = {
            'FL': ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint'],
            'FR': ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint'],
            'RL': ['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'],
            'RR': ['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'],
        }
        
        # 足端geom名称
        self.foot_geom_names = {
            'FL': 'FL',
            'FR': 'FR',
            'RL': 'RL',
            'RR': 'RR',
        }
        
        freejoint_dof = self.model.nv - self.model.nu
        
        # 构建索引映射
        self.joint_idx_map = {}
        for leg_name, joint_names in self.leg_joint_names.items():
            qpos_idxs = []
            qvel_idxs = []
            tau_idxs = []
            
            for joint_name in joint_names:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id == -1:
                    raise ValueError(f"Joint {joint_name} not found in model")
                
                # MuJoCo的关节索引
                qpos_adr = self.model.jnt_qposadr[joint_id]
                qvel_adr = self.model.jnt_dofadr[joint_id]
                
                qpos_idxs.append(qpos_adr)
                qvel_idxs.append(qvel_adr)
                for act_id in range(self.model.nu):
                    if self.model.actuator_trnid[act_id, 0] == joint_id:
                        tau_idxs.append(act_id)
           
            self.joint_idx_map[leg_name] = {
                'qpos_idxs': np.array(qpos_idxs, dtype=np.int32),
                'qvel_idxs': np.array(qvel_idxs, dtype=np.int32),
                'tau_idxs': np.array(tau_idxs, dtype=np.int32),  
            }
        
        self.foot_geom_ids = {}
        self.foot_body_ids = {}
        for leg_name in ['FL', 'FR', 'RL', 'RR']:
            # 找到calf body
            calf_body_name = f"{leg_name}_calf"
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, calf_body_name)
            if body_id == -1:
                raise ValueError(f"Calf body {calf_body_name} not found in model")
            self.foot_body_ids[leg_name] = body_id
            
            # 尝试通过名称查找foot geom
            geom_name = self.foot_geom_names[leg_name]
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            
            self.foot_geom_ids[leg_name] = geom_id

        # 髋关节 body id 映射（用于获取 hip 在世界坐标系下的位置）
        self.hip_body_ids = {}
        for leg_name in ['FL', 'FR', 'RL', 'RR']:
            hip_body_name = f"{leg_name}_hip"
            hip_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, hip_body_name)
            if hip_body_id == -1:
                hip_body_id = None
            self.hip_body_ids[leg_name] = hip_body_id
        
        # 基座body索引
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'trunk')
    
    def _bind_leg_joint_map(self, leg_name: str) -> LegJointMap:
        idx_map = self.joint_idx_map[leg_name]
        leg = LegJointMap(
            leg_name=leg_name,
            qpos_idxs=idx_map['qpos_idxs'],
            qvel_idxs=idx_map['qvel_idxs'],
            tau_idxs=idx_map['tau_idxs'],
        )
        
        return leg
    
    def update_state_from_mujoco(self):
        """从MuJoCo data更新QuadrupedState"""
        if self.state is None:
            # 初始化state
            self.state = QuadrupedState(
                FL=self._bind_leg_joint_map('FL'),
                FR=self._bind_leg_joint_map('FR'),
                RL=self._bind_leg_joint_map('RL'),
                RR=self._bind_leg_joint_map('RR'),
                base=BaseState(),
            )
        
        # 更新基座状态
        # MuJoCo freejoint: qpos[0:3]位置, qpos[3:7]四元数(w,x,y,z)
        self.state.base.pos = self.data.qpos[0:3].copy()
        self.state.base.pos_centered = np.zeros(3, dtype=np.float64)
        self.state.base.com = self._compute_com()
        self.state.base.com_centered = self.state.base.com - self.state.base.pos  # 中心化质心
        self.state.base.quat = self.data.qpos[3:7].copy()
        
        # 设置旋转矩阵和欧拉角
        self.state.base.set_from_quat(self.state.base.quat)
        
        # 速度 (freejoint: qvel[0:3]线速度, qvel[3:6]角速度)
        self.state.base.lin_vel_world = self.data.qvel[0:3].copy()
        self.state.base.ang_vel_world = self.data.qvel[3:6].copy()
      
        self.state.base.lin_vel = self.state.base.rot_mat.T @ self.state.base.lin_vel_world
        self.state.base.ang_vel = self.state.base.rot_mat.T @ self.state.base.ang_vel_world  
        # 重力向量（在基座坐标系中）
        gravity_world = np.array([0, 0, -1], dtype=np.float64)
        self.state.base.gravity_vec = self.state.base.rot_mat.T @ gravity_world
        
        mass_matrix_full = self._compute_mass_matrix()
        # 更新每条腿的状态
        for leg_name in ['FL', 'FR', 'RL', 'RR']:
            leg = self.state.get_leg_by_name(leg_name)
            # 惯性矩阵
            leg.mass_matrix = mass_matrix_full[np.ix_(leg.qvel_idxs, leg.qvel_idxs)].copy()
            # 偏置力
            leg.qfrc_bias = self.data.qfrc_bias[leg.qvel_idxs].copy()
            # 关节状态
            leg.qpos = self.data.qpos[leg.qpos_idxs].copy()
            leg.qvel = self.data.qvel[leg.qvel_idxs].copy()
            leg.tau = self.data.ctrl[leg.tau_idxs].copy()
            
            # 足端位置 (需要通过正运动学计算或从site获取)
            geom_id = self.foot_geom_ids[leg_name]
            if geom_id is not None:
                leg.foot_pos_world = self.data.geom_xpos[geom_id].copy()
            else:
                print("error:not found foot geom, use body pos instead")
                # 使用calf body位置
                body_id = self.foot_body_ids[leg_name]
                leg.foot_pos_world = self.data.xpos[body_id].copy()
            # 转换到基座坐标系
            base_pos = self.state.base.pos
            base_rot = self.state.base.rot_mat
            leg.foot_pos_centered = leg.foot_pos_world - base_pos
            leg.foot_pos = base_rot.T @ (leg.foot_pos_world - base_pos)
            # 髋关节位置：优先使用 hip_body_ids 获取世界坐标位置，并转换到基座坐标系
            hip_body_id = None
            if hasattr(self, 'hip_body_ids'):
                hip_body_id = self.hip_body_ids.get(leg_name, None)

            if hip_body_id is not None:
                leg.hip_pos_world = self.data.xpos[hip_body_id].copy()
                leg.hip_pos = base_rot.T @ (leg.hip_pos_world - base_pos)
            else:
                print(f"Warning: hip body for leg {leg_name} not found, setting hip position to zero")
                raise ValueError(f"Hip body for leg {leg_name} not found in model")
            
            # 计算 Jacobian 矩阵
            self._compute_leg_jacobian(leg_name)
            
            J = leg.jac_pos_base[:, leg.qvel_idxs]      # (3x3)
            leg.foot_vel = J @ leg.qvel
            leg.foot_vel_world = base_rot @ leg.foot_vel
        
        # 更新全局状态数组
        self.state.qpos = self.data.qpos.copy()
        self.state.qvel = self.data.qvel.copy()
        self.state.tau_ctrl = self.data.ctrl.copy()
        self.state.time = self.data.time
        self.state.step_num = self.current_step
        
        # 计算接触力
        self._compute_contact_forces()
    
    def _compute_mass_matrix(self):
        mass_matrix = np.zeros((self.model.nv, self.model.nv), dtype=np.float64)
        mujoco.mj_fullM(self.model, mass_matrix, self.data.qM)
        return mass_matrix
    
    def _compute_com(self) -> np.ndarray:
        """计算机器人质心位置"""
        com = np.zeros(3, dtype=np.float64)
        total_mass = 0.0
        for i in range(self.model.nbody):
            body_mass = self.model.body_mass[i]
            body_pos = self.data.subtree_com[i]
            com += body_mass * body_pos
            total_mass += body_mass
        if total_mass > 0:
            com /= total_mass
        return com

    
    def _compute_contact_forces(self):
        """计算每条腿的接触力和接触状态"""
        for leg_name in ['FL', 'FR', 'RL', 'RR']:
            leg = getattr(self.state, leg_name)
            leg.contact_state = False
            leg.contact_force = np.zeros(3, dtype=np.float64)
            leg.contact_normal_force = 0.0
        
        if any(body_id is None or body_id == -1 for body_id in self.foot_body_ids.values()):
            return

        floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            if floor_geom_id != -1:
                if contact.geom1 == floor_geom_id:
                    foot_geom = contact.geom2
                elif contact.geom2 == floor_geom_id:
                    foot_geom = contact.geom1
                else:
                    continue
                foot_body = self.model.geom_bodyid[foot_geom]
            else:
                body1 = self.model.geom_bodyid[contact.geom1]
                body2 = self.model.geom_bodyid[contact.geom2]
                if not (body1 == 0 or body2 == 0):
                    continue
                foot_body = body2 if body1 == 0 else body1
            leg_name = None
            for leg in ['FL', 'FR', 'RL', 'RR']:
                if foot_body == self.foot_body_ids[leg]:
                    leg_name = leg
                    break
            
            if leg_name is None:
                continue
            
            force_contact = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force_contact)

            if force_contact[0] < 0.01:
                continue
            
            R = contact.frame.reshape(3, 3)
            force_world = R.T @ force_contact[:3]

            leg = getattr(self.state, leg_name)
            leg.contact_state = True
            leg.contact_force += force_world
            leg.contact_normal_force += force_world[2]
    
    def _compute_leg_jacobian(self, leg_name: str):
        """计算单条腿足端的 Jacobian 矩阵及其导数 (世界坐标系和基座坐标系)
        Args:
            leg_name: 腿名称 ('FL', 'FR', 'RL', 'RR')
        """
        leg = getattr(self.state, leg_name)
        foot_body_id = self.foot_body_ids[leg_name]
        
        if foot_body_id is None or foot_body_id == -1:
            return
        jacp_world = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr_world = np.zeros((3, self.model.nv), dtype=np.float64)
        jac_dot_p_world = np.zeros((3, self.model.nv), dtype=np.float64)
        
        # 计算足端的平移和旋转 Jacobian（世界坐标系）
        mujoco.mj_jac(
            m=self.model,
            d=self.data,
            jacp=jacp_world,
            jacr=jacr_world,
            point=leg.foot_pos_world,
            body=foot_body_id,
        )
        mujoco.mj_jacDot(
            m=self.model,
            d=self.data,
            jacp=jac_dot_p_world,
            jacr=None,  
            point=leg.foot_pos_world,
            body=foot_body_id,
        )
        leg.jac_pos_world = jacp_world.copy()
        leg.jac_rot_world = jacr_world.copy()
        leg.jac_dot_world = jac_dot_p_world.copy()
        # ========== 基座坐标系下的 Jacobian ==========
        base_rot = self.state.base.rot_mat  
        leg.jac_pos_base = base_rot.T @ jacp_world
        leg.jac_rot_base = base_rot.T @ jacr_world
        leg.jac_dot_base = base_rot.T @ jac_dot_p_world
