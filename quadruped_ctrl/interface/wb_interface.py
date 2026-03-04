from quadruped_ctrl.datatypes import QuadrupedState
import numpy as np
from typing import Optional
from quadruped_ctrl.utils.inverse_kinematics import InverseKinematics

class WBInterface:
    """Whole-Body 控制接口基类
    
    配置参数来源：
    - swing_kp, swing_kd: 从环境的 robot_config (加载自 robot/*.yaml)
    - use_swing_tau_forward_optimization, use_friction_compensation: 从环境的 sim_config (加载自 sim_config.yaml)
    """
    def __init__(self, env):
        self.env = env
        
        # IK求解器配置
        ik_config = env.sim_config.get('ik_solver', {})
        self.use_ik = ik_config.get('use_ik', True)
        self.ik_solver = InverseKinematics(
            env,
            ik_iterations=ik_config.get('ik_iterations'),
            ik_dt=ik_config.get('ik_dt'),
            damping=ik_config.get('ik_damping')
        )
        self.max_pos_diff = ik_config.get('max_pos_diff', 2.0)  
        self.max_vel_diff = ik_config.get('max_vel_diff', 5.0)  
        # 从环境的 robot 获取摆动腿 PD 参数
        self.swing_kp = self.env.robot.swing_kp
        self.swing_kd = self.env.robot.swing_kd

 
        self.use_swing_tau_forward_optimization = env.sim_config.get('optimize', {}).get('use_swing_tau_forward_optimization', False)
    def compute_tau(self, state: QuadrupedState,
                    swing_targets: Optional[dict] = None,
                    contact_sequence: Optional[np.ndarray] = None,
                    optimal_GRF: Optional[np.ndarray] = None) -> np.ndarray:
        """计算控制力矩
        
        Args:
            state: 机器人状态
            swing_targets: 摆动腿目标 {'FL': {'pos': [3,], 'vel': [3,], 'acc': [3,]}, ...}
            contact_sequence: 计划的接触序列 (4,) 或 (4, H)，用于覆盖测量接触
            optimal_GRF: MPC 输出的 GRF (12,)，用于支撑腿力矩计算
            
        Returns:
            控制力矩 (12,) - [FL(3), FR(3), RL(3), RR(3)]
        """
        tau_total = np.zeros(12)
        des_foot_pos = {}
        des_foot_vel = {}
        
        for leg_idx, leg_name in enumerate(['FL', 'FR', 'RL', 'RR']):
            leg = getattr(state, leg_name) 

            
            planned_contact = contact_sequence[leg_idx]
            if isinstance(planned_contact, (np.ndarray, list)):
                planned_contact = planned_contact[0]
            is_stance = bool(planned_contact)
            target = swing_targets.get(leg_name) 
            if is_stance:
                # ========== 支撑腿力矩 ==========
                des_foot_pos[leg_name] = leg.foot_pos_world.copy()
                des_foot_vel[leg_name] = leg.foot_vel_world.copy()
                # τ = J^T · F (雅可比转置映射接触力)
                J_leg = leg.jac_pos_world[:, leg.qvel_idxs]  # (3, 3) - 只提取该腿3个关节的列
                if optimal_GRF is not None:
                    force = optimal_GRF[leg_idx * 3:(leg_idx + 1) * 3]
                else:
                    force = leg.contact_force
                tau = -J_leg.T @ force  
                tau_total[leg_idx*3:(leg_idx+1)*3] = tau
                
            else:
                des_foot_pos[leg_name] = target['pos'] if target else leg.foot_pos_world.copy()
                des_foot_vel[leg_name] = target['vel'] if target else leg.foot_vel_world.copy()
                # ========== 摆动腿力矩 ==========
                target = swing_targets.get(leg_name) 
                tau_swing = self.compute_swing_leg_tau(leg, target)
                tau_total[leg_idx*3:(leg_idx+1)*3] = tau_swing
        
        if self.use_ik:
            q_solution = self.ik_solver.compute_ik(des_foot_pos)
        else:
            q_solution = self.env.data.qpos.copy()
        
        des_joints_pos = {}
        des_joints_vel = {}
        for leg_idx, leg_name in enumerate(['FL', 'FR', 'RL', 'RR']):
            leg = state.get_leg_by_name(leg_name)
            des_joints_pos[leg_name] = q_solution[leg.qpos_idxs]
            J_leg = leg.jac_pos_world[:, leg.qvel_idxs]  # (3, 3)
            des_joints_vel[leg_name] = np.linalg.pinv(J_leg) @ des_foot_vel[leg_name]
          
        for leg_idx, leg_name in enumerate(['FL', 'FR', 'RL', 'RR']):
            leg = state.get_leg_by_name(leg_name)
            pos_diff = des_joints_pos[leg_name] - leg.qpos
            vel_diff = des_joints_vel[leg_name] - leg.qvel
            
            des_joints_pos[leg_name] = leg.qpos + np.clip(pos_diff, -self.max_pos_diff, self.max_pos_diff)
            des_joints_vel[leg_name] = leg.qvel + np.clip(vel_diff, -self.max_vel_diff, self.max_vel_diff)
            
        return tau_total, des_joints_pos, des_joints_vel
    
    def compute_swing_leg_tau(self, leg, target: Optional[dict] = None) -> np.ndarray:
        """计算摆动腿力矩：通过任务空间 PD 映射到关节空间
        """
        # target: {'pos': [3,], 'vel': [3,], 'acc': [3,]}  
        target = target or {}
        
        d_pos = target.get('pos')
        d_vel = target.get('vel')
        d_acc = target.get('acc')

        J = leg.jac_pos_base[:, leg.qvel_idxs]      # (3x3)
        J_dot = leg.jac_dot_base[:, leg.qvel_idxs] # (3x3)

        # 1. 计算笛卡尔空间目标加速度 (PD)
        # 这里的 Kp/Kd 需要针对动态摆动调优，通常比站立时要小一点，防止撞地
        error_pos = d_pos - leg.foot_pos
        error_vel = d_vel - leg.foot_vel
        v_acc = d_acc + self.swing_kp * error_pos + self.swing_kd * error_vel
        
        tau_swing = J.T @ (self.swing_kp * error_pos + self.swing_kd * error_vel)
        if self.use_swing_tau_forward_optimization:
            tau_swing += leg.mass_matrix @ np.linalg.pinv(J) @ (v_acc - J_dot @ leg.qvel) + leg.qfrc_bias
        return tau_swing
