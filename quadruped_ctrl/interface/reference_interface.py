# des_foot_pos, des_foot_vel, des_foot_acc 
from quadruped_ctrl.datatypes import ReferenceState
from quadruped_ctrl.datatypes import QuadrupedState
from quadruped_ctrl.utils.config_loader import ConfigLoader
from quadruped_ctrl.quadruped_env import QuadrupedEnv
from quadruped_ctrl.planning.swing_trajectory_generator import SwingTrajectoryGenerator
from quadruped_ctrl.planning.periodic_gait_generator import PeriodicGaitGenerator
from quadruped_ctrl.planning.foothold_reference_generator import FootholdGenerator
from quadruped_ctrl.planning.terrain_estimator import TerrainEstimator
import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation as R

class ReferenceInterface:
    def __init__(self, env: QuadrupedEnv, mpc_config_path:str):
        self.env = env
        self.robot = env.robot
        self.sim_config = env.sim_config
        self.mpc_config = ConfigLoader.load_mpc_config(mpc_config_path)
        
        dt = self.sim_config.get('physics', {}).get('dt', 0.002)
        horizon = self.mpc_config.get('horizon')
        if self.mpc_config.get('solver').get('use_nonuniform_discretization'):
            # TODO: 支持非均匀时间步长
            pass
        else:
            self.contact_sequence_dts = np.full(horizon, dt)
          
        self.gait_params = ConfigLoader.load_gait_params(self.env.sim_config_path)
        self.gait_active = self.sim_config.get('gait').get('active', 'trot')
        gait_param = self.gait_params.get(self.gait_active)
        # 初始化步态生成器
        self.gait_generator = PeriodicGaitGenerator(
            duty_factor=gait_param['duty_factor'],
            step_freq=gait_param['step_freq'],
            phase_offsets=gait_param['phase_offsets']
        )
        
        # 初始化摇摆轨迹生成器
        swing_height = self.robot.step_height 
        step_freq = float(self.gait_generator.step_freq)
        swing_duration = (1 - self.gait_generator.duty_factor) * (1 / step_freq)
        stance_time = (1 / step_freq) * self.gait_generator.duty_factor
        self.swing_generator = SwingTrajectoryGenerator(
             swing_height=swing_height,
             swing_duration=swing_duration)
        
        # 初始化 swing_time 追踪数组 (FL, FR, RL, RR)
        self.swing_time = np.zeros(4, dtype=np.float64)
        self.swing_period = swing_duration
        self.is_full_stance = (self.gait_active == 'full_stance') 
        # 落脚点生成器
        self.foothold_generator = FootholdGenerator(
            stance_time=stance_time,
            robot_height=self.robot.hip_height,
            gravity=self.env.sim_config.get('physics').get('gravity', 9.81)
        )
        
        self.terrain_estimator = TerrainEstimator(alpha=0.95)
        # feet_pos: np.ndarray,
        # hip_pos: np.ndarray,
        # joints_pos: np.ndarray,
    def get_reference_state(
        self,
        current_state: QuadrupedState,
        com_pos: np.ndarray,
        heightmaps,
        abs_time: float,
        ref_base_lin_vel: np.ndarray,
        ref_base_ang_vel: np.ndarray,
    ) -> Tuple[ReferenceState, np.ndarray, dict]:
        """计算参考状态      
        Args:
            current_state: 当前机器人状态
            com_pos: 机器人质心位置
            heightmaps: 地形高度图
            abs_time: 当前绝对时间
            ref_base_lin_vel: 参考线速度
            ref_base_ang_vel: 参考角速度
        
        Returns:
            ReferenceState对象
            contact_sequence: 支撑/摆动序列 (4, H)
            swing_refs: 摆动腿参考轨迹 {'FL': {'pos':(H,3),'vel':(H,3),'acc':(H,3)}, ...}
            
        """
        # TODO: 调整期望速度，如果机器人处于异常位置
        
        # 更新 gait 的 start/stop 判定
        # self.gait_generator.update_start_and_stop(
        #     base_lin_vel=current_state.base.lin_vel_world,
        #     base_ang_vel=current_state.base.ang_vel_world,
        #     ref_lin_vel=ref_base_lin_vel,
        #     ref_ang_vel=ref_base_ang_vel,
        #     feet_dist_to_hip_max=current_state.get_max_feet_dist_to_hip(),
        #     base_rpy=current_state.base.euler
        # )
        
        contact_sequence = self.gait_generator.get_horizon_sequence(
            abs_time, 
            self.contact_sequence_dts, 
            is_full_stance=self.is_full_stance
        )
        
        # 更新足端接触状态跟踪
        current_target_contact = self.gait_generator.get_contact_at_time(abs_time)
        self.foothold_generator.update_contact_states(current_state, current_target_contact)

        # Adjust reference velocity based on terrain estimation BEFORE foothold computation
        terrain_roll, terrain_pitch, terrain_height, robot_height = self.terrain_estimator.update(current_state)
        ref_base_lin_vel = R.from_euler("xyz", [terrain_roll, terrain_pitch, 0]).as_matrix() @ ref_base_lin_vel
        if(terrain_pitch > 0.0):
            ref_base_lin_vel[2] = -ref_base_lin_vel[2]
        if(np.abs(terrain_pitch) > 0.2):
            ref_base_lin_vel[0] = ref_base_lin_vel[0]/2.
            ref_base_lin_vel[2] = ref_base_lin_vel[2]*2

        # foothold generator expects reference velocity in world/base XY (2,)
        ref_lin_w = ref_base_lin_vel[0:2]
        ref_footholds = self.foothold_generator.compute_footholds(
            current_state,
            ref_lin_w
        )
        # TODO: ref_pos暂时写死未考虑实际位置
        ref_pos = np.array([0, 0, self.robot.hip_height])
        ref_pos[2] -= current_state.base.pos[2] - com_pos[2]
        
        
        # TODO: only test velocity tracking on flat terrain
        # ref_pos = current_state.base.pos.copy()
        # TODO: 目前只考虑 Roll 和 Pitch ，YAW应该是任务为导向
        # TODO：会有一个持续的负的 Pitch 导致机器人一直前倾，暂时先加个偏置抵消掉
        # 头部（集成了相机和雷达）具有一定的重量 。
        # 如果你的单刚体模型（SRBD）假设质量是均匀分布的，
        # 而没有考虑头部载荷导致的质心前移，MPC 算出的足底力分配就会不足以支撑前端，导致“低头” 。
        reference_orientation =  [-terrain_roll, -terrain_pitch, 0]   
        # 中心化：使用当前机器人位置作为原点
        center_pos = current_state.base.pos.copy()
        
        reference_state = ReferenceState(
            # 世界坐标系下的参考值
            ref_foot_FL = ref_footholds['FL'],
            ref_foot_FR = ref_footholds['FR'],
            ref_foot_RL = ref_footholds['RL'],
            ref_foot_RR = ref_footholds['RR'],
            ref_position = ref_pos,
            
            # 中心化坐标系下的参考值（减去当前机器人位置）
            ref_foot_FL_centered = ref_footholds['FL'] - center_pos,
            ref_foot_FR_centered = ref_footholds['FR'] - center_pos,
            ref_foot_RL_centered = ref_footholds['RL'] - center_pos,
            ref_foot_RR_centered = ref_footholds['RR'] - center_pos,
            ref_position_centered = ref_pos - center_pos,
            
            # 速度和姿态不需要中心化
            ref_linear_velocity = ref_base_lin_vel,
            ref_angular_velocity = ref_base_ang_vel,
            ref_orientation = reference_orientation,
        )
        
        
        # 计算 swing 参考轨迹（pos, vel, acc），以供控制器/日志使用
        swing_refs = self._compute_swing_references(
            current_state=current_state,
            contact_sequence=contact_sequence,
            foothold_targets=ref_footholds,
            dt_list=self.contact_sequence_dts
        )
        
        # if self.env.verbose and self.env.current_step % 50 == 0:
        #     for leg in ['FL', 'FR', 'RL', 'RR']:
        #         if contact_sequence[['FL', 'FR', 'RL', 'RR'].index(leg), 0] == 0:
        #             print(f"{leg} is swinging. Swing time: {self.swing_time[['FL', 'FR', 'RL', 'RR'].index(leg)]:.3f}s")
        #             print(f"{leg} swing_ref pos:", swing_refs[leg]['pos'].tolist())
                    
        return reference_state, contact_sequence, swing_refs

    def _compute_swing_references(self,
                                  current_state: QuadrupedState,
                                  contact_sequence: np.ndarray,
                                  foothold_targets: dict,
                                  dt_list: np.ndarray):
        """Compute swing pos/vel/acc references for the first timestep only.

        Returns dict: {leg: {'pos':(3,),'vel':(3,),'acc':(3,)}}
        """
        legs = ['FL', 'FR', 'RL', 'RR']
        # TODO: 先简单设置
        dt = float(dt_list[0])

        results = {}
        for i, leg_name in enumerate(legs):
            seq = contact_sequence[i, :]
            leg = current_state.get_leg_by_name(leg_name)
            
            # 更新 swing_time：摆动时累加，支撑时清零
            if seq[0] == 0:  # 摆动相
                if self.swing_time[i] < self.swing_period:
                    self.swing_time[i] += dt
            else:  # 支撑相
                self.swing_time[i] = 0.0
        
            if seq[0] == 0:
                lift_off_world = self.foothold_generator.lift_off_positions[leg_name]
                if np.allclose(lift_off_world, 0.0):
                    lift_off_world = leg.foot_pos_world.copy()
                touch_down_world = np.asarray(foothold_targets[leg_name], dtype=np.float64)
                
                # 转换为基座坐标系
                base_pos = current_state.base.pos
                base_rot = current_state.base.rot_mat
                lift_off_pos = base_rot.T @ (lift_off_world - base_pos)
                touch_down = base_rot.T @ (touch_down_world - base_pos)
                
                t_swing = min(self.swing_time[i], self.swing_period)  
                p, v, a = self.swing_generator.get_swing_reference_trajectory(t_swing, lift_off_pos, touch_down)
                
            else:
                # 支撑相：脚保持当前位置
                p = np.asarray(leg.foot_pos, dtype=np.float64).copy()
                v = np.asarray(leg.foot_vel, dtype=np.float64).copy()
                a = np.zeros(3, dtype=np.float64)     
            results[leg_name] = {'pos': p, 'vel': v, 'acc': a}

        return results