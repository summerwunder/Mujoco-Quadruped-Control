import numpy as np

class PeriodicGaitGenerator:
    def __init__(self, duty_factor=0.5, step_freq=1.4, phase_offsets=(0.5, 0.0, 0.0, 0.5)):
        """
        phase_offsets: 长度为 4 的数组 [FL, FR, RL, RR]
        """
        self.duty_factor = duty_factor
        self.step_freq = step_freq
        self.phase_offsets = np.array(phase_offsets)
        
        self.is_full_stance = False
        # 记录前一个步态，用于从 FULL_STANCE 恢复
        self.stored_phase_offsets = np.array(phase_offsets)

    def get_contact_at_time(self, abs_time):
        """核心逻辑：根据绝对时间计算触地状态"""
        if self.is_full_stance:
            return np.ones(4, dtype=np.int8)
            
        # 核心公式：相位 = (时间 * 频率 + 偏移) % 1
        phases = (abs_time * self.step_freq + self.phase_offsets) % 1.0
        return (phases < self.duty_factor).astype(np.int8)

    def get_horizon_sequence(self, abs_time, dt_list, is_full_stance=False):
        """
        为 MPC 计算未来的触地序列
        dt_list: 预测步的时间间隔列表 (支持非均匀采样)
        """
        if self.is_full_stance or is_full_stance:
            return np.ones((4, len(dt_list)), dtype=np.int8)

        # 计算未来的绝对时刻
        future_times = abs_time + np.cumsum(dt_list)
        # 向量化计算所有腿在所有预测点的相位 (4, Horizon)
        phases = (future_times * self.step_freq + self.phase_offsets[:, None]) % 1.0
        return (phases < self.duty_factor).astype(np.int8)

    def update_start_and_stop(self, 
                               base_lin_vel, 
                               base_ang_vel, 
                               ref_lin_vel, 
                               ref_ang_vel, 
                               feet_dist_to_hip_max,
                               base_rpy):
        # 1. 条件检查
        is_command_zero = np.linalg.norm(ref_lin_vel) < 0.01 and np.linalg.norm(ref_ang_vel) < 0.01
        is_robot_static = np.linalg.norm(base_lin_vel) < 0.1 and np.linalg.norm(base_ang_vel) < 0.1
        is_posture_flat = np.abs(base_rpy[0]) < 0.05 and np.abs(base_rpy[1]) < 0.05
        is_feet_home = feet_dist_to_hip_max < 0.06 # 脚踩在臀部附近

        # 2. 状态切换
        if is_command_zero and is_robot_static and is_posture_flat and is_feet_home:
            if not self.is_full_stance:
                self.is_full_stance = True
        elif not is_command_zero:
            # 只要有移动指令，立刻恢复步态
            self.is_full_stance = False

    def set_phase_offsets(self, offsets):
        """手动切换步态，例如从 Trot 切换到 Pace"""
        self.phase_offsets = np.array(offsets)
        self.stored_phase_offsets = np.array(offsets)
    
    # ============ RL-specific methods for gait-aware reward shaping ============
    
    def get_gait_period(self) -> float:
        """获取步态周期
        
        Returns:
            步态周期（秒），= 1 / step_freq
        """
        if self.step_freq <= 0:
            return 1.0
        return 1.0 / self.step_freq
    
    def get_gait_phase(self, current_time: float) -> float:
        """计算当前步态周期内的相位 [0, 1)
        
        Args:
            current_time: 当前绝对时间（秒）
        
        Returns:
            相位值，范围 [0, 1)，其中 0 表示周期开始，1 表示周期结束
        """
        gait_period = self.get_gait_period()
        if gait_period <= 0:
            return 0.0
        # 使用模运算获得周期内的时间位置
        time_in_cycle = current_time % gait_period
        phase = time_in_cycle / gait_period
        return float(phase)
    
    def is_swing_phase(self, leg_idx: int, current_time: float) -> bool:
        """判断指定腿是否处于摆动阶段
        
        Args:
            leg_idx: 腿索引 (0=FL, 1=FR, 2=RL, 3=RR)
            current_time: 当前绝对时间（秒）
        
        Returns:
            True 如果腿在摆动阶段，False 如果在站立阶段
        
        注：摆动阶段 = 不在站立期间，站立期间长度 = duty_factor * gait_period
        """
        if leg_idx < 0 or leg_idx >= 4:
            return False
        
        # 计算该腿相对于周期的相位位置（考虑offset）
        phase_offset = self.phase_offsets[leg_idx]
        current_phase = self.get_gait_phase(current_time)
        leg_phase = (current_phase + phase_offset) % 1.0
        
        # 摆动阶段 = 不在站立期间
        swing_start = self.duty_factor
        is_swing = leg_phase >= swing_start
        
        return is_swing
    
    def get_contact_target(self, leg_idx: int, current_time: float) -> float:
        """根据步态计划获取期望的接触状态
        
        Args:
            leg_idx: 腿索引 (0=FL, 1=FR, 2=RL, 3=RR)
            current_time: 当前绝对时间（秒）
        
        Returns:
            1.0 表示腿应该接触地面（站立阶段）
            0.0 表示腿不应该接触地面（摆动阶段）
        """
        if leg_idx < 0 or leg_idx >= 4:
            return 0.0
        
        # 计算该腿相对于周期的相位位置（考虑offset）
        phase_offset = self.phase_offsets[leg_idx]
        current_phase = self.get_gait_phase(current_time)
        leg_phase = (current_phase + phase_offset) % 1.0
        
        # 接触阶段 = 在站立期间
        should_contact = leg_phase < self.duty_factor
        
        return 1.0 if should_contact else 0.0
        
if __name__ == '__main__':
    # 模拟 Trot 步态：对角腿相位差 0.5
    # FL: 0.5, FR: 0.0, RL: 0.0, RR: 0.5
    pgg = PeriodicGaitGenerator(duty_factor=0.5, step_freq=2.0, phase_offsets=[0.5, 0.0, 0.0, 0.5])
    
    print("--- 场景 1: 绝对时间行走 (t=0.0 到 t=0.2) ---")
    for t in [0.0, 0.3, 0.6]:
        print(f"Time {t:.1f}s | Contact: {pgg.get_contact_at_time(t)}")

    print("\n--- 场景 2: 预测未来序列 (非均匀 dt) ---")
    # 预测未来 5 步，dt 分别为 0.1s
    dt_list = [0.1, 0.1, 0.1, 0.1, 0.1]
    seq = pgg.get_horizon_sequence(0.2, dt_list)
    print(f"未来触地矩阵:\n{seq}")

    print("\n--- 场景 3: 模拟智能停止 ---")
    # 传入静止状态和 0 指令
    pgg.update_start_and_stop(
        base_lin_vel=np.zeros(3), 
        base_ang_vel=np.zeros(3),
        ref_lin_vel=np.zeros(3),
        ref_ang_vel=np.zeros(3),
        feet_dist_to_hip_max=0.02, # 脚已经收回到位
        base_rpy=[0, 0, 0]
    )
    print(f"Time 0.3s | Contact (Static): {pgg.get_contact_at_time(0.3)}")

    print("\n--- 场景 4: 恢复指令 ---")
    # 传入移动指令
    pgg.update_start_and_stop(
        base_lin_vel=np.zeros(3), 
        base_ang_vel=np.zeros(3),
        ref_lin_vel=np.array([1.0, 0, 0]), # 想要前进
        ref_ang_vel=np.zeros(3),
        feet_dist_to_hip_max=0.02,
        base_rpy=[0, 0, 0]
    )
    print(f"Time 0.4s | Contact (Restored): {pgg.get_contact_at_time(0.4)}")