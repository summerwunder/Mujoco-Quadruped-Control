"""
逆运动学求解器 - Numeric方法
"""
import copy
import numpy as np
import mujoco
from typing import Dict, Optional
from quadruped_ctrl.quadruped_env import QuadrupedEnv 

class InverseKinematics:
    def __init__(self, 
                 env: 'QuadrupedEnv',
                 ik_iterations: Optional[int] = None,
                 ik_dt: Optional[float] = None,
                 damping: Optional[float] = None):
        """
        初始化逆运动学求解器
        
        Args:
            env: QuadrupedEnv实例
            ik_iterations: IK迭代次数，若为None则从sim_config读取
            ik_dt: IK数值积分步长，若为None则从sim_config读取
            damping: IK阻尼系数，若为None则从sim_config读取
        """
        self.env = env
        self.model = copy.deepcopy(env.model)
        self.ik_data = mujoco.MjData(self.model)
        
        # 从sim_config读取IK参数
        ik_config = env.sim_config.get('ik_solver', {})
        self.max_iterations = ik_iterations or ik_config.get('ik_iterations', 20)
        self.dt = ik_dt or ik_config.get('ik_dt', 0.01)
        self.damping = damping or ik_config.get('ik_damping', 1e-3)
        self.damp_matrix = self.damping * np.eye(12)

        self.site_names = ['FL', 'FR', 'RL', 'RR']
        self.site_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name) for name in self.site_names]
    def compute_ik(self,
                   target_feet_pos: Dict[str, np.ndarray]
                   ) -> np.ndarray:
        """
        在影子空间求解逆运动学
        
        Args:
            target_feet_pos: 目标足端位置字典 {'FL': (3,), 'FR': (3,), ...}
        
        Returns:
            求解后的关节角度 (12,)
        """
        # 1. 初始化影子 Data 的状态为当前主程序的状态
        self.ik_data.qpos[:] = self.env.data.qpos.copy()
        
        for iteration in range(self.max_iterations):
            # 2. 前向运动学更新 (在影子 Data 上)
            mujoco.mj_fwdPosition(self.model, self.ik_data)

            errors = []
            for i, s_id in enumerate(self.site_ids):
                current_pos = self.ik_data.site_xpos[s_id]
                leg_name = self.site_names[i]
                error = target_feet_pos[leg_name] - current_pos
                errors.append(error)
            
            total_error = np.hstack(errors)
            error_norm = np.linalg.norm(total_error)
            
            if error_norm < 1e-4: break
            
            # 4. 计算雅可比矩阵
            jac_list = []
            for s_id in self.site_ids:
                jac_p = np.zeros((3, self.model.nv))
                jac_r = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(
                    self.model, 
                    self.ik_data, 
                    jac_p, 
                    jac_r, 
                    s_id)
                jac_list.append(jac_p[:, 6:]) 
            
            total_jac = np.vstack(jac_list) # (12, 12)
            
            # dq = (J^T J + λI)^-1 J^T * error
            damped_pinv = np.linalg.inv(total_jac.T @ total_jac + self.damp_matrix) @ total_jac.T
            dq = damped_pinv @ (total_error) 
            
            # TODO: 7:19不应该写死的
            self.ik_data.qpos[7:19] += dq * self.dt

        return self.ik_data.qpos.copy()

if __name__ == "__main__":
    from quadruped_ctrl.quadruped_env import QuadrupedEnv
    import time
    
    env = QuadrupedEnv()
    env.reset()
    
    ik_solver = InverseKinematics(env)
    
    target_pos = {
        'FL': env.state.FL.foot_pos_world.copy() + np.array([0, 0, 0.5]), # 向上抬5厘米
        'FR': env.state.FR.foot_pos_world.copy(),
        'RL': env.state.RL.foot_pos_world.copy(),
        'RR': env.state.RR.foot_pos_world.copy(),
    }
    
    # 求解
    start_time = time.time()
    joint_solution = ik_solver.compute_ik(target_pos)
    elapsed = time.time() - start_time
    print(f"求解耗时: {elapsed:.6f}s")
    print(f"求解的关节角度[7:19]: {joint_solution[7:19]}")
    
    ik_solver.ik_data.qpos[:] = joint_solution
    mujoco.mj_fwdPosition(ik_solver.model, ik_solver.ik_data)
 
    with mujoco.viewer.launch_passive(ik_solver.model, ik_solver.ik_data) as viewer:
        while True:
            viewer.sync()