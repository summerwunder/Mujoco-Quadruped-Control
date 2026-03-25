import mujoco, mujoco.viewer, torch, numpy as np
from scipy.spatial.transform import Rotation as R
import time 
# --- 极简配置 ---
SCENE_XML = "/home/mingrui/Documents/Project/mujoco_project/quadruped/quadruped_ctrl/assets/robot/go2_sim2sim/scene.xml"
POLICY_PT = "/home/mingrui/Documents/Project/mujoco_project/quadruped/runs/sim2sim/lin_vel_no_height_scan/exported/policy.pt"

# 你的映射表：Isaac的第0位对应MuJoCo的第3位，以此类推
JOINT_IDS_MAP =  [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
CFG = {
    "dt": 0.02, "sim_dt": 0.002,
    "stiffness": 25.0, "damping": 0.5,
    "action_scale": 0.5,
    "joint_pos_scale": 0.25,
    "joint_vel_scale": 1,
    "ang_vel_scale": 0.2,
    # 这里保持 MuJoCo 的顺序，因为 PD 控制是直接作用于 MuJoCo 关节的
    "default_q_mj": np.array([
        0.1,  0.8, -1.5,  # FL (3, 4, 5 in map -> 1, 4, 9 in default)
        -0.1,  0.8, -1.5,  # FR (0, 1, 2 in map -> 0, 5, 10 in default)
        0.1,  1.0, -1.5,  # RL (9, 10, 11 in map -> 2, 6, 11 in default)
        -0.1,  1.0, -1.5   # RR (6, 7, 8 in map -> 3, 7, 12 in default)
    ])
}


def get_obs(data, last_action, cmd=[1.0, 0.0, 0.0]):
    # 1. 基础状态 (MuJoCo 原始顺序)
    qj_mj = data.qpos[7:19]
    dqj_mj = data.qvel[6:18]
    
    # default_q 也需要对应映射，或者直接使用你给出的那个 12 维 Isaac 顺序列表
    default_q_isaac = CFG["default_q_mj"]
    # --- 关键重排：MuJoCo -> Isaac ---
    # 根据 joint_ids_map 将 qj 和 dqj 映射回训练时的顺序
    qj_isaac = (qj_mj-default_q_isaac)[JOINT_IDS_MAP]
    dqj_isaac = dqj_mj[JOINT_IDS_MAP]


    # 2. 投影重力与角速度 (Body Frame)
    quat_wxyz = data.qpos[3:7] 
    rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rot_mat_inv = rot.as_matrix().T
    
    projected_gravity = rot_mat_inv.T @ [0, 0, -1]
    lin_vel = rot_mat_inv.T @ data.qvel[0:3]
    ang_vel = rot_mat_inv.T @ data.qvel[3:6]
    
    # 3. 拼接观测 (按 Isaac 训练时的顺序)
    obs = np.concatenate([
        lin_vel ,
        ang_vel * CFG["ang_vel_scale"],
        projected_gravity,
        np.array(cmd),                             # cmd 已假定是 [vx, vy, wz]
        qj_isaac  * CFG["joint_pos_scale"],
        dqj_isaac * CFG["joint_vel_scale"],
        last_action                                # 注意：这里的 last_action 也是 Isaac 顺序
    ])
    return torch.from_numpy(obs).float().unsqueeze(0)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def run():


    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)

    time_start = time.time()
    m.opt.timestep = CFG["sim_dt"]
    policy = torch.jit.load(POLICY_PT).eval()
    
    last_action = np.zeros(12) # Isaac 顺序
    decimation = int(CFG["dt"] / CFG["sim_dt"])

    with mujoco.viewer.launch_passive(m, d) as viewer:
        step = 0
        while viewer.is_running():
            if step % decimation == 0:
                with torch.no_grad():
                    alpha = 0.8

                    # 在循环内
                    new_action_isaac = policy(get_obs(d, last_action, cmd=[0.5, 0, 0])).numpy().squeeze()
                    # 线性插值：当前动作 = 之前的动作 * (1-alpha) + 新输出的动作 * alpha
                    action_isaac = (1 - alpha) * last_action + alpha * new_action_isaac       
                
                # --- 关键重排：Isaac Action -> MuJoCo Target ---
                # 将 Isaac 顺序的 action 映射回 MuJoCo 关节索引
                action_mj = np.zeros(12)
                action_mj[JOINT_IDS_MAP] = action_isaac
                
                target_q_mj = action_mj * CFG["action_scale"] + CFG["default_q_mj"]
                last_action = np.clip(new_action_isaac, -100, 100)
                tau = pd_control(target_q_mj, d.qpos[7:19], CFG["stiffness"], np.zeros(12), d.qvel[6:18], CFG["damping"])
            d.ctrl[:] = tau
            mujoco.mj_step(m, d) 
            viewer.sync()
            step += 1
            time_until_next_step = m.opt.timestep - (time.time() - time_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":

    
    run()