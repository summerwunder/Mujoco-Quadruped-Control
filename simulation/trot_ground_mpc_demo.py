from pathlib import Path
import time
import numpy as np
import mujoco

from quadruped_ctrl.quadruped_env import QuadrupedEnv
from quadruped_ctrl.controllers.controller_factory import ControllerFactory
from quadruped_ctrl.interface.reference_interface import ReferenceInterface
from quadruped_ctrl.interface.wb_interface import WBInterface
from quadruped_ctrl.utils.plot_utils import save_mpc_and_velocity_plots
'''
full stance 站立测试支撑腿 + 可视化参考点、接触力等
'''

def main() -> None:
    env = QuadrupedEnv(
        robot_config='robot/go1.yaml',
        model_path='quadruped_ctrl/assets/robot/go1/scene.xml',
        sim_config_path='sim_config.yaml'
    )
    mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
    obs, _ = env.reset()
    
    mpc_config_path = "go1_mpc_config.yaml"
    mpc_controller = ControllerFactory.create_controller("mpc_gradient", env, mpc_config_path=mpc_config_path)
    ref_interface = ReferenceInterface(env, mpc_config_path=mpc_config_path)
    wb_interface = WBInterface(env)

    mpc_decimation = int(env.sim_config.get("physics", {}).get("mpc_frequency", 10))
    mpc_decimation = max(1, mpc_decimation)
    last_action = np.zeros(env.model.nu, dtype=np.float64)
    last_optimal_footholds = {
        "FL": np.zeros(3, dtype=np.float64),
        "FR": np.zeros(3, dtype=np.float64),
        "RL": np.zeros(3, dtype=np.float64),
        "RR": np.zeros(3, dtype=np.float64),
    }
    ref_lin_vel = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ref_ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    mpc_time_s = []
    mpc_solve_time_ms = []
    vel_time_s = []
    current_vx = []
    reference_vx = []

    step = 0
    last_render_time = 0
    try:
        while True:
            step_start = time.time()
            state = env.get_state()
            com_pos = state.base.com.copy()

            env.ref_base_lin_vel = ref_lin_vel
            env.ref_base_ang_vel = ref_ang_vel

            reference_state, contact_sequence, swing_refs = ref_interface.get_reference_state(
                current_state=state,
                com_pos=com_pos,
                heightmaps=None,
                abs_time=env.dt * env.state.step_num,
                ref_base_lin_vel=ref_lin_vel,
                ref_base_ang_vel=ref_ang_vel,
            )

            if step % round(1 / (mpc_decimation * env.dt)) == 0:
                mpc_start = time.time()
                optimal_GRF, optimal_footholds, optimal_next_state, status = mpc_controller.get_action(
                    state=state,
                    reference=reference_state,
                    contact_sequence=contact_sequence,
                    mass=env.robot.mass,
                    inertia=env.robot.inertia,
                    mu=env.mu
                )
                mpc_elapsed_ms = (time.time() - mpc_start) * 1000.0
                mpc_time_s.append(float(env.data.time))
                mpc_solve_time_ms.append(float(mpc_elapsed_ms))

                last_optimal_footholds = {
                    "FL": optimal_footholds[0].copy(),
                    "FR": optimal_footholds[1].copy(),
                    "RL": optimal_footholds[2].copy(),
                    "RR": optimal_footholds[3].copy(),
                }
                for i, leg_name in enumerate(['FL', 'FR', 'RL', 'RR']):
                    leg = state.get_leg_by_name(leg_name)
                    leg.contact_force = optimal_GRF[i * 3:(i + 1) * 3].copy()

                tau_total, des_joints_pos, des_joints_vel = wb_interface.compute_tau(
                    state,
                    swing_targets=swing_refs,
                    contact_sequence=contact_sequence[:, 0],
                    optimal_GRF=optimal_GRF
                )
                last_action = tau_total

            env.step(last_action)

            vel_time_s.append(float(env.data.time))
            current_vx.append(float(env.data.qvel[0]))
            reference_vx.append(float(ref_lin_vel[0]))

            swing_vis = {
                "swing_generator": ref_interface.swing_generator,
                "swing_period": ref_interface.swing_period,
                "swing_time": {
                    "FL": ref_interface.swing_time[0],
                    "FR": ref_interface.swing_time[1],
                    "RL": ref_interface.swing_time[2],
                    "RR": ref_interface.swing_time[3],
                },
                "lift_off_positions": ref_interface.foothold_generator.lift_off_positions,
                "nmpc_footholds": last_optimal_footholds,
                "ref_feet_pos": {
                    "FL": reference_state.ref_foot_FL.copy(),
                    "FR": reference_state.ref_foot_FR.copy(),
                    "RL": reference_state.ref_foot_RL.copy(),
                    "RR": reference_state.ref_foot_RR.copy(),
                },
            }
            if time.time() - last_render_time > 1.0 / env.sim_config.get("render", {}).get("render_frequency", 30):
                env.render(swing_vis=swing_vis)
                last_render_time = time.time()
                if env.viewer is not None and not env.viewer.is_running():
                    break

            if step % 300 == 0:
                num_contact = sum([state.FL.contact_state, state.FR.contact_state,
                                   state.RL.contact_state, state.RR.contact_state])
                base_height = state.base.pos[2]
                print(f"步数: {step:5d} | 时间: {env.data.time:6.2f}s | "
                      f"支撑腿数: {num_contact} | 身体高度: {base_height:.3f}m")

            step += 1
            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    finally:
        out_dir = Path("runs") / "mpc_plots"
        mpc_plot_path, vel_plot_path = save_mpc_and_velocity_plots(
            sim_time_s=vel_time_s,
            mpc_time_s=mpc_time_s,
            mpc_solve_time_ms=mpc_solve_time_ms,
            vel_time_s=vel_time_s,
            current_vx=current_vx,
            reference_vx=reference_vx,
            output_dir=out_dir,
            prefix="trot_ground",
        )
        print(f"已保存MPC耗时图: {mpc_plot_path}")
        print(f"已保存速度跟踪图: {vel_plot_path}")


if __name__ == "__main__":
    main()

