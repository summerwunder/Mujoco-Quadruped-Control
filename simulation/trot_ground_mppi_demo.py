"""
MPPI控制器平地行走测试
使用Model Predictive Path Integral Control进行四足机器人trot步态控制
"""

from pathlib import Path
import time
import numpy as np
import mujoco
import jax
import jax.numpy as jnp

from quadruped_ctrl.quadruped_env import QuadrupedEnv
from quadruped_ctrl.controllers.mppi.controller_handler import MPPI_Controller
from quadruped_ctrl.interface.reference_interface import ReferenceInterface
from quadruped_ctrl.interface.wb_interface import WBInterface
from quadruped_ctrl.utils.plot_utils import save_mpc_and_velocity_plots


def main() -> None:
    # ========== 初始化环境 ==========
    env = QuadrupedEnv(
        robot_config='robot/go2.yaml',
        model_path='quadruped_ctrl/assets/robot/go2/scene.xml',
        sim_config_path='sim_config.yaml'
    )
    mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
    obs, _ = env.reset()

    # ========== 初始化MPPI控制器 ==========
    mpc_config_path = "go2_mppi_config.yaml"
    mppi_controller = MPPI_Controller(env, mpc_config_path=mpc_config_path)
    
    # 初始化参考接口和Whole-Body接口
    ref_interface = ReferenceInterface(env, mpc_config_path=mpc_config_path)
    wb_interface = WBInterface(env)

    # ========== 控制参数 ==========
    mpc_decimation = int(env.sim_config.get("physics", {}).get("mpc_frequency", 10))
    mpc_decimation = max(1, mpc_decimation)
    
    last_action = np.zeros(env.model.nu, dtype=np.float64)
    last_optimal_footholds = {
        "FL": np.zeros(3, dtype=np.float64),
        "FR": np.zeros(3, dtype=np.float64),
        "RL": np.zeros(3, dtype=np.float64),
        "RR": np.zeros(3, dtype=np.float64),
    }
    
    # 参考速度：前进0.5 m/s
    ref_lin_vel = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    ref_ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # ========== 数据记录 ==========
    mpc_time_s = []
    mpc_solve_time_ms = []
    vel_time_s = []
    current_vx = []
    reference_vx = []

    # ========== MPPI状态 ==========
    previous_contact = np.ones(4, dtype=np.float64)  # 初始假设四腿支撑

    step = 0
    last_render_time = 0
    
    print("=" * 60)
    print("MPPI控制器平地行走测试")
    print(f"Horizon: {mppi_controller.horizon}")
    print(f"Parallel samples: {mppi_controller.num_parallel_computations}")
    print(f"Sigma MPPI: {mppi_controller.sigma_mppi}")
    print(f"Control parametrization: {env.control_parametrization}")
    print(f"Reference velocity: {ref_lin_vel[0]:.2f} m/s")
    print("=" * 60)
    
    try:
        while True:
            step_start = time.time()
            state = env.get_state()
            com_pos = state.base.com.copy()

            env.ref_base_lin_vel = ref_lin_vel
            env.ref_base_ang_vel = ref_ang_vel

            # ========== 获取参考状态和接触序列 ==========
            reference_state, contact_sequence, swing_refs = ref_interface.get_reference_state(
                current_state=state,
                com_pos=com_pos,
                heightmaps=None,
                abs_time=env.dt * env.state.step_num,
                ref_base_lin_vel=ref_lin_vel,
                ref_base_ang_vel=ref_ang_vel,
            )

            # ========== MPPI求解 ==========
            if step % round(1 / (mpc_decimation * env.dt)) == 0:
                mpc_start = time.time()
                
                # 当前接触状态
                current_contact = contact_sequence[:, 0].astype(np.float64)
                
                # 准备状态和参考
                state_jax, ref_jax = mppi_controller.prepare_state_and_reference(
                    reference_state=reference_state,
                    current_contact=current_contact,
                    previous_contact=previous_contact,
                )
                
                # 更新随机key
                mppi_controller = mppi_controller.with_newkey()
                
                # MPPI计算
                optimal_GRF, optimal_footholds, predicted_state, best_control_parameters, best_cost, best_freq, costs = \
                    mppi_controller.compute_control_mppi(
                        state=jnp.array(state_jax, dtype=jnp.float32),
                        reference=jnp.array(ref_jax, dtype=jnp.float32),
                        contact_sequence=contact_sequence,
                        best_control_parameters=mppi_controller.best_control_parameters,
                        key=mppi_controller.master_key
                    )
                
                # 更新控制器状态
                mppi_controller.best_control_parameters = best_control_parameters
                previous_contact = current_contact.copy()
                
                # 转换为numpy
                optimal_GRF = np.array(optimal_GRF, dtype=np.float64)
                
                mpc_elapsed_ms = (time.time() - mpc_start) * 1000.0
                mpc_time_s.append(float(env.data.time))
                mpc_solve_time_ms.append(float(mpc_elapsed_ms))

                # 更新落脚点
                last_optimal_footholds = {
                    "FL": np.zeros(3, dtype=np.float64),
                    "FR": np.zeros(3, dtype=np.float64),
                    "RL": np.zeros(3, dtype=np.float64),
                    "RR": np.zeros(3, dtype=np.float64),
                }
                
                # 更新接触力（用于可视化）
                for i, leg_name in enumerate(['FL', 'FR', 'RL', 'RR']):
                    leg = state.get_leg_by_name(leg_name)
                    leg.contact_force = optimal_GRF[i * 3:(i + 1) * 3].copy()

                # ========== 计算关节力矩 ==========
                tau_total, des_joints_pos, des_joints_vel = wb_interface.compute_tau(
                    state,
                    swing_targets=swing_refs,
                    contact_sequence=contact_sequence[:, 0],
                    optimal_GRF=optimal_GRF
                )
                last_action = tau_total

            # ========== 执行仿真步 ==========
            env.step(last_action)

            # 记录速度数据
            vel_time_s.append(float(env.data.time))
            current_vx.append(float(env.data.qvel[0]))
            reference_vx.append(float(ref_lin_vel[0]))

            # ========== 可视化 ==========
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

            # ========== 打印状态 ==========
            if step % 300 == 0:
                num_contact = sum([state.FL.contact_state, state.FR.contact_state,
                                   state.RL.contact_state, state.RR.contact_state])
                base_height = state.base.pos[2]
                current_vel = state.base.lin_vel_world[0]
                print(f"步数: {step:5d} | 时间: {env.data.time:6.2f}s | "
                      f"支撑腿数: {num_contact} | 身体高度: {base_height:.3f}m | "
                      f"速度: {current_vel:.2f}m/s | MPC耗时: {mpc_solve_time_ms[-1] if mpc_solve_time_ms else 0:.1f}ms")

            step += 1
            
            # 实时控制
            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        # ========== 保存结果 ==========
        out_dir = Path("runs") / "mppi_plots"
        mpc_plot_path, vel_plot_path = save_mpc_and_velocity_plots(
            sim_time_s=vel_time_s,
            mpc_time_s=mpc_time_s,
            mpc_solve_time_ms=mpc_solve_time_ms,
            vel_time_s=vel_time_s,
            current_vx=current_vx,
            reference_vx=reference_vx,
            output_dir=out_dir,
            prefix="trot_ground_mppi",
        )
        print(f"已保存MPC耗时图: {mpc_plot_path}")
        print(f"已保存速度跟踪图: {vel_plot_path}")
        
        # 关闭环境
        env.close()


if __name__ == "__main__":
    main()