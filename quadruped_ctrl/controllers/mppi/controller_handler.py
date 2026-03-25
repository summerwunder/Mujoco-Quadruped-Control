import numpy as np
np.set_printoptions(precision=4, suppress=True)
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

import jax
import jax.numpy as jnp
from jax import random
import sys
import copy
from quadruped_ctrl.datatypes import ReferenceState
from quadruped_ctrl.controllers.mppi.quadruped_model import QuadrupedModel
from quadruped_ctrl.utils.config_loader import ConfigLoader
from quadruped_ctrl.quadruped_env import QuadrupedEnv
DTYPE_GENERAL = "float32"
jax.config.update("jax_default_matmul_precision", "float32")

class MPPI_Controller:
    def __init__(self, env: QuadrupedEnv, mpc_config_path: str = "go2_mpc_config.yaml"):

        self.env = env


        self.state_dim = 24
        self.control_dim = 24
        self.reference_dim = self.state_dim

        self.mpc_config = ConfigLoader.load_mpc_config(mpc_config_path)
        self.horizon = int(self.mpc_config.get('horizon'))

        self.max_sampling_forces_x = 20   # 增大水平力采样范围
        self.max_sampling_forces_y = 20
        self.max_sampling_forces_z = 50   # 增大垂直力采样范围以覆盖更大的GRF范围

        if self.env.device == "gpu":
            try:
                self.device = jax.devices("gpu")[0]
            except:
                self.device = jax.devices("cpu")[0]
                print("GPU not available, using CPU")
        else:
            self.device = jax.devices('cpu')[0]

        if self.env.control_parametrization == "linear_spline":
            # Along the horizon, we have 2 splines per control input (3 forces)
            # Each spline has 2 parameters, but one is shared between the two splines
            self.num_spline = self.env.num_splines
            self.num_control_parameters_single_leg = (self.num_spline + 1) * 3

            # In totale we have 4 legs
            self.num_control_parameters = self.num_control_parameters_single_leg * 4

            # We have 4 different spline functions, one for each leg
            self.spline_fun_FL = self.compute_linear_spline
            self.spline_fun_FR = self.compute_linear_spline
            self.spline_fun_RL = self.compute_linear_spline
            self.spline_fun_RR = self.compute_linear_spline

        elif self.env.control_parametrization == "cubic_spline":
            # Along the horizon, we have 1 splines per control input (3 forces)
            # Each spline has 3 parameters
            self.num_spline = self.env.num_splines
            self.num_control_parameters_single_leg = 4 * 3 * self.num_spline

            # In totale we have 4 legs
            self.num_control_parameters = self.num_control_parameters_single_leg * 4

            # We have 4 different spline functions, one for each leg
            self.spline_fun_FL = self.compute_cubic_spline
            self.spline_fun_FR = self.compute_cubic_spline
            self.spline_fun_RL = self.compute_cubic_spline
            self.spline_fun_RR = self.compute_cubic_spline

        else:
            # We have 1 parameters for every 3 force direction (x,y,z)...for each time horizon!!
            self.num_control_parameters_single_leg = self.horizon * 3

            # In totale we have 4 legs
            self.num_control_parameters = self.num_control_parameters_single_leg * 4

            # We have 4 different spline functions, one for each leg
            self.spline_fun_FL = self.compute_zero_order_spline
            self.spline_fun_FR = self.compute_zero_order_spline
            self.spline_fun_RL = self.compute_zero_order_spline
            self.spline_fun_RR = self.compute_zero_order_spline
        self.num_parallel_computations = self.env.num_parallel_computations
        # MPPI
        self.sigma_mppi = self.env.sigma_mppi
        self.temperature = self.mpc_config.get('temperature', 0.1)  # MPPI温度参数，默认0.1
        self.compute_control = self.compute_control_mppi
        self.shift_solution_enabled = bool(self.mpc_config.get('shift_solution', False))

        self.robot = QuadrupedModel(env)
        self.mu = env.mu
        self.f_z_max = self.mpc_config.get("grf_max")
        self.f_z_min = self.mpc_config.get("grf_min")
        self.Q, self.R = self._set_weight_by_config()
        self.best_control_parameters = jnp.zeros((self.num_control_parameters,), dtype=DTYPE_GENERAL)
        self.master_key = jax.random.PRNGKey(0)
        self.initial_random_parameters = jax.random.uniform(
            key = self.master_key,
            minval = -self.max_sampling_forces_z,
            maxval = self.max_sampling_forces_z,
            shape = (self.num_parallel_computations,self.num_control_parameters),
        )
        self.vectorized_rollout = jax.vmap(self.compute_rollout, in_axes=(None, None, 0, None), out_axes=0)
        self.jit_vectorized_rollout = jax.jit(self.vectorized_rollout, device=self.device)

    def compute_rollout(self, initial_state, reference, control_parameters, contact_sequence):
        """Calculate cost of a rollout of the dynamics given random parameters
        Args:
            initial_state (np.array): actual state of the robot
            reference (np.array): desired state of the robot
            control_parameters (np.array): parameters for the controllers
            parameters (np.array): parameters for the simplified dynamics
        Returns:
            (float): cost of the rollout
        """
        # 确保 contact_sequence 是 JAX 数组
        contact_sequence = jnp.asarray(contact_sequence, dtype=DTYPE_GENERAL)

        state = initial_state
        cost = jnp.float32(0.0)
        n_ = jnp.array([-1, -1, -1, -1])
        prev_input = jnp.zeros(12, dtype=DTYPE_GENERAL)  # 记录上一步控制量，用于平滑项

        FL_num_of_contact = self.horizon
        FR_num_of_contact = self.horizon
        RL_num_of_contact = self.horizon
        RR_num_of_contact = self.horizon

        def iterate_fun(n, carry):
            cost, state, reference, n_, prev_input = carry
            n_ = n_.at[0].set(n)
            n_ = n_.at[1].set(n)
            n_ = n_.at[2].set(n)
            n_ = n_.at[3].set(n)
            f_x_FL, f_y_FL, f_z_FL = self.spline_fun_FL(
                control_parameters[0 : self.num_control_parameters_single_leg], 
                n_[0], 
                FL_num_of_contact
            )
            f_x_FR, f_y_FR, f_z_FR = self.spline_fun_FR(
                control_parameters[self.num_control_parameters_single_leg : self.num_control_parameters_single_leg * 2],
                n_[1],
                FR_num_of_contact,
            )
            f_x_RL, f_y_RL, f_z_RL = self.spline_fun_RL(
                control_parameters[
                    self.num_control_parameters_single_leg * 2 : self.num_control_parameters_single_leg * 3
                ],
                n_[2],
                RL_num_of_contact,
            )
            f_x_RR, f_y_RR, f_z_RR = self.spline_fun_RR(
                control_parameters[
                    self.num_control_parameters_single_leg * 3 : self.num_control_parameters_single_leg * 4
                ],
                n_[3],
                RR_num_of_contact,
            )
            number_of_legs_in_stance = (
                contact_sequence[0][n] + contact_sequence[1][n] + contact_sequence[2][n] + contact_sequence[3][n]
            )
            reference_force_stance_legs = (self.robot.mass * 9.81) / number_of_legs_in_stance

            f_z_FL = reference_force_stance_legs + f_z_FL
            f_z_FR = reference_force_stance_legs + f_z_FR
            f_z_RL = reference_force_stance_legs + f_z_RL
            f_z_RR = reference_force_stance_legs + f_z_RR

            # Foot in swing (contact sequence = 0) have zero force
            f_x_FL = f_x_FL * contact_sequence[0][n] / (self.max_sampling_forces_z/self.max_sampling_forces_x)
            f_y_FL = f_y_FL * contact_sequence[0][n] / (self.max_sampling_forces_z/self.max_sampling_forces_y)
            f_z_FL = f_z_FL * contact_sequence[0][n]

            f_x_FR = f_x_FR * contact_sequence[1][n] / (self.max_sampling_forces_z/self.max_sampling_forces_x)
            f_y_FR = f_y_FR * contact_sequence[1][n] / (self.max_sampling_forces_z/self.max_sampling_forces_y)
            f_z_FR = f_z_FR * contact_sequence[1][n]

            f_x_RL = f_x_RL * contact_sequence[2][n] / (self.max_sampling_forces_z/self.max_sampling_forces_x)
            f_y_RL = f_y_RL * contact_sequence[2][n] / (self.max_sampling_forces_z/self.max_sampling_forces_y)
            f_z_RL = f_z_RL * contact_sequence[2][n]

            f_x_RR = f_x_RR * contact_sequence[3][n] / (self.max_sampling_forces_z/self.max_sampling_forces_x)
            f_y_RR = f_y_RR * contact_sequence[3][n] / (self.max_sampling_forces_z/self.max_sampling_forces_y)
            f_z_RR = f_z_RR * contact_sequence[3][n]
            # Enforce force constraints
            f_x_FL, f_y_FL, f_z_FL, f_x_FR, f_y_FR, f_z_FR, f_x_RL, f_y_RL, f_z_RL, f_x_RR, f_y_RR, f_z_RR = (
                self.enforce_force_constraints(
                    f_x_FL, f_y_FL, f_z_FL, f_x_FR, f_y_FR, f_z_FR, f_x_RL, f_y_RL, f_z_RL, f_x_RR, f_y_RR, f_z_RR
                )
            )
            input = jnp.array(
                [
                    jnp.float32(0),
                    jnp.float32(0),
                    jnp.float32(0),
                    jnp.float32(0),
                    jnp.float32(0),
                    jnp.float32(0),
                    jnp.float32(0),
                    jnp.float32(0),
                    jnp.float32(0),
                    jnp.float32(0),
                    jnp.float32(0),
                    jnp.float32(0),
                    f_x_FL,
                    f_y_FL,
                    f_z_FL,  
                    f_x_FR,
                    f_y_FR,
                    f_z_FR,  
                    f_x_RL,
                    f_y_RL,
                    f_z_RL,  
                    f_x_RR,
                    f_y_RR,
                    f_z_RR,
                ],
                dtype=DTYPE_GENERAL,
            )
            current_contact = jnp.array(
                [contact_sequence[0][n], contact_sequence[1][n], contact_sequence[2][n], contact_sequence[3][n]],
                dtype=DTYPE_GENERAL,
            )
            state_next = self.robot.integrate_jax(state, input, current_contact, n)

            # Calculate cost: state error + control smoothing
            state_error = state_next - reference[0 : self.state_dim]
            error_cost = state_error.T @ self.Q @ state_error
            
            # Control smoothing cost (penalize large changes in GRF)
            current_forces = input[12:24]  # 只取力部分
            force_change = current_forces - prev_input
            control_cost = force_change.T @ self.R[12:24, 12:24] @ force_change
            
            return (cost + error_cost + control_cost, state_next, reference, n_, current_forces)


        carry = (cost, state, reference, n_, prev_input)
        cost, state, reference, n_, _ = jax.lax.fori_loop(0, self.horizon, iterate_fun, carry)

        return cost

    def compute_control_mppi(self,
                             state,
                             reference,
                             contact_sequence,
                             best_control_parameters,
                             key):
        additional_random_parameters = self.initial_random_parameters * 0
        # GAUSSIAN
        num_sample_gaussian_1 = self.num_parallel_computations - 1
        additional_random_parameters = additional_random_parameters.at[1 : self.num_parallel_computations].set(
            self.sigma_mppi * jax.random.normal(key=key, shape=(num_sample_gaussian_1, self.num_control_parameters))
        )
        control_parameters_vec = best_control_parameters + additional_random_parameters
        costs = self.jit_vectorized_rollout(state, reference, control_parameters_vec, contact_sequence)
        costs = jnp.where(jnp.isnan(costs), 1000000, costs)
        costs = jnp.where(jnp.isinf(costs), 1000000, costs)
        best_index = jnp.nanargmin(costs)
        best_cost = costs.take(best_index)

        # compute mppi update
        beta = best_cost 
        temperature = self.temperature  # 从配置读取
        # 数值稳定性：限制指数范围防止溢出
        costs_normalized = jnp.clip(costs - beta, -50.0 / temperature, 50.0 / temperature)
        exp_costs = jnp.exp(-costs_normalized / temperature)
        # 防止权重全零
        exp_costs = jnp.maximum(exp_costs, 1e-10)
        denom = jnp.sum(exp_costs)
        weights = exp_costs / denom

        # 正确的 MPPI 更新：对控制参数加权平均
        weighted_control_parameters = weights[:, None] * control_parameters_vec
        best_control_parameters = jnp.sum(weighted_control_parameters, axis=0)
        # 限制范围
        best_control_parameters = jnp.clip(best_control_parameters, -self.max_sampling_forces_z, self.max_sampling_forces_z)

        # And redistribute it to each leg
        best_control_parameters_FL = best_control_parameters[0 : self.num_control_parameters_single_leg]
        best_control_parameters_FR = best_control_parameters[
            self.num_control_parameters_single_leg : self.num_control_parameters_single_leg * 2
        ]
        best_control_parameters_RL = best_control_parameters[
            self.num_control_parameters_single_leg * 2 : self.num_control_parameters_single_leg * 3
        ]
        best_control_parameters_RR = best_control_parameters[
            self.num_control_parameters_single_leg * 3 : self.num_control_parameters_single_leg * 4
        ]
        fx_FL, fy_FL, fz_FL = self.spline_fun_FL(best_control_parameters_FL, 0.0, 1)
        fx_FR, fy_FR, fz_FR = self.spline_fun_FR(best_control_parameters_FR, 0.0, 1)
        fx_RL, fy_RL, fz_RL = self.spline_fun_RL(best_control_parameters_RL, 0.0, 1)
        fx_RR, fy_RR, fz_RR = self.spline_fun_RR(best_control_parameters_RR, 0.0, 1)

        number_of_legs_in_stance = (
            contact_sequence[0][0] + contact_sequence[1][0] + contact_sequence[2][0] + contact_sequence[3][0]
        )
        reference_force_stance_legs = (self.robot.mass * 9.81) / number_of_legs_in_stance
        fz_FL = reference_force_stance_legs + fz_FL
        fz_FR = reference_force_stance_legs + fz_FR
        fz_RL = reference_force_stance_legs + fz_RL
        fz_RR = reference_force_stance_legs + fz_RR

        fx_FL = fx_FL * contact_sequence[0][0] / (self.max_sampling_forces_z/self.max_sampling_forces_x)
        fy_FL = fy_FL * contact_sequence[0][0] / (self.max_sampling_forces_z/self.max_sampling_forces_y)
        fz_FL = fz_FL * contact_sequence[0][0] 

        fx_FR = fx_FR * contact_sequence[1][0] / (self.max_sampling_forces_z/self.max_sampling_forces_x)
        fy_FR = fy_FR * contact_sequence[1][0] / (self.max_sampling_forces_z/self.max_sampling_forces_y)
        fz_FR = fz_FR * contact_sequence[1][0]

        fx_RL = fx_RL * contact_sequence[2][0] / (self.max_sampling_forces_z/self.max_sampling_forces_x)
        fy_RL = fy_RL * contact_sequence[2][0] / (self.max_sampling_forces_z/self.max_sampling_forces_y)
        fz_RL = fz_RL * contact_sequence[2][0]

        fx_RR = fx_RR * contact_sequence[3][0] / (self.max_sampling_forces_z/self.max_sampling_forces_x)
        fy_RR = fy_RR * contact_sequence[3][0] / (self.max_sampling_forces_z/self.max_sampling_forces_y)
        fz_RR = fz_RR * contact_sequence[3][0]


        # Enforce force constraints
        fx_FL, fy_FL, fz_FL, fx_FR, fy_FR, fz_FR, fx_RL, fy_RL, fz_RL, fx_RR, fy_RR, fz_RR = (
            self.enforce_force_constraints(
                fx_FL, fy_FL, fz_FL, fx_FR, fy_FR, fz_FR, fx_RL, fy_RL, fz_RL, fx_RR, fy_RR, fz_RR
            )
        )

        nmpc_GRFs = jnp.array([fx_FL, fy_FL, fz_FL, fx_FR, fy_FR, fz_FR, fx_RL, fy_RL, fz_RL, fx_RR, fy_RR, fz_RR])
        nmpc_footholds = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Compute predicted state for IK
        input = jnp.array(
            [
                jnp.float32(0),
                jnp.float32(0),
                jnp.float32(0),
                jnp.float32(0),
                jnp.float32(0),
                jnp.float32(0),
                jnp.float32(0),
                jnp.float32(0),
                jnp.float32(0),
                jnp.float32(0),
                jnp.float32(0),
                jnp.float32(0),
                fx_FL,
                fy_FL,
                fz_FL,
                fx_FR,
                fy_FR,
                fz_FR,
                fx_RL,
                fy_RL,
                fz_RL,
                fx_RR,
                fy_RR,
                fz_RR,
            ],
            dtype=DTYPE_GENERAL,
        )
        current_contact = jnp.array(
            [contact_sequence[0][0], contact_sequence[1][0], contact_sequence[2][0], contact_sequence[3][0]],
            dtype=DTYPE_GENERAL,
        )
        nmpc_predicted_state = self.robot.integrate_jax(state, input, current_contact, 0)

        best_freq = 1.4

        return nmpc_GRFs, nmpc_footholds, nmpc_predicted_state, best_control_parameters, best_cost, best_freq, costs



    def enforce_force_constraints(
        self, f_x_FL, f_y_FL, f_z_FL, f_x_FR, f_y_FR, f_z_FR, f_x_RL, f_y_RL, f_z_RL, f_x_RR, f_y_RR, f_z_RR
    ):
        """
        Enforce the friction cone and the force limits constraints
        """

        # Enforce push-only of the ground!
        f_z_FL = jnp.where(f_z_FL > self.f_z_min, f_z_FL, self.f_z_min)
        f_z_FR = jnp.where(f_z_FR > self.f_z_min, f_z_FR, self.f_z_min)
        f_z_RL = jnp.where(f_z_RL > self.f_z_min, f_z_RL, self.f_z_min)
        f_z_RR = jnp.where(f_z_RR > self.f_z_min, f_z_RR, self.f_z_min)

        # Enforce maximum force per leg!
        f_z_FL = jnp.where(f_z_FL < self.f_z_max, f_z_FL, self.f_z_max)
        f_z_FR = jnp.where(f_z_FR < self.f_z_max, f_z_FR, self.f_z_max)
        f_z_RL = jnp.where(f_z_RL < self.f_z_max, f_z_RL, self.f_z_max)
        f_z_RR = jnp.where(f_z_RR < self.f_z_max, f_z_RR, self.f_z_max)

        # Enforce friction cone
        # ( f_{\text{min}} \leq f_z \leq f_{\text{max}} )
        # ( -\mu f_{\text{z}} \leq f_x \leq \mu f_{\text{z}} )
        # ( -\mu f_{\text{z}} \leq f_y \leq \mu f_{\text{z}} )

        f_x_FL = jnp.where(f_x_FL > -self.mu * f_z_FL, f_x_FL, -self.mu * f_z_FL)
        f_x_FL = jnp.where(f_x_FL < self.mu * f_z_FL, f_x_FL, self.mu * f_z_FL)
        f_y_FL = jnp.where(f_y_FL > -self.mu * f_z_FL, f_y_FL, -self.mu * f_z_FL)
        f_y_FL = jnp.where(f_y_FL < self.mu * f_z_FL, f_y_FL, self.mu * f_z_FL)

        f_x_FR = jnp.where(f_x_FR > -self.mu * f_z_FR, f_x_FR, -self.mu * f_z_FR)
        f_x_FR = jnp.where(f_x_FR < self.mu * f_z_FR, f_x_FR, self.mu * f_z_FR)
        f_y_FR = jnp.where(f_y_FR > -self.mu * f_z_FR, f_y_FR, -self.mu * f_z_FR)
        f_y_FR = jnp.where(f_y_FR < self.mu * f_z_FR, f_y_FR, self.mu * f_z_FR)

        f_x_RL = jnp.where(f_x_RL > -self.mu * f_z_RL, f_x_RL, -self.mu * f_z_RL)
        f_x_RL = jnp.where(f_x_RL < self.mu * f_z_RL, f_x_RL, self.mu * f_z_RL)
        f_y_RL = jnp.where(f_y_RL > -self.mu * f_z_RL, f_y_RL, -self.mu * f_z_RL)
        f_y_RL = jnp.where(f_y_RL < self.mu * f_z_RL, f_y_RL, self.mu * f_z_RL)

        f_x_RR = jnp.where(f_x_RR > -self.mu * f_z_RR, f_x_RR, -self.mu * f_z_RR)
        f_x_RR = jnp.where(f_x_RR < self.mu * f_z_RR, f_x_RR, self.mu * f_z_RR)
        f_y_RR = jnp.where(f_y_RR > -self.mu * f_z_RR, f_y_RR, -self.mu * f_z_RR)
        f_y_RR = jnp.where(f_y_RR < self.mu * f_z_RR, f_y_RR, self.mu * f_z_RR)

        return f_x_FL, f_y_FL, f_z_FL, f_x_FR, f_y_FR, f_z_FR, f_x_RL, f_y_RL, f_z_RL, f_x_RR, f_y_RR, f_z_RR
    

    def _set_weight_by_config(self)-> tuple[np.ndarray, np.ndarray]:
        """设置MPC的权重矩阵Q和R
            Q_mat (np.ndarray), R_mat (np.ndarray)
            
            MPPI状态维度: 24
            [com_pos(3), com_vel(3), euler(3), ang_vel(3), foot_pos_FL(3), foot_pos_FR(3), foot_pos_RL(3), foot_pos_RR(3)]
        """

        weights = self.mpc_config.get('weights', {}) if isinstance(self.mpc_config, dict) else {}
        R_conf = self.mpc_config.get('R', {}) if isinstance(self.mpc_config, dict) else {}

        def pick(name, default=None):
            if name in weights:
                return jnp.array(weights[name], dtype=jnp.float32)
            if default is not None:
                return jnp.array(default, dtype=jnp.float32)
            raise KeyError(f"Weight {name} not found in config")

        Q_position = pick('Q_position')
        Q_velocity = pick('Q_velocity')
        Q_base_angle = pick('Q_base_angle')
        Q_base_angle_rates = pick('Q_base_angle_rates')
        Q_foot_pos = pick('Q_foot_pos')

        R_foot_vel = R_conf.get('R_foot_vel', [0.0001, 0.0001, 0.00001])
        R_foot_force = R_conf.get('R_foot_force', [0.1, 0.1, 0.1])
        
        # MPPI状态维度: 24 (不包含积分项)
        # [com_pos(3), com_vel(3), euler(3), ang_vel(3), foot_pos(4*3)]
        Q_list = jnp.concatenate(
            (
                Q_position,           # 3
                Q_velocity,           # 3
                Q_base_angle,         # 3
                Q_base_angle_rates,   # 3
                Q_foot_pos,           # 3
                Q_foot_pos,           # 3
                Q_foot_pos,           # 3
                Q_foot_pos,           # 3
            )
        )

        Q_mat = jnp.diag(Q_list)
        R_list = jnp.concatenate(
            (
                R_foot_vel,
                R_foot_vel,
                R_foot_vel,
                R_foot_vel,
                R_foot_force,
                R_foot_force,
                R_foot_force,
                R_foot_force,
            )
        )
        R_mat = jnp.diag(R_list)
        return Q_mat, R_mat

    def compute_linear_spline(self, parameters, step, horizon_leg):
        """
        Compute the linear spline parametrization of the GRF (N splines)
        """

        # Adding the last boundary for the case when step is exactly self.horizon
        chunk_boundaries = jnp.linspace(0, self.horizon, self.num_spline + 1)
        # Find the chunk index by checking in which interval the step falls
        index = jnp.max(jnp.where(step >= chunk_boundaries, jnp.arange(self.num_spline + 1), 0))
        
        tau = step / (horizon_leg/self.num_spline)
        tau = tau - 1 * index

        q = (tau - 0.0) / (1.0 - 0.0)

        shift = self.num_spline + 1
        f_x = (1 - q) * parameters[index + 0] + q * parameters[index + 1]
        f_y = (1 - q) * parameters[index + shift] + q * parameters[index + shift + 1]
        f_z = (1 - q) * parameters[index + shift * 2] + q * parameters[index + shift * 2 + 1]

        return f_x, f_y, f_z

    def compute_cubic_spline(self, parameters, step, horizon_leg):
        """
        Compute the cubic spline parametrization of the GRF (N splines)
        """

        # Adding the last boundary for the case when step is exactly self.horizon
        chunk_boundaries = jnp.linspace(0, self.horizon, self.num_spline + 1)
        # Find the chunk index by checking in which interval the step falls
        index = jnp.max(jnp.where(step >= chunk_boundaries, jnp.arange(self.num_spline + 1), 0))
        
        tau = step / (horizon_leg/self.num_spline)
        tau = tau - 1*index

        q = (tau - 0.0) / (1.0 - 0.0)

        start_index = 10 * index

        q = (tau - 0.0) / (1.0 - 0.0)
        a = 2 * q * q * q - 3 * q * q + 1
        b = (q * q * q - 2 * q * q + q) * 1.0
        c = -2 * q * q * q + 3 * q * q
        d = (q * q * q - q * q) * 1.0

        phi = (1.0 / 2.0) * (
            ((parameters[start_index + 2] - parameters[start_index + 1]) / 1.0)
            + ((parameters[start_index + 1] - parameters[start_index + 0]) / 1.0)
        )
        phi_next = (1.0 / 2.0) * (
            ((parameters[start_index + 3] - parameters[start_index + 2]) / 1.0)
            + ((parameters[start_index + 2] - parameters[start_index + 1]) / 1.0)
        )
        f_x = a * parameters[start_index + 1] + b * phi + c * parameters[start_index + 2] + d * phi_next

        phi = (1.0 / 2.0) * (
            ((parameters[start_index + 6] - parameters[start_index + 5]) / 1.0)
            + ((parameters[start_index + 5] - parameters[start_index + 4]) / 1.0)
        )
        phi_next = (1.0 / 2.0) * (
            ((parameters[start_index + 7] - parameters[start_index + 6]) / 1.0)
            + ((parameters[start_index + 6] - parameters[start_index + 5]) / 1.0)
        )
        f_y = a * parameters[start_index + 5] + b * phi + c * parameters[start_index + 6] + d * phi_next

        phi = (1.0 / 2.0) * (
            ((parameters[start_index + 10] - parameters[start_index + 9]) / 1.0)
            + ((parameters[start_index + 9] - parameters[start_index + 8]) / 1.0)
        )
        phi_next = (1.0 / 2.0) * (
            ((parameters[start_index + 11] - parameters[start_index + 10]) / 1.0)
            + ((parameters[start_index + 10] - parameters[start_index + 9]) / 1.0)
        )
        f_z = a * parameters[start_index + 9] + b * phi + c * parameters[start_index + 10] + d * phi_next

        return f_x, f_y, f_z

    def compute_zero_order_spline(self, parameters, step, horizon_leg):
        """
        Compute the zero-order control input
        """

        index = jnp.int16(step)
        f_x = parameters[index]
        f_y = parameters[index + self.horizon]
        f_z = parameters[index + self.horizon * 2]
        return f_x, f_y, f_z

    def shift_solution(self, best_control_parameters, step):
        """
        Shift the control parameters ahead for warm-start.
        """

        best_control_parameters = np.array(best_control_parameters)
        FL_control = copy.deepcopy(best_control_parameters[0 : self.num_control_parameters_single_leg])
        FR_control = copy.deepcopy(
            best_control_parameters[self.num_control_parameters_single_leg : self.num_control_parameters_single_leg * 2]
        )
        RL_control = copy.deepcopy(
            best_control_parameters[
                self.num_control_parameters_single_leg * 2 : self.num_control_parameters_single_leg * 3
            ]
        )
        RR_control = copy.deepcopy(
            best_control_parameters[
                self.num_control_parameters_single_leg * 3 : self.num_control_parameters_single_leg * 4
            ]
        )

        FL_control_temp = copy.deepcopy(FL_control)
        FL_control[0], FL_control[2], FL_control[4] = self.spline_fun_FL(FL_control_temp, step, self.horizon)

        FR_control_temp = copy.deepcopy(FR_control)
        FR_control[0], FR_control[2], FR_control[4] = self.spline_fun_FR(FR_control_temp, step, self.horizon)

        RL_control_temp = copy.deepcopy(RL_control)
        RL_control[0], RL_control[2], RL_control[4] = self.spline_fun_RL(RL_control_temp, step, self.horizon)

        RR_control_temp = copy.deepcopy(RR_control)
        RR_control[0], RR_control[2], RR_control[4] = self.spline_fun_RR(RR_control_temp, step, self.horizon)

        best_control_parameters[0 : self.num_control_parameters_single_leg] = FL_control
        best_control_parameters[self.num_control_parameters_single_leg : self.num_control_parameters_single_leg * 2] = (
            FR_control
        )
        best_control_parameters[
            self.num_control_parameters_single_leg * 2 : self.num_control_parameters_single_leg * 3
        ] = RL_control
        best_control_parameters[
            self.num_control_parameters_single_leg * 3 : self.num_control_parameters_single_leg * 4
        ] = RR_control

        return best_control_parameters
    
    def prepare_state_and_reference(self, reference_state:ReferenceState, current_contact, previous_contact):
        # 1. 统一位移步长：如果是向前平移一帧，直接传 1.0
        if self.shift_solution_enabled:
            # 假设样条是按 step 索引定义的
            self.best_control_parameters = self.shift_solution(self.best_control_parameters, 1.0)

        # 2. 构造状态 (尽量保持在 JAX 框架内)
        # 注意：使用质心位置和世界坐标系下的速度，与参考坐标系一致
        # 足端位置使用相对于质心的位置（foot_pos_centered），确保坐标系一致
        state_current = jnp.concatenate([
            self.env.state.base.com,            # 质心位置（世界坐标系）
            self.env.state.base.lin_vel_world,  # 世界坐标系速度
            self.env.state.base.euler,
            self.env.state.base.ang_vel,
            self.env.state.FL.foot_pos_world - self.env.state.base.com,  # 足端相对于质心的位置
            self.env.state.FR.foot_pos_world - self.env.state.base.com,
            self.env.state.RL.foot_pos_world - self.env.state.base.com,
            self.env.state.RR.foot_pos_world - self.env.state.base.com
        ])

        # 3. 摆动腿参考替换 (向量化操作更优雅)
        # 假设每条腿 3 维，从索引 12 开始
        # 注意：ref_foot_* 是世界坐标系，需要转换为相对于质心的位置
        com = self.env.state.base.com
        for i in range(4):
            if current_contact[i] == 0:
                start_idx = 12 + i * 3
                ref_foot = getattr(reference_state, f'ref_foot_{["FL","FR","RL","RR"][i]}')
                # 转换为相对于质心的位置
                ref_foot_relative = ref_foot.flatten() - com
                state_current = state_current.at[start_idx : start_idx+3].set(ref_foot_relative)

        # 4. 状态切换重置 (使用 JAX 风格)
        for i in range(4):
            if previous_contact[i] == 1 and current_contact[i] == 0:
                start = i * self.num_control_parameters_single_leg
                end = (i + 1) * self.num_control_parameters_single_leg
                self.best_control_parameters = self.best_control_parameters.at[start:end].set(0.0)

        # 构造参考（足端位置转换为相对于质心的位置）
        # 注意：ref_position 是相对于当前位置的参考高度，不是绝对位置
        # 所以参考状态的位置应该是当前质心位置 + 参考高度偏移
        ref_pos = np.array([
            com[0],  # x 保持当前位置
            com[1],  # y 保持当前位置
            reference_state.ref_position[2]  # z 使用参考高度
        ])
        reference_jax = np.concatenate([
            ref_pos,
            reference_state.ref_linear_velocity,
            reference_state.ref_orientation,
            reference_state.ref_angular_velocity,
            reference_state.ref_foot_FL.reshape((3,)) - com,
            reference_state.ref_foot_FR.reshape((3,)) - com,
            reference_state.ref_foot_RL.reshape((3,)) - com,
            reference_state.ref_foot_RR.reshape((3,)) - com
        ]).reshape((24,))

        return state_current, reference_jax
    
    def with_newkey(self):
        """生成新的随机key"""
        self.master_key, subkey = jax.random.split(self.master_key)
        return self
