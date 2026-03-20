import time 
import jax 
from jax import numpy as jnp
from jax import jit, random
from quadruped_ctrl.quadruped_env import QuadrupedEnv

DTYPE_GENERAL = 'float32'
jax.config.update("jax_default_matmul_precision", "float32")

class QuadrupedModel:
    def __init__(self, env: QuadrupedEnv) -> None:
        self.dt = env.dt
        self.env = env

        if env.device == "gpu":
            self.device = jax.devices("gpu")[0]
        elif env.device == "cpu":
            self.device = jax.devices("cpu")[0]
        
        self.mass = env.robot.mass
        self.inertia = env.robot.get_inertia_matrix()
        self.inertia_inv = jnp.linalg.inv(self.inertia)

        self.state_dim = 12
        self.full_state_dim = 24
        self.input_dim = 24

        vectorized_integrate_jax = jax.vmap(self.integrate_jax, in_axes=(None, 0, None), out_axes=0)
        self.compiled_integrate_jax = jit(vectorized_integrate_jax, device=self.device)

    
    def integrate_jax(self, state, input, contact_status, step_idx=None):
        """积分一步动力学
        
        Args:
            state: 状态向量 (24,)
            input: 输入向量 (24,)
            contact_status: 接触状态 (4,)
            step_idx: 时间步索引（可选，用于非均匀时间步长）
        """
        fd = self.fd(state, input, contact_status)
        new_state = state[0:12] + fd * self.dt
        return jnp.concatenate([new_state, state[12:]])
    
    def fd(self, state, input, contact_status):
        state = jnp.asarray(state, dtype=DTYPE_GENERAL)
        input = jnp.asarray(input, dtype=DTYPE_GENERAL)
        contact_status = jnp.asarray(contact_status, dtype=DTYPE_GENERAL)

        if state.shape[0] < self.full_state_dim:
            raise ValueError(f"state size mismatch: expected >= {self.full_state_dim}, got {state.shape[0]}")
        if input.shape[0] < self.input_dim:
            raise ValueError(f"input size mismatch: expected >= {self.input_dim}, got {input.shape[0]}")
        if contact_status.shape[0] < 4:
            raise ValueError(f"contact_status size mismatch: expected >= 4, got {contact_status.shape[0]}")

        def skew(v):
            return jnp.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
        
        foot_position_fl, foot_position_fr, foot_position_rl, foot_position_rr = jnp.split(state[12:24], 4)
        foot_force_fl, foot_force_fr, foot_force_rl, foot_force_rr = jnp.split(input[12:24], 4)
        com_position = state[:3]
        stanceFL, stanceFR, stanceRL, stanceRR = contact_status[:4]
        com_velocity = state[3:6]
        temp = jnp.dot(foot_force_fl, stanceFL) + \
                jnp.dot(foot_force_fr, stanceFR) + \
                jnp.dot(foot_force_rl, stanceRL) + \
                jnp.dot(foot_force_rr, stanceRR)
        
        gravity = jnp.array([jnp.float32(0), jnp.float32(0), jnp.float32(-self.env.robot.gravity)])
        linear_com_acc = jnp.dot(1/self.mass, temp) + gravity

        w = state[9 :12]
        roll,pitch,yaw = state[6:9]
        conj_euler_rates = jnp.array([
            [1, 0, -jnp.sin(pitch)],
            [0, jnp.cos(roll), jnp.cos(pitch) * jnp.sin(roll)],
            [0, -jnp.sin(roll), jnp.cos(pitch) * jnp.cos(roll)]
        ])

        temp2 = jnp.dot(skew(foot_position_fl - com_position), foot_force_fl) * stanceFL + \
                jnp.dot(skew(foot_position_fr - com_position), foot_force_fr) * stanceFR + \
                jnp.dot(skew(foot_position_rl - com_position), foot_force_rl) * stanceRL + \
                jnp.dot(skew(foot_position_rr - com_position), foot_force_rr) * stanceRR
        euler_rates_base = jnp.linalg.inv(conj_euler_rates) @ w

        b_R_w = jnp.array(
            [
                [jnp.cos(pitch) * jnp.cos(yaw), jnp.cos(pitch) * jnp.sin(yaw), -jnp.sin(pitch)],
                [
                    jnp.sin(roll) * jnp.sin(pitch) * jnp.cos(yaw) - jnp.cos(roll) * jnp.sin(yaw),
                    jnp.sin(roll) * jnp.sin(pitch) * jnp.sin(yaw) + jnp.cos(roll) * jnp.cos(yaw),
                    jnp.sin(roll) * jnp.cos(pitch),
                ],
                [
                    jnp.cos(roll) * jnp.sin(pitch) * jnp.cos(yaw) + jnp.sin(roll) * jnp.sin(yaw),
                    jnp.cos(roll) * jnp.sin(pitch) * jnp.sin(yaw) - jnp.sin(roll) * jnp.cos(yaw),
                    jnp.cos(roll) * jnp.cos(pitch),
                ],
            ]
        )

        angular_acc_base = -jnp.dot(self.inertia_inv, jnp.dot(skew(w), jnp.dot(self.inertia, w))) + jnp.dot(
            self.inertia_inv, jnp.dot(b_R_w, temp2)
        )


        return jnp.concatenate([com_velocity, linear_com_acc, euler_rates_base, angular_acc_base])

if __name__ == "__main__":
    env = QuadrupedEnv(robot_config='robot/go2.yaml',
                       model_path='quadruped_ctrl/assets/robot/go2/scene.xml',
                       sim_config_path='sim_config.yaml',)
    model_jax = QuadrupedModel(env = env)

    # all stance, friction, stance proximity
    param = jnp.array([1.0, 1.0, 1.0, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0], dtype=DTYPE_GENERAL)

    state = jnp.array(
        [
            0.0,
            0.0,
            0.0,  # com position
            0.0,
            0.0,
            0.0,  # com velocity
            0.0,
            0.0,
            0.0,  # euler angles
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # foot position fl
            0.0,
            0.0,
            0.0,  # foot position fr
            0.0,
            0.0,
            0.0,  # foot position rl
            0.0,
            0.0,
            0.0,  # foot position rr
        ],
        dtype=DTYPE_GENERAL,
    )  # [base state(12), foot positions(12)]

    input = jnp.array(
        [
            0.0,
            0.0,
            0.0,  # foot position fl
            0.0,
            0.0,
            0.0,  # foot position fr
            0.0,
            0.0,
            0.0,  # foot position rl
            0.0,
            0.0,
            0.0,  # foot position rr
            0.0,
            0.0,
            0.0,  # foot force fl
            0.0,
            0.0,
            0.0,  # foot force fr
            0.0,
            0.0,
            0.0,  # foot force rl
            0.0,
            0.0,
            0.0,
        ],
        dtype=DTYPE_GENERAL,
    )  # foot force rr

    # test fd
    acc = model_jax.fd(state, input, param)

    # test integrated
    print("testing SINGLE integration PYTHON")
    time_start = time.time()
    state_integrated = model_jax.integrate_jax(state, input, param)
    print("computation time: ", time.time() - time_start)

    # test compiled integrated
    print("\ntesting SINGLE integration COMPILED-FIRST TIME")
    compiled_integrated_jax_single = jit(model_jax.integrate_jax, device=model_jax.device)
    time_start = time.time()
    state_integrated = compiled_integrated_jax_single(state, input, param).block_until_ready()
    print("computation time: ", time.time() - time_start)

    print("\ntesting SINGLE integration COMPILED-SECOND TIME")
    time_start = time.time()
    state_integrated = compiled_integrated_jax_single(state, input, param).block_until_ready()
    print("computation time: ", time.time() - time_start)

    print("\ntesting MULTIPLE integration COMPILED-FIRST TIME")
    threads = 10000

    key = random.PRNGKey(42)
    input_vec = random.randint(key, (model_jax.input_dim * threads,), minval=-2, maxval=2).reshape(
        threads, model_jax.input_dim
    )
