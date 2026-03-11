import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SwingTrajectoryGenerator:
    def __init__(self, swing_height: float, swing_duration: float):
        self.swing_height = swing_height
        self.swing_duration = swing_duration
        # 归一化时间缩放因子 (1/T)
        self.factor_duration = 1.0 / self.swing_duration
    
    def _evaluate_bezier_quintic(self, t_rel: float, p0, p1, p2, p3, p4, p5):
        """
        五阶贝塞尔曲线评估函数
        t_rel: 归一化时间 [0, 1]
        p0-p5: 6个控制点 (3D)
        """
        t = t_rel
        t_inv = 1.0 - t
        
        # 1. 位置 (Position)
        pos = (t_inv**5) * p0 + \
              5 * (t_inv**4) * t * p1 + \
              10 * (t_inv**3) * (t**2) * p2 + \
              10 * (t_inv**2) * (t**3) * p3 + \
              5 * t_inv * (t**4) * p4 + \
              (t**5) * p5
              
        # 2. 速度 (Velocity) - 位置对时间t的一阶导数 (P'(t) * dt_rel/dt)
        vel = 5 * ( (t_inv**4) * (p1 - p0) + \
                    4 * (t_inv**3) * t * (p2 - p1) + \
                    6 * (t_inv**2) * (t**2) * (p3 - p2) + \
                    4 * t_inv * (t**3) * (p4 - p3) + \
                    (t**4) * (p5 - p4) ) * self.factor_duration
                    
        # 3. 加速度 (Acceleration) - 位置对时间t的二阶导数 (P''(t) * (dt_rel/dt)^2)
        acc = 20 * ( (t_inv**3) * (p2 - 2*p1 + p0) + \
                     3 * (t_inv**2) * t * (p3 - 2*p2 + p1) + \
                     3 * t_inv * (t**2) * (p4 - 2*p3 + p2) + \
                     (t**3) * (p5 - 2*p4 + p3) ) * (self.factor_duration ** 2)
                     
        return pos.flatten(), vel.flatten(), acc.flatten()

    def get_swing_reference_trajectory(self, t: float, lift_off: np.ndarray, touch_down: np.ndarray):
        t = np.clip(t, 0.0, self.swing_duration)
        t_rel = t * self.factor_duration  
        
        lift_off = lift_off.reshape(3)
        touch_down = touch_down.reshape(3)
        z_peak = max(lift_off[2], touch_down[2]) + self.swing_height
        
        # 定义五阶控制点 (6个)
        # P0, P1 锁死在起点：保证起始速度和加速度为0（或平滑起步）
        p0 = lift_off
        p1 = lift_off 
        
        # P2, P3 负责隆起高度。
        # 注意：贝塞尔曲线不经过控制点，所以控制点要设得比 z_peak 高一些，实际轨迹才能达到 z_peak
        # 这里补偿系数取 1.5 左右比较接近物理期望
        p2 = np.array([lift_off[0], lift_off[1], z_peak + self.swing_height * 0.5])
        p3 = np.array([touch_down[0], touch_down[1], z_peak + self.swing_height * 0.5])
        
        # P4, P5 锁死在终点：保证落地平稳
        p4 = touch_down
        p5 = touch_down

        return self._evaluate_bezier_quintic(t_rel, p0, p1, p2, p3, p4, p5)

    def plot_trajectory(self, lift_off: np.ndarray, touch_down: np.ndarray, 
                       num_samples: int = 100, save_path: str = None):
        t_array = np.linspace(0, self.swing_duration, num_samples)
        positions, velocities, accelerations = [], [], []
        
        for t in t_array:
            pos, vel, acc = self.get_swing_reference_trajectory(t, lift_off, touch_down)
            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)
        
        positions = np.array(positions)
        velocities = np.array(velocities)
        accelerations = np.array(accelerations)
        
        fig = plt.figure(figsize=(15, 10))
        
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
        ax1.scatter(*lift_off, color='green', s=100, label='Lift-off')
        ax1.scatter(*touch_down, color='red', s=100, marker='s', label='Touch-down')
        ax1.set_title('3D Quintic Bezier Trajectory')
        ax1.legend()
        
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(t_array, positions[:, 0], 'r-', label='X')
        ax2.plot(t_array, positions[:, 2], 'b-', label='Z')
        ax2.set_title('Position vs Time')
        ax2.legend(); ax2.grid(True)
        
        # 速度-时间图 (观察平滑度)
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(t_array, velocities[:, 0], 'r-', label='Vx')
        ax3.plot(t_array, velocities[:, 2], 'b-', label='Vz')
        ax3.set_title('Velocity vs Time (C1 Continuous)')
        ax3.legend(); ax3.grid(True)
        
        # 加速度-时间图 (观察重点：中间无阶跃)
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(t_array, accelerations[:, 0], 'r-', label='Ax')
        ax4.plot(t_array, accelerations[:, 2], 'b-', label='Az')
        ax4.set_title('Acceleration vs Time (C2 Continuous)')
        ax4.legend(); ax4.grid(True)
        
        # XY 平面投影
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(positions[:, 0], positions[:, 1], 'b-')
        ax5.set_title('XY Plane Projection')
        ax5.axis('equal'); ax5.grid(True)

        plt.tight_layout()
        if save_path: plt.savefig(save_path)
        else: plt.show()

if __name__ == '__main__':
    swing_h = 0.2
    duration = 0.3
    gen = SwingTrajectoryGenerator(swing_h, duration)
    
    lo = np.array([0.0, 0.0, 0.0])
    td = np.array([0.5, 0.0, 0.0]) 
    print(f"Testing Z: {lo[2]} -> {td[2]} with swing_height: {swing_h}")
    
    t_test = np.linspace(0, duration, 50)
    heights = [gen.get_swing_reference_trajectory(t, lo, td)[0][2] for t in t_test]
    print(f"Max Height Reached: {max(heights):.4f} m")
    
    gen.plot_trajectory(lo, td)