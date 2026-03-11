from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def save_mpc_and_velocity_plots(
    sim_time_s: Sequence[float],
    mpc_time_s: Sequence[float],
    mpc_solve_time_ms: Sequence[float],
    vel_time_s: Sequence[float],
    current_vx: Sequence[float],
    reference_vx: Sequence[float],
    output_dir: str | Path,
    prefix: str = "trot_ground",
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sim_time_arr = np.asarray(sim_time_s, dtype=np.float64)
    mpc_time_arr = np.asarray(mpc_time_s, dtype=np.float64)
    mpc_solve_arr = np.asarray(mpc_solve_time_ms, dtype=np.float64)
    vel_time_arr = np.asarray(vel_time_s, dtype=np.float64)
    current_vx_arr = np.asarray(current_vx, dtype=np.float64)
    reference_vx_arr = np.asarray(reference_vx, dtype=np.float64)

    mpc_plot_path = output_path / f"{prefix}_mpc_timing.png"
    vel_plot_path = output_path / f"{prefix}_velocity_tracking.png"

    fig1, ax1 = plt.subplots(figsize=(10, 4.5))
    if mpc_time_arr.size > 0 and mpc_solve_arr.size > 0:
        ax1.plot(mpc_time_arr, mpc_solve_arr, linewidth=1.5, label="MPC solve time")
        ax1.set_xlim(left=max(0.0, float(np.min(mpc_time_arr))))
    ax1.set_title("MPC Solve Time")
    ax1.set_xlabel("Simulation time [s]")
    ax1.set_ylabel("Solve time [ms]")
    ax1.grid(True, alpha=0.3)
    if mpc_solve_arr.size > 0:
        ax1.legend(loc="best")
    fig1.tight_layout()
    fig1.savefig(mpc_plot_path, dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    if vel_time_arr.size > 0 and current_vx_arr.size > 0:
        ax2.plot(vel_time_arr, current_vx_arr, linewidth=1.5, label="Current vx")
    if vel_time_arr.size > 0 and reference_vx_arr.size > 0:
        ax2.plot(vel_time_arr, reference_vx_arr, linewidth=1.5, linestyle="--", label="Reference vx")
        ax2.set_xlim(left=max(0.0, float(np.min(vel_time_arr))))
    ax2.set_title("Velocity Tracking")
    ax2.set_xlabel("Simulation time [s]")
    ax2.set_ylabel("Velocity x [m/s]")
    ax2.grid(True, alpha=0.3)
    if current_vx_arr.size > 0 or reference_vx_arr.size > 0:
        ax2.legend(loc="best")
    fig2.tight_layout()
    fig2.savefig(vel_plot_path, dpi=150)
    plt.close(fig2)

    return mpc_plot_path, vel_plot_path
