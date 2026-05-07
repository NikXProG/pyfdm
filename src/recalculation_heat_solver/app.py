from .analysis import final_max_point
from .config import RecalculationHeatConfig
from .solver import solve
from .visualization import plot_final_with_max, plot_surface
from solver_utils import parse_common_cli, persist_open_figures, plotting_mode


def run(show_plots: bool = True, save_dir: str | None = None) -> None:
    config = RecalculationHeatConfig()
    result = solve(config)
    x_max, u_max = final_max_point(result)
    print(f"Final max at T=7: x={x_max:.4f}, u={u_max:.6f}")
    with plotting_mode(show_plots):
        plot_final_with_max(result)
        plot_surface(result)
    persist_open_figures(save_dir, "recalculation_heat")


def cli() -> None:
    show_plots, save_dir = parse_common_cli("Run recalculation heat solver")
    run(show_plots=show_plots, save_dir=save_dir)
