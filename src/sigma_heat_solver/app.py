from .analysis import print_summary
from .config import SigmaHeatConfig
from .solver import solve_all
from .visualization import plot_final_curves_with_maxima, plot_surface_for_run
from solver_utils import parse_common_cli, persist_open_figures, plotting_mode


def run(show_plots: bool = True, save_dir: str | None = None) -> None:
    config = SigmaHeatConfig()
    runs = solve_all(config)
    print_summary(runs)
    with plotting_mode(show_plots):
        plot_final_curves_with_maxima(runs)
        representative = next(r for r in runs if abs(r.sigma - 0.5) < 1e-12)
        plot_surface_for_run(representative)
    persist_open_figures(save_dir, "sigma_heat")


def cli() -> None:
    show_plots, save_dir = parse_common_cli("Run sigma heat solver")
    run(show_plots=show_plots, save_dir=save_dir)
