from .config import SimulationConfig
from .solver import solve
from .visualization import plot_results
from solver_utils import parse_common_cli, persist_open_figures, plotting_mode


def run(show_plots: bool = True, save_dir: str | None = None) -> None:
    config = SimulationConfig()
    result = solve(config)
    with plotting_mode(show_plots):
        plot_results(config, result, show=show_plots)
    persist_open_figures(save_dir, "cauchy")


def cli() -> None:
    show_plots, save_dir = parse_common_cli("Run Cauchy solver")
    run(show_plots=show_plots, save_dir=save_dir)
