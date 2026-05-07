from .config import WaveTask12Config
from .solver import solve
from .visualization import plot_results
from solver_utils import parse_common_cli, persist_open_figures, plotting_mode


def run(show_plots: bool = True, save_dir: str | None = None) -> None:
    config = WaveTask12Config()
    result = solve(config)
    with plotting_mode(show_plots):
        plot_results(config, result, show=show_plots)
    persist_open_figures(save_dir, "wave_task12")


def cli() -> None:
    show_plots, save_dir = parse_common_cli("Run wave task 12 solver")
    run(show_plots=show_plots, save_dir=save_dir)
