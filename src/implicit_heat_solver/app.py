from .config import ImplicitHeatConfig
from .solver import solve
from .visualization import plot_snapshots_with_maxima, plot_surface
from solver_utils import parse_common_cli, persist_open_figures, plotting_mode


def run(show_plots: bool = True, save_dir: str | None = None) -> None:
    config = ImplicitHeatConfig()
    result = solve(config)
    with plotting_mode(show_plots):
        plot_snapshots_with_maxima(result)
        plot_surface(result)
    persist_open_figures(save_dir, "implicit_heat")


def cli() -> None:
    show_plots, save_dir = parse_common_cli("Run implicit heat solver")
    run(show_plots=show_plots, save_dir=save_dir)
