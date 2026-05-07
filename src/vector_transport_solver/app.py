from .config import VectorTransportConfig
from .solver import solve
from .visualization import plot_3d_surfaces, plot_dynamic_frames
from solver_utils import parse_common_cli, persist_open_figures, plotting_mode


def run(show_plots: bool = True, save_dir: str | None = None) -> None:
    config = VectorTransportConfig()
    result = solve(config)
    with plotting_mode(show_plots):
        plot_dynamic_frames(result, stride=config.animation_stride, show=show_plots)
        plot_3d_surfaces(result, show=show_plots)
    persist_open_figures(save_dir, "vector_transport")


def cli() -> None:
    show_plots, save_dir = parse_common_cli("Run vector transport solver")
    run(show_plots=show_plots, save_dir=save_dir)
