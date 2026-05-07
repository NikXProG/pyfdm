from .analysis import print_results
from .config import LWConfig
from .derivation import approximation_order_summary
from .solver import run_all
from .visualization import plot_sims
from .von_neumann import stability_summary
from solver_utils import parse_common_cli, persist_open_figures, plotting_mode


def run(show_plots: bool = True, save_dir: str | None = None) -> None:
    print(approximation_order_summary())
    print()
    print(stability_summary())

    config = LWConfig()
    sims = run_all(config)
    print_results(sims)
    with plotting_mode(show_plots):
        plot_sims(sims)
    persist_open_figures(save_dir, "lax_wendroff")


def cli() -> None:
    show_plots, save_dir = parse_common_cli("Run Lax-Wendroff analysis")
    run(show_plots=show_plots, save_dir=save_dir)
