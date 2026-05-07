from .analysis import build_conclusion, compare_final_profiles
from .config import ViscosityConfig
from .derivation import derivation_summary
from .solver import run_simulation
from .visualization import plot_comparison
from solver_utils import parse_common_cli, persist_open_figures, plotting_mode


def run(show_plots: bool = True, save_dir: str | None = None) -> None:
    config = ViscosityConfig()
    print(derivation_summary(config))

    pair = run_simulation(config)
    metrics = compare_final_profiles(pair)
    print(
        f"\nМетрики сравнения: L_inf={metrics['l_inf']:.3e}, "
        f"L2={metrics['l2']:.3e}, "
        f"amp_upwind={metrics['amp_upwind']:.3e}, "
        f"amp_advdiff={metrics['amp_advdiff']:.3e}"
    )
    print(build_conclusion(metrics))

    with plotting_mode(show_plots):
        plot_comparison(pair)
    persist_open_figures(save_dir, "numerical_viscosity")


def cli() -> None:
    show_plots, save_dir = parse_common_cli("Run numerical viscosity solver")
    run(show_plots=show_plots, save_dir=save_dir)
