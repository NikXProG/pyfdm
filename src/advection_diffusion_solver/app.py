from .analysis import print_observations
from .config import GridConfig, default_scenarios
from .schemes import ux_schemes_registry
from .solver import SimulationResult, solve_explicit_euler
from .visualization import plot_results
from solver_utils import parse_common_cli, persist_open_figures, plotting_mode


def run(show_plots: bool = True, save_dir: str | None = None) -> None:
    grid = GridConfig()
    scenarios = default_scenarios()
    ux_schemes = ux_schemes_registry()

    results: list[SimulationResult] = []
    for scenario in scenarios:
        for ux_name, ux_scheme in ux_schemes.items():
            result = solve_explicit_euler(
                grid=grid,
                scenario=scenario,
                ux_scheme_name=ux_name,
                ux_scheme=ux_scheme,
            )
            results.append(result)

    print_observations(results)
    with plotting_mode(show_plots):
        plot_results(results)
    persist_open_figures(save_dir, "advection_diffusion")


def cli() -> None:
    show_plots, save_dir = parse_common_cli("Run advection-diffusion solver")
    run(show_plots=show_plots, save_dir=save_dir)
