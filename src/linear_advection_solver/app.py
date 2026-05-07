from .analysis import print_observations
from .config import AdvectionConfig
from .exact import exact_solution
from .schemes import scheme_registry
from .solver import SimulationResult, solve
from .visualization import plot_comparison
from solver_utils import parse_common_cli, persist_open_figures, plotting_mode


def run(show_plots: bool = True, save_dir: str | None = None) -> None:
    config = AdvectionConfig()
    x_grid = config.x_grid()
    exact = exact_solution(x_grid, config.t_target, config.a)

    results: list[SimulationResult] = []
    for name, step_fn in scheme_registry().items():
        results.append(solve(config, name, step_fn))

    print_observations(results, exact, x_grid)
    with plotting_mode(show_plots):
        plot_comparison(results, exact, x_grid)
    persist_open_figures(save_dir, "linear_advection")


def cli() -> None:
    show_plots, save_dir = parse_common_cli("Run linear advection solver")
    run(show_plots=show_plots, save_dir=save_dir)
