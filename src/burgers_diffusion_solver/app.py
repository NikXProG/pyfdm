from .analysis import print_observations
from .config import BurgersConfig
from .solver import solve_single_epsilon, sweep_epsilons
from .visualization import plot_epsilon_sweep, plot_surface
from solver_utils import parse_common_cli, persist_open_figures, plotting_mode


def run(show_plots: bool = True, save_dir: str | None = None) -> None:
    config = BurgersConfig()

    base = solve_single_epsilon(config, config.eps_base)
    with plotting_mode(show_plots):
        plot_surface(base)

    eps_values = config.epsilon_values()
    profiles = sweep_epsilons(config, eps_values)
    print_observations(profiles)
    with plotting_mode(show_plots):
        plot_epsilon_sweep(base.x_grid, profiles)
    persist_open_figures(save_dir, "burgers_diffusion")


def cli() -> None:
    show_plots, save_dir = parse_common_cli("Run Burgers diffusion solver")
    run(show_plots=show_plots, save_dir=save_dir)
