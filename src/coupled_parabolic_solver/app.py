from .analysis import compare_part_a, final_max
from .config import CoupledConfig
from .datasets import get_dataset
from .part_a_fourier import solve_t_fourier
from .part_a_implicit import solve_t_implicit
from .part_b_implicit import solve_phi_implicit
from .visualization import plot_case
from solver_utils import parse_common_cli, persist_open_figures, plotting_mode


def run(show_plots: bool = True, save_dir: str | None = None) -> None:
    config = CoupledConfig()
    x = config.x_grid()
    t = config.t_grid()

    with plotting_mode(show_plots):
        for case_id in config.case_ids():
            data = get_dataset(case_id)
            t_impl = solve_t_implicit(config, data)
            t_fourier, n_terms = solve_t_fourier(config, data)
            phi = solve_phi_implicit(config, data, t_impl)

            cmp_metrics = compare_part_a(t_impl, t_fourier)
            x_tmax, tmax = final_max(x, t_impl[-1])
            x_pmax, pmax = final_max(x, phi[-1])
            print(
                f"\nCase {case_id}: Fourier terms={n_terms}, "
                f"Linf(Timpl-Tfour)={cmp_metrics['linf']:.3e}, L2={cmp_metrics['l2']:.3e}"
            )
            print(f"  max T_end at x={x_tmax:.3f}, value={tmax:.3f}")
            print(f"  max phi_end at x={x_pmax:.3f}, value={pmax:.3f}")

            plot_case(case_id, x, t, t_impl, t_fourier, phi, show=show_plots)
    persist_open_figures(save_dir, "coupled_parabolic")


def cli() -> None:
    show_plots, save_dir = parse_common_cli("Run coupled parabolic solver")
    run(show_plots=show_plots, save_dir=save_dir)
