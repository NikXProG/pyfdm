from pathlib import Path
from contextlib import contextmanager

import matplotlib.pyplot as plt


def persist_open_figures(save_dir: str | None, stem: str) -> None:
    if not save_dir:
        return
    target = Path(save_dir)
    target.mkdir(parents=True, exist_ok=True)
    for idx, fig_num in enumerate(plt.get_fignums(), start=1):
        figure = plt.figure(fig_num)
        figure.savefig(target / f"{stem}_{idx:02d}.png", dpi=150, bbox_inches="tight")


@contextmanager
def plotting_mode(show_plots: bool):
    original_show = plt.show
    if not show_plots:
        plt.show = lambda *args, **kwargs: None
    try:
        yield
    finally:
        plt.show = original_show
