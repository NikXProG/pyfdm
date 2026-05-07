import argparse


def parse_common_cli(description: str) -> tuple[bool, str | None]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable interactive plotting for headless/batch mode.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory for saving generated figures.",
    )
    args = parser.parse_args()
    show_plots = not args.no_plot
    return show_plots, args.save_dir
