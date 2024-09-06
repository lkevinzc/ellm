import launchpad as lp

from ellm.args import default_args_validation, get_default_parser
from ellm.baselines.apl import APLActor, APLLearner
from ellm.interface import get_program


def run_apl(args):
    program, local_resources = get_program(args, APLLearner, APLActor)
    lp.launch(
        program,
        launch_type="local_mp",
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument(
        "--apl_pref_certainty_only",
        help=(
            "Fig 2b and Fig 5 both show this variant is better than random, "
            "while Fig 2b shows the learning is not robust with entropy."
        ),
        action="store_true",
    )

    args = default_args_validation(parser.parse_args())
    if args.apl_pref_certainty_only:
        args.num_samples = 2
    run_apl(args)
