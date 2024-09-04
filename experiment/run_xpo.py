import launchpad as lp

from ellm.args import default_args_validation, get_default_parser
from ellm.baselines.xpo import XPOActor, XPOLearner
from ellm.interface import get_program


def run_xpo(args):
    program, local_resources = get_program(args, XPOLearner, XPOActor)
    lp.launch(
        program,
        launch_type="local_mp",
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument("--xpo_alpha", type=float, default=5e-6)

    args = default_args_validation(parser.parse_args())
    run_xpo(args)
