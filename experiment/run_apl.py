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

    args = default_args_validation(parser.parse_args())
    run_apl(args)
