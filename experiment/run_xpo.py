import launchpad as lp

from ellm.args import default_args_validation, get_default_parser
from ellm.interface import get_program
from ellm.learners import DAPLearner, DAPwRMLearner


def run_xpo(args):
    if args.learn_rm:
        learner_cls = DAPwRMLearner
    else:
        learner_cls = DAPLearner
    program, local_resources = get_program(args, learner_cls)
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
