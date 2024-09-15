import launchpad as lp

from ellm.args import default_args_validation, get_default_parser
from ellm.baselines.rs import RSActor
from ellm.interface import get_program
from ellm.learners import DAPwRMLearner


def run_rs(args):
    program, local_resources = get_program(args, DAPwRMLearner, RSActor)
    lp.launch(
        program,
        launch_type="local_mp",
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    parser = get_default_parser()

    args = parser.parse_args()
    args.num_ensemble = 1
    # Just to reuse existing class, but has nothing to do with DTS;
    # we re-defined an Explorer in RSActor.
    args.exp_method = "EnnDTS"
    args = default_args_validation(args)
    run_rs(args)
