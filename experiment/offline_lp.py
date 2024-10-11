"""Offline alignment with online vLLM evaluation."""

import launchpad as lp

from ellm.args import default_args_validation, get_default_parser
from ellm.interface import get_program
from ellm.learners import OfflineDAPLearner


def main(args):
    program, local_resources = get_program(args, OfflineDAPLearner)
    lp.launch(
        program,
        launch_type="local_mp",
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument("--preference_data", type=str, default="")
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument("--offline_buffer_path", type=str, default="./data/buffer.pkl")
    args = default_args_validation(parser.parse_args())
    main(args)
