from ellm.args import default_args_validation, get_default_parser
from ellm.learners.offline_dap import OfflineDAPLearner


def main(args):
    learner = OfflineDAPLearner(args)
    learner.run()


if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument("--preference_data", type=str, default="")
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument("--offline_buffer_path", type=str, default="./data/buffer.pkl")
    args = default_args_validation(parser.parse_args())
    main(args)
