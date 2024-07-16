import launchpad as lp
from absl import app, flags, logging

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_producers", 2, "The number of concurrent producers.")


class Producer:
    def work(self, context):
        return context


class Consumer:
    def __init__(self, producers):
        self._producers = producers

    def run(self):
        futures = [
            producer.futures.work(context)
            for context, producer in enumerate(self._producers)
        ]
        results = [future.result() for future in futures]
        logging.info("Results: %s", results)
        lp.stop()


def make_program(num_producers):
    program = lp.Program("consumer_producers")
    with program.group("producer"):
        producers = [
            program.add_node(lp.CourierNode(Producer)) for _ in range(num_producers)
        ]
    node = lp.CourierNode(Consumer, producers=producers)
    program.add_node(node, label="consumer")
    return program


def main(_):
    program = make_program(num_producers=FLAGS.num_producers)
    lp.launch(program, launch_type="local_mp", terminal="current_terminal")


if __name__ == "__main__":
    app.run(main)
