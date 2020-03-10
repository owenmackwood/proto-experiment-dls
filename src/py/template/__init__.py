from dlens_vx.sta import generate, ExperimentInit, PlaybackProgramExecutor


def initialize_experiment(connection: PlaybackProgramExecutor):
    """
    Trivial example for an experiment library function.

    :param connection: Connection to be used for initialization
    """
    builder, _ = generate(ExperimentInit())
    connection.run(builder.done())


def add(val_a: int, val_b: int) -> int:
    """
    Special implementation of a number adder.
    :param val_a: First addition argument
    :param val_b: Second addition argument
    :return: Addition result
    """
    return val_a + val_b
