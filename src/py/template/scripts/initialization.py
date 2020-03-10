from dlens_vx.sta import AutoConnection

from template import initialize_experiment


def main(num_initializations: int):
    """
    Example for an experiment-run script.

    :param num_initializations: Dummy parameter, initialize multiple times
    """
    with AutoConnection() as connection:
        for _ in range(num_initializations):
            initialize_experiment(connection)
