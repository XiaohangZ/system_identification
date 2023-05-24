import pathlib

from example.data_generation.simulation import AutopilotSimulation


def main() -> None:
    dataset_directory = pathlib.Path('datasets/autopilot')
    train_directory = dataset_directory.joinpath('train')

    AutopilotSimulation()