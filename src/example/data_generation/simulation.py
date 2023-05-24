import dataclasses
import json
import pathlib
import pickle

import torch.nn


class AutopilotSimulation:
    def simulate(
        self,
        time_steps: int,
        initial_state: tuple[float, float],
        controls: list[float]
    ) -> list[float]:
        pass


@dataclasses.dataclass
class LSTMConfiguration:
    hidden_dimension: int
    number_layers: int
    dropout: float


class LSTMModel:
    def __init__(self, configuration: LSTMConfiguration) -> None:
        torch.nn.LSTM(
            hidden_size=configuration.hidden_dimension,
            num_layers=configuration.number_layers,
            dropout=configuration.dropout
        )
    def train(self, dataset):
        pass



def save_model(model: LSTMModel, model_path) -> None:
        pickle.dump(model, path)


def load_configuration_from_json(json_path: pathlib.Path) -> LSTMConfiguration:
    with open(json_path) as f:
        config_dict = json.load(f)

    return HyperparameterConfiguration(
        hidden_dimension=config_dict['hidden_dimension'],
        number_layers=config_dict['number_layers']
    )


def train_model(configuration_path, model_path, dataset_path):
    config = load_configuration_from_json(configuration_path)
    dataset = load_training_data(dataset_path)
    model = LSTMModel(config)
    model.train(dataset)
    save_model(model, model_path)


def gridsearch():
    for hyperparameter_set in configuration:
        train_model()