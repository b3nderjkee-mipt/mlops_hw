from dataclasses import dataclass


@dataclass
class Training:
    batch_size: int
    n_epochs: int
    learning_rate: float


@dataclass
class Model:
    output_dim: int
    dropout: float


@dataclass
class Params:
    model: Model
    training_params: Training


@dataclass
class Server:
    tracking_uri: str
    model_path: str
    batch_size: int
