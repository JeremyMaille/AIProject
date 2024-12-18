from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelConfig:
    learning_rate: float
    dropout_rates: Tuple[float, float, float]
    layer_sizes: Tuple[int, int, int]
    batch_size: int
    
    def to_dict(self):
        return {
            'learning_rate': self.learning_rate,
            'dropout_rates': self.dropout_rates,
            'layer_sizes': self.layer_sizes,
            'batch_size': self.batch_size
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

LEARNING_RATES = [0.001, 0.0005, 0.0001]
DROPOUT_RATES = [
    (0.3, 0.2, 0.1),
    (0.4, 0.3, 0.2),
    (0.5, 0.4, 0.3)
]
LAYER_SIZES = [
    (512, 256, 128),
    (256, 128, 64),
    (1024, 512, 256)
]
BATCH_SIZES = [32, 64, 128]