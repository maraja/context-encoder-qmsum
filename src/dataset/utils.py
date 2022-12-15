from dataclasses import dataclass
from typing import List


@dataclass
class Params:
    model_name: str
    num_samples: int
    epochs: int
    batch_size: int
    path: str
    validation_pct: float
    max_seq_length: int
    data_augmentation: bool
    max_segment_length: int
    dataset_type: str
    augment_with_target_sentence_duplication: bool
    log_file_path: str


def save_results(params: Params, accuracy: List, val_accuracy: List, loss: List, val_loss: List):
    pass


# destructure the segments as they are currently in lists
def flatten(arr):
    return [item for sublist in arr for item in sublist]
