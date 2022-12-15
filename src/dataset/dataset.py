import sys
import os
import pandas
import json
import config

sys.path.append("../../")

default_data_path = os.path.join("..", "data")


class RawData:
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        self.split = None

    def get_data(self, split="train"):
        self.split = split
        data_path = f"{config.root_path}/raw_data/{self.dataset_type}/jsonl/{split}.jsonl"
        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line))

        return data


class DatasetMixin:
    # save dataset and retrieve dataset information.
    # load datasets based on parameters.
    @staticmethod
    def get_datasets():
        df = pandas.read_csv(os.path.join(default_data_path, "datasets.csv"))
        print(df)

    @staticmethod
    def save_dataset():
        pass
