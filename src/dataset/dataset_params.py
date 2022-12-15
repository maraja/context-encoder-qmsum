import random
import string
import os


def generate_random_string(length=5):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


class DatasetParams:
    def __init__(
        self,
        *,
        data_path="",
        dataset_slice="training",
        dataset_type="default",
        pct_data=0.005,
        max_seq_length=256,
        random=False,
        augment_pct=0.0,
        remove_duplicates=False,
        max_segment_length=5,
    ):
        """function to extract and process dataset

        Args:
            data_path (string): path to your dataset
            pct_data (float, optional): percentage of your dataset you want to use. Defaults to 0.005.
            max_seq_length (int, optional): max sequence length for BERT embeddings to input. Defaults to 256.
            random (boolean, optional): whether to randomize the data segments
            remove_duplicates (boolean, optional): whether to remove lines that are the same. Only used for inference.
            max_segment_length (int, optional): maximum number of items in a segment

        Raises:
            Exception: [description]

        Returns:
            list of sentences, list of labels: returns sentences and labels
        """
        # if not os.path.exists(dataset_type):
        #     os.makedirs(dataset_type)
        # self.data_path = os.path.join(
        #     ".",
        #     dataset_type,
        #     f"{pct_data}-{dataset_slice}-{generate_random_string()}.pkl",
        # )
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.pct_data = pct_data
        self.max_seq_length = max_seq_length
        self.random = random
        self.augment_pct = augment_pct
        self.remove_duplicates = remove_duplicates
        self.max_segment_length = max_segment_length
        self.dataset_slice = dataset_slice
        self.filename = self.data_path.split("/")[-1].split("\\")[-1]
        self.savefile = (
            f"{dataset_type}-{pct_data}-{dataset_slice}-{generate_random_string()}.pkl"
        )

    def save_dataset(self):
        pass
