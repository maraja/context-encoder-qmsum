import toml
import json
import itertools
from matplotlib import pyplot as plt


def get_experiments(key):
    return dict(toml.load("../settings/experiments.toml")).get(key, [])


def get_experiments_json(key):
    # Opening JSON file
    f = open(
        "../settings/experiments.json",
    )

    data = json.load(f)
    experiment_manifest = data.get(key, {})

    if not experiment_manifest:
        return {}

    # get every permutation of the experiment
    keys, values = zip(*experiment_manifest.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    f.close()
    return experiments


def save_results(experiment):
    with open("../results/experiment_results.json", "r+") as file:
        # First we load existing data into a dict.
        file_data = json.load(file)

        if "hash" in experiment:
            for i, saved in enumerate(file_data["results"]):
                if "hash" in saved and saved["hash"] == experiment["hash"]:
                    file_data["results"][i] = experiment
            # Sets file's current position at offset.
            file.seek(0)
            # convert back to json.
            json.dump(file_data, file, indent=4)
            return
        # Join new_data with file_data inside emp_details
        file_data["results"].append(experiment)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent=4)
        return


def print_break(text: str):
    print(
        "==============================================================================")
    print(
        f"=========================== {text} ==============================")
    print(
        "==============================================================================")
