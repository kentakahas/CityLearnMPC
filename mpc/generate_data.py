# System operations
import os
import sys
import csv

# Data manipulation
import pandas as pd

# CityLearn
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from preprocessing import set_schema_buildings, set_schema_simulation_period


def set_schema(schema, building_index=None, simulation_periods=(0, 8760)):
    if building_index is None:
        building_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17]

    schema, _ = set_schema_buildings(schema, building_index=building_index)
    schema = set_schema_simulation_period(schema, start=simulation_periods[0], stop=simulation_periods[1])
    return schema


def setup_csv(env: CityLearnEnv, folder: str = 'data/') -> list:
    # Filenames for each file
    file_names = [os.path.join(folder, f"{b.name}.csv") for b in env.buildings]
    file_handles = [open(file_name, 'a', newline='') for file_name in file_names]

    # Headers for each observation
    header = env.observation_names[0]
    header.append("action")

    # Create csv files and write header
    for file in file_handles:
        writer = csv.writer(file)
        writer.writerow(header)

    # Close all file handles
    for file_handle in file_handles:
        file_handle.close()
    return file_names


def generate_csv(env: CityLearnEnv, file_names: list):
    env.reset()
    done = False

    current = 0

    while not done:
        rand_action = [x.sample() for x in env.action_space]
        obv = pd.DataFrame([{**b.observations()} for b in env.buildings])
        obv["action"] = [r[0] for r in rand_action]

        file_handles = [open(file_name, 'a', newline='') for file_name in file_names]

        for (index, row), file in zip(obv.iterrows(), file_handles):
            data = row.values
            writer = csv.writer(file)
            writer.writerow(data)

        # Close all file handles
        for file_handle in file_handles:
            file_handle.close()

        sys.stdout.write(f"\rCurrent timestep: {current}")
        sys.stdout.flush()

        current += 1
        _, _, done, _ = env.step(rand_action)


if __name__ == '__main__':
    DATASET_NAME = 'citylearn_challenge_2022_phase_all'
    rand_schema = DataSet.get_schema(DATASET_NAME)
    rand_schema = set_schema(rand_schema, simulation_periods=(0, 8760-(31*24)))
    rand_env = CityLearnEnv(rand_schema)

    files = setup_csv(rand_env)
    generate_csv(rand_env, files)
