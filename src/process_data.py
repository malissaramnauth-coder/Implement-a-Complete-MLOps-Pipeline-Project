# write a program that splits my dataset into train and test sets and saves them to the specified paths in the config file. The program should read the dataset from the path specified in the config file, split it into train and test sets, and save them to the paths specified in the config file. The program should also log the number of rows in the train and test sets.
# Here's a Python program that reads a dataset from a specified path in a configuration file, splits it into training and testing sets, saves them to the specified paths, and logs the number of rows in each set. This program uses the `pandas` library for data manipulation and `yaml` for reading the configuration file.
import pandas as pd
import yaml
import os
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def split_data(config):
    # Load dataset
    df = pd.read_csv(config['data']['dataset_path'])
    logging.info(f"Loaded dataset with {df.shape[0]} rows.")

    # Split dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    logging.info(f"Split dataset into {train_df.shape[0]} train rows and {test_df.shape[0]} test rows.")

    # Save split datasets
    train_df.to_csv(config['data']['train_path'], index=False)
    test_df.to_csv(config['data']['test_path'], index=False)
    logging.info("Saved split datasets.")

if __name__ == "__main__":
    config = load_config("configs/model.yaml")
    split_data(config)

