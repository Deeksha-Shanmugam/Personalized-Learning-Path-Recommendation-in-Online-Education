'''
from preprocess import load_and_merge

def create_clients():
    df, features, label = load_and_merge(sample_size=50000, replace=True)
    grouped = df.groupby("code_module")
    clients = {}

    for module, group in grouped:
        X = group[features].values
        y = group[label].values
        clients[module] = (X, y)

    return clients
'''
from preprocess import load_and_merge
import numpy as np

def create_clients(num_clients=5):
    df, features, label = load_and_merge(sample_size=50000, replace=True)

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into num_clients parts
    client_data = np.array_split(df, num_clients)

    clients = {}

    for i, data in enumerate(client_data):
        X = data[features].values
        y = data[label].values
        clients[f"client_{i}"] = (X, y)

    return clients
