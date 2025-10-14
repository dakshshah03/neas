import pickle
import torch

test_pkl = "/home/daksh/neas/data/chest_50.pickle"

with open(test_pkl, 'rb') as file:
    loaded_data = pickle.load(file)
    print("Successfully loaded data from the pickle file.")
    print(loaded_data.keys())