import numpy as np
import pickle

features = np.load('features.npy')
with open('image_paths.pkl', 'rb') as f:
    paths = pickle.load(f)

print("Number of features:", len(features))
print("First image path:", paths[0])
print("Feature vector shape:", features[0].shape)
