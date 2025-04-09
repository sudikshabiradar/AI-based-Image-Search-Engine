import os
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import pickle

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features[0]

def build_feature_database(image_folder='static/images'):
    features = []
    img_paths = []

    for img_name in os.listdir(image_folder):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(image_folder, img_name)
            feat = extract_features(img_path)
            features.append(feat)
            img_paths.append(img_path)

    np.save('features.npy', features)
    with open('image_paths.pkl', 'wb') as f:
        pickle.dump(img_paths, f)

if __name__ == '__main__':
    build_feature_database()
