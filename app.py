# app.py
from flask import Flask, request, render_template
import os
import numpy as np
import pickle
from extract_features import extract_features
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
features = np.load('features.npy')
with open('image_paths.pkl', 'rb') as f:
    image_paths = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["query_img"]
        filepath = os.path.join("static/uploads", file.filename)
        file.save(filepath)

        query_feat = extract_features(filepath)
        similarities = cosine_similarity([query_feat], features)[0]
        top_indices = similarities.argsort()[-5:][::-1]

        results = [image_paths[i] for i in top_indices]
        return render_template("index.html", query_path=filepath, results=results)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
