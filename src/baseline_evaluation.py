import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import insightface

# Initialize the face model
face_model = insightface.app.FaceAnalysis(name='buffalo_l')
face_model.prepare(ctx_id=0)

def get_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = face_model.get(img)
    if faces:
        return faces[0].embedding
    return None

def load_embeddings(folder_path):
    embeddings = {}
    for person in os.listdir(folder_path):
        person_folder = os.path.join(folder_path, person)
        if not os.path.isdir(person_folder):
            continue
        embeddings[person] = []
        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)
            emb = get_embedding(img_path)
            if emb is not None:
                embeddings[person].append(emb)
    return embeddings

def generate_pairs(embeddings):
    X = []
    y = []

    people = list(embeddings.keys())

    for person in people:
        person_embs = embeddings[person]
        
        for i in range(len(person_embs)):
            for j in range(i + 1, len(person_embs)):
                X.append((person_embs[i], person_embs[j]))
                y.append(1)

    
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            emb1 = embeddings[people[i]]
            emb2 = embeddings[people[j]]
            for e1 in emb1:
                for e2 in emb2:
                    X.append((e1, e2))
                    y.append(0)
    return X, y

def evaluate_baseline(eval_folder):
    print("🔍 Extracting embeddings from evaluation set...")
    embeddings = load_embeddings(eval_folder)
    X, y = generate_pairs(embeddings)

    print("📏 Calculating similarity scores...")
    similarities = [cosine_similarity([a], [b])[0][0] for a, b in tqdm(X)]

    
    pos_scores = [s for s, label in zip(similarities, y) if label == 1]
    neg_scores = [s for s, label in zip(similarities, y) if label == 0]

    plt.hist(pos_scores, bins=50, alpha=0.5, label='Same Person')
    plt.hist(neg_scores, bins=50, alpha=0.5, label='Different People')
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Similarity Score Distribution")
    plt.legend()
    plt.savefig("outputs/similarity_distribution.png")
    plt.clf()

    
    fpr, tpr, _ = roc_curve(y, similarities)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("outputs/roc_curve.png")
    plt.clf()

    
    precision, recall, _ = precision_recall_curve(y, similarities)
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig("outputs/precision_recall.png")

    print("Baseline evaluation complete. Plots saved in `outputs/` folder.")
