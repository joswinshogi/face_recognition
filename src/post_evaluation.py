import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from src.fine_tuning import FineTuner
import insightface


fine_tuned_model = FineTuner()
fine_tuned_model.load_state_dict(torch.load("outputs/fine_tuned_model.pt"))
fine_tuned_model.eval()


face_model = insightface.app.FaceAnalysis(name='buffalo_l')
face_model.prepare(ctx_id=0)


def get_original_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = face_model.get(img)
    if faces:
        return torch.tensor(faces[0].embedding, dtype=torch.float32)
    return None

def get_finetuned_embedding(image_path, model):
    original_emb = get_original_embedding(image_path)
    if original_emb is None:
        return None
    with torch.no_grad():
        new_emb = model(original_emb.unsqueeze(0)).squeeze(0)
    return new_emb.numpy()


def load_finetuned_embeddings(eval_folder, model):
    embeddings = {}
    for person in os.listdir(eval_folder):
        person_path = os.path.join(eval_folder, person)
        if not os.path.isdir(person_path):
            continue
        person_embs = []
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            emb = get_finetuned_embedding(img_path, model)
            if emb is not None:
                person_embs.append(emb)
        if person_embs:
            embeddings[person] = person_embs
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


def evaluate_finetuned_model(eval_folder, model):
    print("🔍 Extracting fine-tuned embeddings...")
    embeddings = load_finetuned_embeddings(eval_folder, model)
    print("🔁 Generating evaluation pairs...")
    X, y = generate_pairs(embeddings)

    print("📏 Calculating cosine similarities...")
    similarities = [cosine_similarity([a], [b])[0][0] for a, b in tqdm(X)]

    
    pos_scores = [s for s, label in zip(similarities, y) if label == 1]
    neg_scores = [s for s, label in zip(similarities, y) if label == 0]

    plt.hist(pos_scores, bins=50, alpha=0.5, label='Same Person')
    plt.hist(neg_scores, bins=50, alpha=0.5, label='Different People')
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Fine-tuned Similarity Distribution")
    plt.legend()
    plt.savefig("outputs/similarity_distribution_finetuned.png")
    plt.clf()

    
    fpr, tpr, _ = roc_curve(y, similarities)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Fine-tuned ROC Curve")
    plt.legend()
    plt.savefig("outputs/roc_curve_finetuned.png")
    plt.clf()

    
    precision, recall, _ = precision_recall_curve(y, similarities)
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Fine-tuned Precision-Recall Curve")
    plt.legend()
    plt.savefig("outputs/precision_recall_finetuned.png")

    
