import cv2
import torch
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from src.fine_tuning import FineTuner
import insightface

face_model = insightface.app.FaceAnalysis(name='buffalo_l')
face_model.prepare(ctx_id=0)

model = FineTuner()
model.load_state_dict(torch.load("outputs/fine_tuned_model.pt"))
model.eval()

def get_embedding(img):
    faces = face_model.get(img)
    if faces:
        return torch.tensor(faces[0].embedding, dtype=torch.float32)
    return None

def get_finetuned_embedding(img, model):
    emb = get_embedding(img)
    if emb is None:
        return None
    with torch.no_grad():
        return model(emb.unsqueeze(0)).squeeze(0).numpy()

known_embeddings = {}
for filename in os.listdir("known_faces"):
    path = os.path.join("known_faces", filename)
    img = cv2.imread(path)
    name = os.path.splitext(filename)[0]
    emb = get_finetuned_embedding(img, model)
    if emb is not None:
        known_embeddings[name] = emb
print("✅ Known faces loaded.")

# Start webcam
cap = cv2.VideoCapture(0)
threshold = 0.75

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_model.get(frame)
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        face_crop = frame[y1:y2, x1:x2]
        emb = get_finetuned_embedding(face_crop, model)

        matched_name = "Unknown"
        if emb is not None:
            for name, known_emb in known_embeddings.items():
                sim = cosine_similarity([emb], [known_emb])[0][0]
                if sim > threshold:
                    matched_name = name
                    break

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, matched_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Matching", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
