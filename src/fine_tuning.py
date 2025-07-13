import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import insightface
import cv2
from tqdm import tqdm

# Load InsightFace model
face_model = insightface.app.FaceAnalysis(name='buffalo_l')
face_model.prepare(ctx_id=0)


class FineTuner(nn.Module):
    def __init__(self, input_dim=512, embedding_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        return self.model(x)


def contrastive_loss(out1, out2, label, margin=1.0):
    euclidean_distance = nn.functional.pairwise_distance(out1, out2)
    loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                      label * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss


def get_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = face_model.get(img)
    if faces:
        return torch.tensor(faces[0].embedding, dtype=torch.float32)
    return None


def create_pairs(train_folder):
    data = []
    labels = []
    persons = os.listdir(train_folder)
    person_embs = {}

    for person in persons:
        person_path = os.path.join(train_folder, person)
        embeddings = []
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            emb = get_embedding(img_path)
            if emb is not None:
                embeddings.append(emb)
        person_embs[person] = embeddings

    
    for person, embs in person_embs.items():
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                data.append((embs[i], embs[j]))
                labels.append(0)

    
    people = list(person_embs.keys())
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            for emb1 in person_embs[people[i]]:
                for emb2 in person_embs[people[j]]:
                    data.append((emb1, emb2))
                    labels.append(1)

    return data, labels


def fine_tune_model(train_folder, epochs=10, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FineTuner().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("🔄 Creating training pairs...")
    data, labels = create_pairs(train_folder)
    data1 = torch.stack([d[0] for d in data])
    data2 = torch.stack([d[1] for d in data])
    labels = torch.tensor(labels).float()

    dataset = TensorDataset(data1, data2, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("🚀 Starting fine-tuning...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for a, b, label in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            a, b, label = a.to(device), b.to(device), label.to(device)
            out1, out2 = model(a), model(b)
            loss = contrastive_loss(out1, out2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"✅ Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "outputs/fine_tuned_model.pt")
    print("✅ Fine-tuned model saved to outputs/fine_tuned_model.pt")
