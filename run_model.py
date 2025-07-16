import torch
import numpy as np
import cv2
import insightface
from src.fine_tuning import FineTuner  
from torchvision import transforms


face_model = insightface.app.FaceAnalysis(name='buffalo_l')
face_model.prepare(ctx_id=0) 


fine_tuned_model = FineTuner()
fine_tuned_model.load_state_dict(torch.load("outputs/fine_tuned_model.pt", map_location='cpu'))
fine_tuned_model.eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112)),
])

def get_embedding(image_path):
    img = cv2.imread(image_path)
    faces = face_model.get(img)
    if not faces:
        print(f"No face found in {image_path}")
        return None
    arcface_embedding = faces[0].embedding  
    tensor_emb = torch.tensor(arcface_embedding).float().unsqueeze(0)
    final_emb = fine_tuned_model(tensor_emb).detach().numpy().flatten()  
    return final_emb


image1 = "face1/photo.jpg"
image2 = "face2/download.jpg"

embedding1 = get_embedding(image1)
embedding2 = get_embedding(image2)

if embedding1 is not None and embedding2 is not None:
    
    cos_sim = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    print(f"Cosine Similarity between images: {cos_sim:.4f}")
else:
    print("Failed to extract embeddings from one or both images.")
