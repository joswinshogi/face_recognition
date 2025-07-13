import os
import shutil

def generate_known_faces(images_dir="Images", output_dir="known_faces"):
    os.makedirs(output_dir, exist_ok=True)

    people = os.listdir(images_dir)
    for person in people:
        person_path = os.path.join(images_dir, person, "high_quality")
        if not os.path.exists(person_path):
            print(f"⚠️ Skipping {person}: high_quality folder not found.")
            continue

        
        images = [f for f in os.listdir(person_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not images:
            print(f"⚠️ No images found for {person}")
            continue

        source_path = os.path.join(person_path, images[0])
        target_path = os.path.join(output_dir, f"{person}.jpg")

        shutil.copy2(source_path, target_path)
        print(f"✅ Copied {source_path} → {target_path}")

    
