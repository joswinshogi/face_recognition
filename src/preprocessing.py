import os
import shutil
import random

def split_low_quality_images(source_dir="Images", dest_dir="dataset", train_ratio=0.7):
    persons = os.listdir(source_dir)
    print(persons)

    for person in persons:
        low_path = os.path.join(source_dir, person, "low_quality")
        if not os.path.isdir(low_path):
            continue

        images = [img for img in os.listdir(low_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)

        train_images = images[:split_idx]
        eval_images = images[split_idx:]

        for phase, img_list in zip(["train", "eval"], [train_images, eval_images]):
            dest_person_dir = os.path.join(dest_dir, phase, person)
            os.makedirs(dest_person_dir, exist_ok=True)

            for img_file in img_list:
                src = os.path.join(low_path, img_file)
                dst = os.path.join(dest_person_dir, img_file)
                shutil.copy2(src, dst)

    
