from src.preprocessing import split_low_quality_images
from src.baseline_evaluation import evaluate_baseline
from src.fine_tuning import fine_tune_model
from src.post_evaluation import evaluate_finetuned_model, fine_tuned_model
from src.generate_known_faces import generate_known_faces

if __name__ == "__main__":
    # split_low_quality_images()
    # evaluate_baseline(eval_folder="dataset/eval")
    # fine_tune_model(train_folder="dataset/train")
    # evaluate_finetuned_model("dataset/eval", fine_tuned_model)
    generate_known_faces()
