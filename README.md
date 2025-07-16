# Fine-Tuning Face Recognition on Low-Quality Images

This project demonstrates how to fine-tune a face recognition model to work better with low-quality images, such as those from CCTV cameras or poor lighting environments. It includes dataset preprocessing, baseline evaluation, contrastive learning-based fine-tuning, post-evaluation, and a real-time face matching demo.

---

## 🚀 How to Run the Project

### 💻 Set Up a Virtual Environment (Recommended)
# Create virtual environment
```bash
python -m venv venv

venv\Scripts\activate - windows
source venv/bin/activate - Mac/Linux
```

### 1. Clone the Repository

```bash
git clone https://github.com/joswinshogi/face_recognition.git
cd face_recognition

pip install -r requirements.txt
```
### 2. Ensure your Dataset Folder structure looks like this:
```bash
Images/
├── person1/
│   ├── high_quality/
│   └── low_quality/
├── person2/
│   ├── high_quality/
│   └── low_quality/
...
```
### 3. Run the file preprocessing.py
```bash
python src/preprocessing.py
```
### 4. Run Baseline Evaluation
```bash
python src/baseline_evaluation.py
```
### 5.  Fine-Tune the Model:- fine tuned model will be saved to outputs/fine_tuned_model.pt
```bash
python src/fine_tuning.py
```
### 6. Post-Fine-Tuning Evaluation 
```bash
python src/post_evaluation.py
```
### 7. Real-Time Face Matching
```bash
python src/generate_known_faces.py
```
### This creates:
```bash
known_faces/
├── ananthu.jpg
├── firoz.jpg
...

```
### 8. Run Real-Time Matching with Webcam
```bash
python realtime_match.py

```
### 9. Data-set Link
```bash
https://drive.google.com/file/d/1Vl1co8juIZkeM6urQV_JKfHRGr4dRu5W/view
```
### 10. You can use run_model.py to compare and check similarity between 2 images
```bash
python run_model.py
```






