# AI-Powered-Enhanced-EHR-Imaging-Documentation-System

### 📘 Module 1 – Data Collection & Preprocessing  

---

## 🔍 Overview
This module focuses on the **collection, cleaning, encoding, and preparation** of both structured and unstructured healthcare data.  
It forms the foundation for subsequent modules like **Medical Image Enhancement (Module 2)** and **AI-based Disease Prediction (Module 3)**.

The main goal is to build a clean, consistent, and machine-readable dataset that integrates various healthcare sources such as:
- Patient medical records  
- ICD-10 diagnostic codes  
- Doctor’s handwritten prescriptions (image data)  
- Chest X-ray images  

---

## 🎯 Objectives
- Collect and preprocess structured (CSV) and unstructured (image/text) data  
- Handle missing values, duplicates, and inconsistent data formats  
- Encode categorical columns for machine learning compatibility  
- Scale numerical features for model stability  
- Generate synthetic links between patient records, prescriptions, and X-ray data  
- Save all cleaned datasets for use in later modules  

---

## 🧩 Datasets Used
| Dataset Name | Type | Description |
|---------------|------|-------------|
| **Healthcare Dataset** | Structured | Contains patient details like age, gender, condition, admission details, and test results |
| **ICDCodeSet.csv** | Structured | Contains ICD-10 medical codes and disease descriptions |
| **Doctor’s Handwritten Prescription BD Dataset** | Unstructured (Images) | Handwritten prescription images used for OCR and text recognition |
| **X-ray Dataset** | Unstructured (Images) | Chest X-ray images for later enhancement and diagnosis |

---

## ⚙️ Data Preprocessing Steps

### 1️⃣ Environment Setup  
Imported necessary libraries including `pandas`, `numpy`, `matplotlib`, `seaborn`, and scikit-learn preprocessing modules.

### 2️⃣ Data Loading  
Loaded all structured and unstructured datasets from `/content/` or Google Drive and inspected their structure.

### 3️⃣ Data Cleaning  
- Checked for missing values  
- Filled missing numeric values with median/mode  
- Converted date columns into `Stay_Duration` (Discharge − Admission)  
- Dropped irrelevant columns such as Name, Doctor, Hospital, etc.

### 4️⃣ Data Encoding & Scaling  
- Encoded categorical variables (Gender, Blood Type, Admission Type, etc.) using `LabelEncoder`  
- Scaled numerical values using `StandardScaler`  

### 5️⃣ ICD Dataset Preparation  
- Removed duplicates  
- Cleaned column names  
- Created a mapping dictionary for ICD codes  

### 6️⃣ Prescription Dataset Processing  
- Created dataframes for `train`, `test`, and `validation` splits  
- Generated image paths and unique prescription IDs  

### 7️⃣ X-ray Dataset Processing  
- Indexed image paths for X-ray splits (train/test/val)  
- Stored class labels and IDs  

### 8️⃣ Synthetic Linking  
- Randomly associated each patient record with `prescription_id` and `xray_id`  
- Ensured consistent linkage for later modules  

### 9️⃣ Saving Cleaned Data  
Exported the following cleaned CSV files:
- `cleaned_healthcare_dataset.csv`  
- `cleaned_icdcodeset.csv`  
- `cleaned_prescription_dataset.csv`  
- `cleaned_xray_index.csv`

---

## 📊 Exploratory Data Analysis (EDA)
Performed post-cleaning validation to ensure:
- No missing or duplicate values remain  
- Encoded data distributions are balanced  
- Numerical features are properly scaled  
- Correlations between attributes are visualized through heatmaps

---


## ✅ Module-1 Results Summary
- Structured and unstructured datasets successfully preprocessed  
- Patient records are standardized and machine-readable  
- Clean datasets saved for Module 2 (Medical Image Enhancement)  
- Dataset linkage between EHR, prescriptions, and X-rays established  

---

# 🩻 Module 2: Medical Imaging Enhancement

## 🎯 Objective
Enhance the quality of medical images — **X-rays** and **handwritten prescriptions** — using traditional image processing techniques to improve visibility, contrast, and diagnostic usability before AI model training.

---

## ⚙️ Methods Used

This module applies multiple enhancement and preprocessing techniques to the images to ensure uniformity and high diagnostic quality.

| Technique | Description |
|------------|--------------|
| **CLAHE (Contrast Limited Adaptive Histogram Equalization)** | Enhances local contrast and brightness. |
| **Gaussian & Bilateral Filtering** | Reduces image noise while preserving edges. |
| **Unsharp Masking** | Improves fine structural and text details. |
| **Normalization** | Maintains consistent intensity across all datasets. |

---

## 🧪 Evaluation Metrics

To validate the quality of the enhanced images, the following quantitative metrics were computed:

| Metric | Description | Ideal Range |
|---------|--------------|--------------|
| **PSNR (Peak Signal-to-Noise Ratio)** | Measures how much noise was removed — higher is better. | > 20 dB |
| **SSIM (Structural Similarity Index)** | Evaluates visual similarity and structural fidelity — closer to 1 is better. | > 0.70 for X-rays, > 0.85 for Prescriptions |

---

## 📈 Results Summary

| Dataset | PSNR (avg) | SSIM (avg) | Remarks |
|----------|-------------|-------------|----------|
| **X-ray Images** | 20.94 | 0.685 | Good quality; slightly improvable with mild sharpening. |
| **Prescription Images** | 21.266 | 0.891 | Excellent enhancement and readability. |

---

## 📁 Output Files

| File/Folder | Description |
|--------------|--------------|
| `/content/enhanced_images/` | Folder containing all enhanced X-ray and prescription images. |
| `/content/cleaned_data/enhanced_image_index.csv` | CSV containing image paths with PSNR and SSIM scores. |
| `/content/cleaned_data/module2_summary.txt` | Text summary of all enhancement metrics. |

---

## 🧩 Integration Notes
- These enhanced images will be **used as input for Module 3**, where AI models will be trained for disease prediction and diagnostic support.
- The improvement in contrast and clarity ensures that **ML and CNN-based models** can extract better features from images.
- The enhanced datasets maintain compatibility with the cleaned structured data generated in **Module 1**.



---



## 🧠 Key Learnings
- Image enhancement improves **diagnostic interpretability** before AI model training.
- PSNR and SSIM are effective metrics to evaluate **image restoration and denoising quality**.
- Proper preprocessing significantly influences **AI model accuracy** in later stages.

---

## ✅ Module-2 Results Summary

- X-ray and Prescription image datasets successfully enhanced and denoised  
- Applied CLAHE, Bilateral Filtering, Gaussian Smoothing, and Unsharp Masking for visual clarity  
- Achieved strong quality metrics for medical image usability  
- Average X-ray quality → PSNR: **20.94**, SSIM: **0.685**  
- Average Prescription quality → PSNR: **21.266**, SSIM: **0.891**  
- Enhanced images saved to `/content/enhanced_images/`  
- Quality metrics and summary stored in `/content/cleaned_data/`  
- Outputs ready for Module 3: **AI-based Medical Image Analysis and Prediction**


---

# 🤖 Module 3: Clinical Note Generation & ICD-10 Coding Automation

## 🎯 Objective
Automate **clinical documentation** and **medical coding** using **Hugging Face Transformer models**, to generate structured clinical notes and corresponding **ICD-10 codes** from patient data.  
This module serves as the **AI core** of the project — transforming structured datasets into meaningful medical narratives and diagnostic classifications.

---



## ⚙️ Workflow Overview

The process involves three major steps:

| Step | Process |
|------|----------|
| **1️⃣ Prompt Preparation** | Generate AI-friendly text prompts using patient details (name, condition, test results). |
| **2️⃣ Clinical Note Generation** | Use a fine-tuned text-to-text model to produce structured clinical notes. |
| **3️⃣ ICD-10 Code Automation** | Predict corresponding ICD-10 codes for each generated clinical note. |



---

## 🧠 Models Used (from Hugging Face)

| Task | Model Name | Purpose | Framework |
|------|-------------|----------|------------|
| **Clinical Note Generation** | `google/flan-t5-large` | Generates structured clinical notes based on patient information. | Transformers (Text2Text Generation) |
| **ICD-10 Code Prediction** | `AkshatSurolia/ICD-10-Code-Prediction` | Automatically predicts ICD-10 codes from generated notes. | Transformer (Sequence Classification) |

---

## 🩺 Clinical Note Generation

### 🧩 Method
- The **Flan-T5-Large** model from Hugging Face was used for generating **realistic, structured clinical notes**.
- Custom prompts were created with patient demographics, condition, test results, and X-ray summaries.
- The model was tuned for structured output containing:
  - **Clinical Summary**
  - **Assessment**
  - **Plan / Recommendations**

### 🧾 Sample Output

| Input (Prompt Snippet) | Output (Generated Note) |
|----------------|----------------|
| *Patient: Bobby Jackson, 30 years, Male, Condition: Cancer* |  **Clinical Summary:** Patient presents with fatigue and weight loss.<br> |

---

## 💊 ICD-10 Code Prediction

### 🧩 Method
- Model: `AkshatSurolia/ICD-10-Code-Prediction`
- For each **generated clinical note**, the model predicts an ICD-10 code.
- Utilized **Transformers & PyTorch** backend for classification.

### 🧾 Sample Output

| Clinical Note (Input) | Predicted ICD-10 Code |
|----------------|---------------------------|
| “Patient presents with chest discomfort and abnormal X-ray findings…” | **I20.9** – Angina pectoris, unspecified |
| “Observed symptoms consistent with acute appendicitis…” | **K35.80** – Unspecified acute appendicitis |

---

## 🧮 Unified Dataset Creation

After the AI generation process, all outputs were **combined into a single CSV dataset** for integration and visualization.

| Column | Description |
|--------|-------------|
| `patient_id` | Unique patient identifier |
| `Name`, `Age`, `Gender` | Patient demographics |
| `Medical Condition` | Primary diagnosis |
| `clinical_note` | AI-generated structured note |
| `Predicted_ICD` | ICD-10 code predicted by the model |
| `Test Results`, `Medication`, `xray_caption` | Contextual data for note generation |

**File Created:**  
📁 `/content/UnifiedDataset_with_images.csv`

---

## 🧭 Streamlit Dashboard Integration

After generating the unified dataset, an **interactive Streamlit Dashboard** was developed for clinical data visualization and exploration.

| Section | Functionality |
|----------|---------------|
| **Overview Table** | Displays all patient records with condition and ICD code. |
| **Detailed View** | Shows individual patient details, AI-generated notes, and ICD prediction. |
| **Data Insights** | Includes Pie Chart, Bar Graph, and Line Graph for better trend analysis. |

### 💡 Visualization Highlights

| Chart Type | Purpose |
|-------------|----------|
| **Pie Chart** | Shows medical condition distribution among patients. |
| **Bar Graph** | Displays top 10 most frequent ICD-10 codes. |
| **Line Graph** | Tracks hospital admissions over time. |

---

## 📊 Streamlit App Layout Overview

| Section | Description |
|----------|-------------|
| **Header** | Displays the app title and tagline with a professional aesthetic. |
| **Search / Filter Sidebar** | Allows filtering by patient ID or name. |
| **Clinical Note Viewer** | Shows AI-generated notes with styled text box (`#e6f2ff`). |
| **ICD Code Display** | Highlighted in a success box (green) for easy readability. |
| **Analytics Area** | Interactive Plotly visualizations for medical trend insights. |
| **Footer** | Includes author information and GitHub/LinkedIn links. |

---

## ⚙️ Key Libraries Used

| Category | Libraries |
|-----------|------------|
| **AI / NLP** | `transformers`, `torch`, `tqdm` |
| **Visualization** | `streamlit`, `plotly`, `matplotlib` |
| **Data Processing** | `pandas`, `numpy` |
| **Deployment** | `pyngrok` (for public Colab app access) |
| **Utilities** | `os`, `Pillow`, `re` |

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `/content/UnifiedDataset_with_images.csv` | Final unified dataset with AI notes and ICD codes |
| `app.py` | Streamlit dashboard code |
| `requirements.txt` | All dependencies required for model and dashboard execution |

---

## 🧩 Integration Notes

- The generated **Unified Dataset** acts as the input for the **Streamlit visualization dashboard**.  
- Enhancements in the model prompts improved note readability and diagnosis consistency.  
- ICD-10 prediction achieved high contextual relevance with clinical notes.  
- The overall system allows real-time EHR exploration through a **web-based AI dashboard**.

---

## 🧠 Key Learnings

- Leveraging **pre-trained transformer models** (like T5 and ICD prediction) simplifies medical text generation.  
- AI-driven ICD coding automation reduces human effort and improves coding accuracy.  
- Combining AI outputs into a unified visualization system aids in **clinical decision-making** and **medical record transparency**.

---

## ✅ Module-3 Results Summary

- ✅ Structured clinical notes generated using **Flan-T5-Large**  
- ✅ Automated ICD-10 code prediction using **AkshatSurolia/ICD-10-Code-Prediction**  
- ✅ Created unified dataset: `/content/UnifiedDataset_with_images.csv`  
- ✅ Designed a clean and interactive Streamlit dashboard (`app.py`)  
- ✅ Integrated analytical visuals — Pie, Bar, and Line charts  
- ✅ Deployment-ready via **Ngrok** for Colab-based public access  

---

## 🧾 Summary

Module 3 successfully bridges AI text generation, medical coding automation, and interactive visualization — forming the **intelligent documentation core** of the entire EHR system.  
This serves as a critical step toward building AI-driven, structured, and interpretable healthcare data platforms.

## AI-Powered Clinical Dashboard

<img width="1918" height="933" alt="Screenshot 2025-11-02 201813" src="https://github.com/user-attachments/assets/be0ab400-dfe6-4240-a3a9-c92fac4b2b7c" />

---

<img width="1919" height="961" alt="Screenshot 2025-11-02 201846" src="https://github.com/user-attachments/assets/e9424f92-0a3d-4815-83f3-79eb560e23e2" />

---

<img width="1912" height="956" alt="Screenshot 2025-11-02 201906" src="https://github.com/user-attachments/assets/f682e8cf-bbee-4695-9c4b-a23d5dd339b8" />

---
# 📘 Module 4 — Integration & Deployment
### *AI-Powered Enhanced EHR Imaging & Documentation System*

This module represents the final deployment stage of the project using **Google Colab + Streamlit + Ngrok**.  
The deployed dashboard allows real-time interaction with patient records, AI-generated clinical notes, ICD-10 code prediction, and visual medical analytics.

---

## 🚀 Deployment Overview

The dashboard is executed in **Google Colab** and exposed publicly using **Ngrok**, enabling access from any system without requiring local Python installation.

### ✔ Features Included in This Deployment
- Patient search & selection
- AI-Generated clinical notes using *Flan-T5*
- ICD-10 Code Prediction
- Medical condition keywords extraction
- Data visual insights (Pie, Bar, Line graphs)
- Fully responsive Streamlit UI

---

## 📁 Folder Contents
| File / Folder | Description |
|---------------|------------|
| `app.py` | Streamlit dashboard application |
| `Module4_Ai_Powered_Clinical_Dashboard.ipynb` | Google Colab notebook for deployment |
| `module4_ai_powered_clinical_dashboard.py` | Backend logic file |
| `UnifiedDataset_with_images.csv` | Final dataset used for predictions and visualization |
| `deployment_screenshots/` | Screenshots of deployed working system |
| `requirements.txt` | Required dependency list |

---

# 🖥 How to Run This App (Google Colab + Ngrok Deployment)

### **1️⃣ Upload required files into Google Colab**
- `app.py`
- `UnifiedDataset_with_images.csv`
- `requirements.txt`

### **2️⃣ Install all dependencies**
```bash
!pip install -r requirements.txt

### **3️⃣ Start Streamlit and create Ngrok tunnel**
- `!streamlit run app.py --server.port 6006 & npx ngrok http 6006`

### **4️⃣ Copy the Ngrok Public URL generated**
- `https://unquaking-alberta-caenogenetic.ngrok-free.dev`
🔗 Use this URL to access the dashboard from any device (desktop / mobile / tablet).

### 📸 Deployment Demonstration Screenshots

All proof of deployment screenshots are available inside: /deployment_screenshots/

### Module 4 Outcome

✔ Full integration and deployment completed
✔ Working real-time dashboard using Streamlit
✔ AI inference working via Flan-T5 & ICD-10 model
✔ Accessible anywhere using Ngrok public link
✔ Ready for evaluator demo and project submission


