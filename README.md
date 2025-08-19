Autism Prediction using Machine Learning:-

This project focuses on building machine learning models to predict autism spectrum disorder (ASD) based on structured data. The notebook demonstrates the end-to-end pipeline including data exploration, preprocessing, handling imbalanced data, training multiple classifiers, evaluating performance, and saving the best model for future use.

📂 Project Structure
📦 Autism_Prediction_Project
 ┣ 📜 Autism_Preidiction_using_machine_Learning.ipynb   # Jupyter Notebook with code and analysis
 ┣ 📜 train.csv                                         # Dataset 
 ┣ 📜 README.md                                         # Documentation
 ┣ 📜 requirements.txt                                  # Python dependencies

⚙️ Installation & Dependencies

Install dependencies:

cd Autism-Prediction-ML
pip install -r requirements.txt

Required Libraries

numpy – numerical computations

pandas – data manipulation and analysis

matplotlib & seaborn – data visualization

scikit-learn – preprocessing, model building, and evaluation

imbalanced-learn (SMOTE) – handling imbalanced datasets

xgboost – gradient boosting classifier

pickle – saving trained models

🚀 Workflow
1. Import Dependencies

All required libraries for data preprocessing, visualization, and modeling are imported.

2. Data Loading & Inspection

The dataset (train.csv) is loaded using pandas.

Initial exploration is performed with shape, head(), and info().

Missing values and data distribution are analyzed.

3. Data Preprocessing

Encoding: Categorical features are converted into numerical values using LabelEncoder.

Imbalance Handling: Since the dataset is imbalanced, SMOTE (Synthetic Minority Oversampling Technique) is applied to balance the target classes.

Splitting: Data is split into training and testing sets using train_test_split.

4. Model Training

The following models are implemented:

Decision Tree Classifier – simple baseline model.

Random Forest Classifier – ensemble of decision trees for better generalization.

XGBoost Classifier – gradient boosting algorithm for improved accuracy.

5. Model Evaluation

Models are evaluated using:

Accuracy Score – overall correctness.

Confusion Matrix – class-level performance visualization.

Classification Report – precision, recall, and F1-score.

6. Model Saving

The best-performing model is serialized using pickle, enabling future predictions without retraining.

📊 Results

Decision Tree gave a baseline accuracy but tended to overfit.

Random Forest significantly improved performance by reducing variance.

XGBoost achieved the best trade-off between accuracy and generalization.

Final model can be easily integrated into a web application (Flask/Streamlit) or deployed as a standalone service for autism prediction.

📌 Usage Example

Once the model is trained and saved, you can load it in Python and make predictions:

import pickle

# Load the model
model = pickle.load(open("best_model.pkl", "rb"))

# Example input (replace with actual values)
sample_input = [[23, 1, 0, 4, 2, 1]]  

# Prediction
prediction = model.predict(sample_input)
print("Autism Prediction:", prediction)
