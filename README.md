# Credit Card Fraud Detection

**Project Status: 🚧 Work in Progress 🚧**

This project is currently under development. The initial model shows promising results, but further analysis, hyperparameter tuning, and model comparison are planned.

## Project Overview

This project aims to build a machine learning model to detect fraudulent credit card transactions. Using a historical dataset of transactions, the model learns to differentiate between legitimate (Class 0) and fraudulent (Class 1) activities. The primary model explored in this initial phase is a `RandomForestClassifier`.

## Dataset

The dataset used is `creditcard_2023.csv`. It contains anonymized transaction features from European cardholders.

Key characteristics:
- **Features**: The dataset consists of 31 columns.
  - `id`: A unique identifier for each transaction.
  - `V1` to `V28`: These are numerical features that are the result of a PCA (Principal Component Analysis) transformation. Due to confidentiality, the original feature names are not provided.
  - `Amount`: The transaction amount. This feature has not been transformed by PCA.
  - `Class`: The target variable, where `1` indicates a fraudulent transaction and `0` indicates a legitimate one.
- **Data Balance**: This is a balanced dataset, with an equal 50/50 split between fraudulent and non-fraudulent transactions, which is ideal for training but atypical for real-world fraud detection scenarios.
- **Missing Values**: The dataset is clean with no missing values.

## Methodology

The current workflow follows these steps:

1.  **Data Exploration (EDA)**: Initial analysis was performed to understand the data's structure, statistics, and class distribution.
2.  **Data Preprocessing**:
    - The `id` and `Class` columns were separated from the features.
    - The data was split into training (80%) and testing (20%) sets.
    - The features were scaled using `StandardScaler` to ensure all features contribute equally to the model's performance.
3.  **Model Training**:
    - A `RandomForestClassifier` was chosen for its robustness and performance.
    - The model was trained on the scaled training data.
4.  **Model Evaluation**:
    - 5-fold cross-validation was performed on the training set to get a baseline F1-score.
    - The model's performance was evaluated on the unseen test set using a classification report and a confusion matrix.
    - Feature importance was analyzed to identify the most influential features in fraud prediction.

## Preliminary Results

The initial `RandomForestClassifier` model performs exceptionally well on this dataset:
- **Cross-Validation F1-Score**: ~0.985
- **Test Set Accuracy**: 99%
- **Test Set F1-Score**: 99% for both classes.

The most important features for prediction, according to the model, are `V14`, `V10`, `V12`, and `V17`.

![Feature Importance](images/feature_importance.png)
![Confusion Matrix](images/confusion_matrix.png)
*(Note: You would need to save the plots from your notebook into an `images` directory in your repository for these links to work).*

## Future Work

This project is ongoing. The next steps include:
- **Hyperparameter Tuning**: Use techniques like `GridSearchCV` or `RandomizedSearchCV` to find the optimal parameters for the Random Forest model.
- **Model Comparison**: Implement and evaluate other classification models, such as:
    - Logistic Regression
    - Gradient Boosting Machines (XGBoost, LightGBM)
    - A simple Neural Network using TensorFlow/Keras (since `tensorflow` was imported).
- **In-depth Analysis**: Further investigate the relationships between the top features and fraudulent activity.
- **Deployment**: Create a simple web application using Flask or Streamlit to serve the best-performing model.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```
2.  **Install the required libraries.** It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *(You will need to create a `requirements.txt` file from your environment. You can do this with `pip freeze > requirements.txt`)*

3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
4.  Open and run the `Credit_Card_Fraud_Detection.ipynb` notebook.
