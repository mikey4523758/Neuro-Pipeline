# ============================================================
# Name:    Michael Lopez
# Date:    3/29/26
# Purpose: Train and evaluate SVM and Random Forest classifiers
#          on extracted EEG features. Splits data into train/test
#          sets, fits both models using config-defined parameters,
#          and returns accuracy scores alongside the trained model
#          objects for downstream use or comparison.
# ============================================================

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json

def train_and_evaluate(X, y, config):
    """
    Trains SVM and Random Forest models and compares performance.
    """
    # Split the dataset into training and testing subsets.
    # test_size and random_state are pulled from config for reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config['random_state']
    )
    
    # Dictionary to store accuracy scores for each model
    results = {}
    
    # --- SVM ---
    # Initialize the Support Vector Machine with parameters from config
    # (e.g., kernel type, regularization C, gamma)
    svm = SVC(**config['svm_params'])
    svm.fit(X_train, y_train)

    # Generate predictions on the held-out test set and record accuracy
    y_pred_svm = svm.predict(X_test)
    results['SVM'] = accuracy_score(y_test, y_pred_svm)
    
    # --- Random Forest ---
    # Initialize the Random Forest ensemble with parameters from config
    # (e.g., n_estimators, max_depth, random_state)
    rf = RandomForestClassifier(**config['rf_params'])
    rf.fit(X_train, y_train)

    # Generate predictions on the held-out test set and record accuracy
    y_pred_rf = rf.predict(X_test)
    results['RandomForest'] = accuracy_score(y_test, y_pred_rf)
    
    # Print a quick side-by-side accuracy summary for both models
    print(f"Results: {results}")

    # Return accuracy scores dict and both trained model objects
    # so the caller can inspect, save, or run further evaluation on them
    return results, {'SVM': svm, 'RF': rf}