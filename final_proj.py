# Import necessary libraries
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Import Classification Algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Import dataset
from sklearn.datasets import load_breast_cancer

# Function to calculate performance metrics
def calculate_metrics(y_true, y_pred):
    # Get TN, FP, FN, TP Values from Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    tss = tp / (tp + fn) - fpr
    hss = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    return accuracy, precision, recall, fpr, fnr, tss, hss

def main():
    # Parse dataset
    bc = load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Random Forest Algorithm
    rf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    rf.predict(X_train)
    y_pred = rf.predict(X_test)
    rf_metrics = calculate_metrics(y_test, y_pred)

    # Deep Learning
    dl = MLPClassifier(hidden_layer_sizes=(100), max_iter=1000).fit(X_train, y_train)
    dl.predict(X_train)
    y_pred = dl.predict(X_test)
    dl_metrics = calculate_metrics(y_test, y_pred)

    # SVM Algorithm
    svm = SVC(kernel='linear').fit(X_train, y_train)
    svm.predict(X_train)
    y_pred = svm.predict(X_test)
    svm_metrics = calculate_metrics(y_test, y_pred)

    # Display the results
    results_df = pd.DataFrame({
        'Algorithm': ['Random Forest', 'Deep Learning', 'SVM'],
        'Accuracy': [rf_metrics[0], dl_metrics[0], svm_metrics[0]],
        'Precision': [rf_metrics[1], dl_metrics[1], svm_metrics[1]],
        'Recall': [rf_metrics[2], dl_metrics[2], svm_metrics[2]],
        'FPR': [rf_metrics[3], dl_metrics[3], svm_metrics[3]],
        'FNR': [rf_metrics[4], dl_metrics[4], svm_metrics[4]],
        'TSS': [rf_metrics[5], dl_metrics[5], svm_metrics[5]],
        'HSS': [rf_metrics[6], dl_metrics[6], svm_metrics[6]],
    })

    # Print the results
    print(results_df)

if __name__ == "__main__":
    main()
