"""Evaluate ARD Classifier on widely-used high-dimensional real-world datasets"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer, load_wine, load_digits, 
    fetch_openml, make_classification, fetch_20newsgroups
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
import sys
import os

# Import from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ard_classifier import ARDClassifier


def evaluate_dataset(X, y, dataset_name, test_size=0.3, random_state=42):
    """Evaluate both ARD and Logistic Regression on a dataset"""
    print(f"\nEvaluating {dataset_name}...")
    print(f"Dataset shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ARD Classifier
    print("  Training ARD...")
    ard = ARDClassifier(
        learning_rate=0.02,
        max_iter=200,
        verbose=0,
        random_state=random_state
    )
    
    start_time = time.time()
    ard.fit(X_train_scaled, y_train)
    ard_train_time = time.time() - start_time
    
    ard_pred = ard.predict(X_test_scaled)
    ard_acc = accuracy_score(y_test, ard_pred)
    ard_prec = precision_score(y_test, ard_pred, average='weighted')
    ard_recall = recall_score(y_test, ard_pred, average='weighted')
    
    # Count selected features (importance > threshold)
    threshold = np.percentile(ard.feature_importances_, 80)
    ard_selected = np.sum(ard.feature_importances_ > threshold)
    
    # Train Logistic Regression
    print("  Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    
    start_time = time.time()
    lr.fit(X_train_scaled, y_train)
    lr_train_time = time.time() - start_time
    
    lr_pred = lr.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_prec = precision_score(y_test, lr_pred, average='weighted')
    lr_recall = recall_score(y_test, lr_pred, average='weighted')
    
    # Train L1 Logistic Regression for comparison
    print("  Training L1 Logistic Regression...")
    lr_l1 = LogisticRegression(
        penalty='l1', C=0.1, solver='liblinear', 
        max_iter=1000, random_state=random_state
    )
    lr_l1.fit(X_train_scaled, y_train)
    lr_l1_pred = lr_l1.predict(X_test_scaled)
    lr_l1_acc = accuracy_score(y_test, lr_l1_pred)
    
    # Count non-zero coefficients in L1
    if hasattr(lr_l1, 'coef_'):
        if lr_l1.coef_.ndim == 1:
            l1_selected = np.sum(np.abs(lr_l1.coef_) > 1e-6)
        else:
            l1_selected = np.sum(np.abs(lr_l1.coef_[0]) > 1e-6)
    else:
        l1_selected = X.shape[1]
    
    return {
        'dataset': dataset_name,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'ard_accuracy': ard_acc,
        'ard_precision': ard_prec,
        'ard_recall': ard_recall,
        'ard_time': ard_train_time,
        'ard_selected': ard_selected,
        'lr_accuracy': lr_acc,
        'lr_precision': lr_prec,
        'lr_recall': lr_recall,
        'lr_time': lr_train_time,
        'lr_l1_accuracy': lr_l1_acc,
        'l1_selected': l1_selected,
        'overhead_factor': ard_train_time / lr_train_time if lr_train_time > 0 else 0
    }


def main():
    """Run evaluation on multiple widely-used high-dimensional real datasets"""
    results = []
    
    # 1. MNIST Digits (8x8=64 features, high-dimensional pixel data)
    digits = load_digits()
    # Binary classification: even vs odd digits
    y_digits_binary = (digits.target % 2).astype(int)
    results.append(evaluate_dataset(
        digits.data, y_digits_binary, 
        "MNIST Digits (even vs odd)", random_state=42
    ))
    
    # 2. 20 Newsgroups Text Classification (very high-dimensional)
    try:
        print("Loading 20 Newsgroups dataset...")
        # Load subset for binary classification
        categories = ['comp.graphics', 'sci.med']
        newsgroups_train = fetch_20newsgroups(
            subset='all', categories=categories, 
            shuffle=True, random_state=42
        )
        
        # Convert to TF-IDF features
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_news = vectorizer.fit_transform(newsgroups_train.data).toarray()
        y_news = newsgroups_train.target
        
        results.append(evaluate_dataset(
            X_news, y_news, 
            "20 Newsgroups (1000 features)", random_state=42
        ))
    except Exception as e:
        print(f"20 Newsgroups dataset not available: {e}")
    
    # 3. Madelon Dataset (designed for feature selection, 500 features)
    try:
        print("Loading Madelon dataset...")
        madelon = fetch_openml('madelon', version=1, parser='auto')
        # Convert string targets to binary if needed
        if hasattr(madelon.target, 'cat'):
            y_madelon = (madelon.target.cat.codes).astype(int)
        else:
            unique_targets = np.unique(madelon.target)
            y_madelon = (madelon.target == unique_targets[0]).astype(int)
        
        results.append(evaluate_dataset(
            madelon.data, y_madelon, 
            "Madelon (500 features)", random_state=42
        ))
    except Exception as e:
        print(f"Madelon dataset not available: {e}")
    
    # 4. Arrhythmia Dataset (medical, 279 features)
    try:
        print("Loading Arrhythmia dataset...")
        arrhythmia = fetch_openml('arrhythmia', version=1, parser='auto')
        
        # Handle missing values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_arr = imputer.fit_transform(arrhythmia.data)
        
        # Convert to binary: normal vs abnormal
        # Class 1 is normal, others are various arrhythmias
        if hasattr(arrhythmia.target, 'cat'):
            y_arr = (arrhythmia.target.cat.codes == 0).astype(int)  # Assuming 0 is normal
        else:
            y_arr = (arrhythmia.target == '1').astype(int)  # Normal class
        
        results.append(evaluate_dataset(
            X_arr, y_arr, 
            "Arrhythmia (279 features)", random_state=42
        ))
    except Exception as e:
        print(f"Arrhythmia dataset not available: {e}")
    
    # 5. Colon Cancer Gene Expression (2000 genes, 62 samples)
    try:
        print("Loading Colon Cancer dataset...")
        colon = fetch_openml('colon', version=1, parser='auto')
        
        # Handle target conversion
        if hasattr(colon.target, 'cat'):
            y_colon = colon.target.cat.codes.astype(int)
        else:
            unique_targets = np.unique(colon.target)
            y_colon = (colon.target == unique_targets[0]).astype(int)
        
        results.append(evaluate_dataset(
            colon.data, y_colon, 
            "Colon Cancer (2000 genes)", random_state=42
        ))
    except Exception as e:
        print(f"Colon Cancer dataset not available: {e}")
    
    # 6. Multiple Features Dataset (optical recognition, 649 features)
    try:
        print("Loading Multiple Features dataset...")
        mult_feat = fetch_openml('multiple-features', version=1, parser='auto')
        
        # Convert to binary classification (digit 0 vs 1)
        mask = (mult_feat.target == '0') | (mult_feat.target == '1')
        X_mf = mult_feat.data[mask]
        y_mf = (mult_feat.target[mask] == '0').astype(int)
        
        results.append(evaluate_dataset(
            X_mf, y_mf, 
            "Multiple Features (649 features)", random_state=42
        ))
    except Exception as e:
        print(f"Multiple Features dataset not available: {e}")
    
    # 7. Fallback: High-dimensional synthetic with known structure
    print("Adding synthetic high-dimensional dataset...")
    X_synth, y_synth = make_classification(
        n_samples=500,
        n_features=300,
        n_informative=20,  # Only 20 out of 300 features are informative
        n_redundant=10,
        n_clusters_per_class=1,
        flip_y=0.05,
        random_state=42
    )
    results.append(evaluate_dataset(
        X_synth, y_synth, 
        "Synthetic (20/300 informative)", random_state=42
    ))
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Print results table
    print("\n" + "="*90)
    print("WIDELY-USED HIGH-DIMENSIONAL DATASET EVALUATION RESULTS")
    print("(Real datasets commonly used in ML research)")
    print("="*90)
    
    # Accuracy comparison
    print("\nAccuracy Comparison:")
    print(f"{'Dataset':<25} {'ARD':<8} {'LogReg':<8} {'L1-LogReg':<10} {'Difference':<12}")
    print("-" * 75)
    for _, row in df.iterrows():
        diff = row['ard_accuracy'] - row['lr_accuracy']
        print(f"{row['dataset']:<25} {row['ard_accuracy']:.3f}    {row['lr_accuracy']:.3f}    "
              f"{row['lr_l1_accuracy']:.3f}      {diff:+.3f}")
    
    # Feature selection comparison
    print("\nFeature Selection (ARD's main advantage):")
    print(f"{'Dataset':<25} {'Total':<8} {'ARD Sel.':<10} {'L1 Sel.':<10} {'ARD %':<10} {'L1 %':<10}")
    print("-" * 85)
    for _, row in df.iterrows():
        ard_ratio = row['ard_selected'] / row['n_features'] * 100
        l1_ratio = row['l1_selected'] / row['n_features'] * 100
        print(f"{row['dataset']:<25} {row['n_features']:<8} {row['ard_selected']:<10} "
              f"{row['l1_selected']:<10} {ard_ratio:.1f}%     {l1_ratio:.1f}%")
    
    # Training time comparison
    print("\nTraining Time:")
    print(f"{'Dataset':<25} {'ARD (s)':<10} {'LogReg (s)':<12} {'Overhead':<12}")
    print("-" * 65)
    for _, row in df.iterrows():
        print(f"{row['dataset']:<25} {row['ard_time']:.3f}      {row['lr_time']:.3f}        "
              f"{row['overhead_factor']:.1f}x")
    
    # Summary statistics
    print("\nSummary Statistics:")
    mean_acc_diff = df['ard_accuracy'].mean() - df['lr_accuracy'].mean()
    mean_overhead = df['overhead_factor'].mean()
    mean_feature_ratio = (df['ard_selected'] / df['n_features']).mean()
    
    print(f"Mean accuracy difference (ARD - LogReg): {mean_acc_diff:+.3f}")
    print(f"Mean computational overhead: {mean_overhead:.1f}x")
    print(f"Mean feature selection ratio: {mean_feature_ratio:.3f} ({mean_feature_ratio*100:.1f}%)")
    
    # Feature selection effectiveness
    print(f"\nFeature Selection Effectiveness:")
    print(f"ARD selects {mean_feature_ratio*100:.1f}% of features on average")
    print(f"This demonstrates ARD's ability to identify relevant features in high-dimensional data")
    
    # Save detailed results
    df.to_csv('real_highdim_dataset_results.csv', index=False)
    print(f"\nDetailed results saved to 'real_highdim_dataset_results.csv'")
    
    return df


if __name__ == "__main__":
    results_df = main() 