#!/usr/bin/env python3
"""
Project 1: Data Preprocessing
CS 535/CS435 - Spring 2025

Datasets:
  - Adult (https://archive.ics.uci.edu/dataset/2/adult)
  - Wine quality (https://archive.ics.uci.edu/dataset/186/wine+quality)

Assignment Tasks:
  Q1. Clean the Adult dataset by removing missing values using the mode.
  Q2. Convert the Adult dataset into a numerical dataset (one-hot encoding).
  Q3. Implement PCA and DCT for dimensionality reduction on:
      - Adult training set (numeric)
      - Adult testing set (numeric)
      - Wine quality red
      - Wine quality white
  Q4. Discuss the relationships between PCA and DCT results and between datasets (to be provided in your report).
  Q5. Remove all categorical attributes from the Adult dataset and reapply PCA and DCT.
  Q6. Generate a 20-dimensional, 100-sample dataset that makes PCA fail.
  Q7. Generate a 20-dimensional, 100-sample dataset that makes DCT fail.
  
All outputs are saved with headers (feature names) and include verification checks.
"""

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.fftpack import dct

# -------------------------------------
# Q1. Clean Adult Dataset (Missing Value Imputation)
# -------------------------------------
def q1_clean_adult_data(adult_train_path, adult_test_path, output_dir, fill_method='mode'):
    """
    Reads raw Adult training and testing files, strips extra whitespace,
    and replaces missing values (indicated by '?') with the mode (most frequent value).
    
    Checks:
      - No missing values remain.
      - No "?" remains in any string column.
    
    Saves the cleaned files with headers.
    """
    os.makedirs(output_dir, exist_ok=True)
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
               'marital_status', 'occupation', 'relationship', 'race', 'sex',
               'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
    
    # Read raw data (adult.test has an extra header row; skip it)
    train_df = pd.read_csv(adult_train_path, header=None, names=columns, na_values='?')
    test_df  = pd.read_csv(adult_test_path, header=None, names=columns, skiprows=1, na_values='?')
    
    # Strip whitespace from string columns.
    for col in train_df.select_dtypes(include=['object']).columns:
        train_df[col] = train_df[col].str.strip()
    for col in test_df.select_dtypes(include=['object']).columns:
        test_df[col] = test_df[col].str.strip()
    
    # Replace missing values using the mode for each column.
    if fill_method == 'mode':
        for col in train_df.columns:
            if train_df[col].isnull().sum() > 0:
                mode_val = train_df[col].mode()[0]
                train_df[col].fillna(mode_val, inplace=True)
        for col in test_df.columns:
            if test_df[col].isnull().sum() > 0:
                mode_val = test_df[col].mode()[0]
                test_df[col].fillna(mode_val, inplace=True)
    else:
        train_df.dropna(inplace=True)
        test_df.dropna(inplace=True)
    
    # Save cleaned files with headers.
    train_out = os.path.join(output_dir, 'adult_train_filled.csv')
    test_out  = os.path.join(output_dir, 'adult_test_filled.csv')
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    
    # Verification Checks
    clean_train = pd.read_csv(train_out)
    clean_test  = pd.read_csv(test_out)
    assert clean_train.isnull().sum().sum() == 0, "Q1 Check Failed: Missing values in training data."
    assert clean_test.isnull().sum().sum() == 0, "Q1 Check Failed: Missing values in testing data."
    for col in clean_train.select_dtypes(include=['object']).columns:
        assert (~clean_train[col].str.contains(r'\?')).all(), f"Q1 Check Failed: '?' found in training column {col}."
    for col in clean_test.select_dtypes(include=['object']).columns:
        assert (~clean_test[col].str.contains(r'\?')).all(), f"Q1 Check Failed: '?' found in testing column {col}."
    print("[Q1] Cleaned Adult data verified: no missing values or '?' remain.")
    print("     Saved at:", train_out, "and", test_out)
    return train_out, test_out

# -------------------------------------
# Q2. Convert Adult Dataset to Numeric
# -------------------------------------
def q2_convert_adult_to_numeric(adult_train_filled_path, adult_test_filled_path, output_dir):
    """
    Converts the cleaned Adult datasets into numerical format by applying one-hot encoding
    on all categorical attributes. The test set is reindexed to match the training set.
    
    Checks:
      - No object-type columns remain.
    
    Saves the numeric files with headers.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_train = pd.read_csv(adult_train_filled_path)
    df_test  = pd.read_csv(adult_test_filled_path)
    
    cat_cols = df_train.select_dtypes(include=['object']).columns.tolist()
    df_train_num = pd.get_dummies(df_train, columns=cat_cols, drop_first=True)
    df_test_num  = pd.get_dummies(df_test, columns=cat_cols, drop_first=True)
    df_test_num = df_test_num.reindex(columns=df_train_num.columns, fill_value=0)
    
    train_out = os.path.join(output_dir, 'adult_train_numeric.csv')
    test_out  = os.path.join(output_dir, 'adult_test_numeric.csv')
    df_train_num.to_csv(train_out, index=False)
    df_test_num.to_csv(test_out, index=False)
    
    # Verification Checks
    num_train = pd.read_csv(train_out)
    num_test  = pd.read_csv(test_out)
    assert num_train.select_dtypes(include=['object']).empty, "Q2 Check Failed: Object columns in training numeric file."
    assert num_test.select_dtypes(include=['object']).empty, "Q2 Check Failed: Object columns in testing numeric file."
    print("[Q2] Numeric Adult data verified: all columns are numeric.")
    print("     Saved at:", train_out, "and", test_out)
    return train_out, test_out

# -------------------------------------
# Q3. Dimensionality Reduction (PCA and DCT)
# -------------------------------------
def custom_pca(data, n_components=10):
    """
    Applies PCA using scikit-learn. Uses n_components = min(desired, available features).
    Returns the transformed data and the PCA model.
    """
    n_comp = min(n_components, data.shape[1])
    pca_model = PCA(n_components=n_comp)
    transformed = pca_model.fit_transform(data)
    return transformed, pca_model

def custom_dct(data, n_coeff=10):
    """
    Applies the Discrete Cosine Transform (DCT-II) on each row of the data.
    Returns the first n_coeff coefficients.
    """
    dct_result = dct(data, type=2, norm='ortho', axis=1)
    return dct_result[:, :n_coeff]

def q3_apply_dim_reduction(adult_train_numeric_path, adult_test_numeric_path, wine_red_path, wine_white_path, output_dir, n_components=10):
    """
    Applies PCA and DCT on four datasets:
      - Adult training (numeric)
      - Adult testing (numeric)
      - Wine quality red (CSV, delimiter ';')
      - Wine quality white (CSV, delimiter ';')
      
    Saves PCA outputs with headers "PC1", "PC2", ... and DCT outputs with headers "DCT1", "DCT2", ...
    Verifies that the saved files have the proper header names.
    """
    os.makedirs(output_dir, exist_ok=True)
    def process_dataset(path, ds_name):
        if ds_name.startswith("wine"):
            df = pd.read_csv(path, sep=";")
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
        else:
            df = pd.read_csv(path)
        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.shape[1] == 0:
            print(f"[Q3] {ds_name}: No numeric columns. Skipping.")
            return
        data = df_numeric.values
        # PCA
        pca_data, pca_model = custom_pca(data, n_components)
        pca_df = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(pca_data.shape[1])])
        pca_file = os.path.join(output_dir, f"{ds_name}_pca.csv")
        pca_df.to_csv(pca_file, index=False)
        # DCT
        dct_data = custom_dct(data, n_coeff=n_components)
        dct_df = pd.DataFrame(dct_data, columns=[f"DCT{i+1}" for i in range(dct_data.shape[1])])
        dct_file = os.path.join(output_dir, f"{ds_name}_dct.csv")
        dct_df.to_csv(dct_file, index=False)
        # Verification Checks
        saved_pca = pd.read_csv(pca_file)
        saved_dct = pd.read_csv(dct_file)
        assert all(col.startswith("PC") for col in saved_pca.columns), f"Q3 Check Failed: Improper PCA headers in {ds_name}."
        assert all(col.startswith("DCT") for col in saved_dct.columns), f"Q3 Check Failed: Improper DCT headers in {ds_name}."
        print(f"[Q3] {ds_name}: PCA shape {saved_pca.shape}, DCT shape {saved_dct.shape}.")
        print("     PCA explained variance ratio:", pca_model.explained_variance_ratio_)
    
    process_dataset(adult_train_numeric_path, "adult_train")
    process_dataset(adult_test_numeric_path, "adult_test")
    process_dataset(wine_red_path, "wine_red")
    process_dataset(wine_white_path, "wine_white")

# -------------------------------------
# Q4. Discussion (Placeholder)
# -------------------------------------
def q4_discussion():
    """
    Q4: Please include in your report a detailed discussion of the relationships between the PCA and DCT
        dimensionality reduction results and how the Adult and Wine datasets compare.
    """
    print("[Q4] Please refer to your written report for a detailed discussion of PCA vs. DCT results.")

# -------------------------------------
# Q5. Remove Categorical Attributes and Reapply PCA/DCT
# -------------------------------------
def q5_remove_categorical_and_apply_dim_reduction(adult_train_filled_path, adult_test_filled_path, output_dir, n_components=10):
    """
    Removes all categorical (object) columns from the cleaned Adult datasets and re-applies PCA and DCT.
    
    Saves:
      - Numeric-only versions of the Adult datasets.
      - PCA and DCT outputs (with headers) from the numeric-only data.
      
    Checks:
      - The numeric-only files contain no object columns.
      - The PCA/DCT output files have proper headers.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_df = pd.read_csv(adult_train_filled_path)
    test_df  = pd.read_csv(adult_test_filled_path)
    train_numeric_only = train_df.select_dtypes(exclude=['object']).copy()
    test_numeric_only  = test_df.select_dtypes(exclude=['object']).copy()
    
    train_only_out = os.path.join(output_dir, "adult_train_numeric_only.csv")
    test_only_out  = os.path.join(output_dir, "adult_test_numeric_only.csv")
    train_numeric_only.to_csv(train_only_out, index=False)
    test_numeric_only.to_csv(test_only_out, index=False)
    
    # Check numeric-only files
    only_train = pd.read_csv(train_only_out)
    only_test  = pd.read_csv(test_only_out)
    assert only_train.select_dtypes(include=['object']).empty, "Q5 Check Failed: Training numeric-only file contains object columns."
    assert only_test.select_dtypes(include=['object']).empty, "Q5 Check Failed: Testing numeric-only file contains object columns."
    print("[Q5] Numeric-only Adult data verified: no categorical attributes remain.")
    
    def apply_and_save(df, prefix):
        data = df.values
        pca_data, _ = custom_pca(data, n_components)
        dct_data = custom_dct(data, n_coeff=n_components)
        pca_df = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(pca_data.shape[1])])
        dct_df = pd.DataFrame(dct_data, columns=[f"DCT{i+1}" for i in range(dct_data.shape[1])])
        pca_file = os.path.join(output_dir, f"{prefix}_pca.csv")
        dct_file = os.path.join(output_dir, f"{prefix}_dct.csv")
        pca_df.to_csv(pca_file, index=False)
        dct_df.to_csv(dct_file, index=False)
        saved_pca = pd.read_csv(pca_file)
        saved_dct = pd.read_csv(dct_file)
        assert all(col.startswith("PC") for col in saved_pca.columns), f"Q5 Check Failed: Improper PCA headers in {prefix}."
        assert all(col.startswith("DCT") for col in saved_dct.columns), f"Q5 Check Failed: Improper DCT headers in {prefix}."
        print(f"[Q5] {prefix}: PCA shape {saved_pca.shape}, DCT shape {saved_dct.shape}.")
    
    apply_and_save(train_numeric_only, "adult_train_numeric_only")
    apply_and_save(test_numeric_only, "adult_test_numeric_only")
    print("[Q5] PCA and DCT applied on numeric-only Adult data verified.")
    
# -------------------------------------
# Q6. Generate Dataset That Makes PCA Fail
# -------------------------------------
def q6_generate_pca_fail_dataset(output_dir, n_samples=100, n_features=20):
    """
    Generates a dataset with zero variance (all rows identical) which causes PCA to fail.
    
    Saves the dataset with feature names.
    
    Check:
      - All rows are identical.
    """
    os.makedirs(output_dir, exist_ok=True)
    row = np.random.rand(1, n_features)
    data = np.tile(row, (n_samples, 1))
    df_fail = pd.DataFrame(data, columns=[f"F{i+1}" for i in range(n_features)])
    fail_path = os.path.join(output_dir, "pca_fail_dataset.csv")
    df_fail.to_csv(fail_path, index=False)
    
    loaded = pd.read_csv(fail_path)
    assert loaded.nunique().max() == 1, "Q6 Check Failed: Not all rows in the PCA fail dataset are identical."
    print("[Q6] PCA fail dataset verified (all rows identical).")
    print("     Saved at:", fail_path)
    return fail_path

# -------------------------------------
# Q7. Generate Dataset That Makes DCT Fail
# -------------------------------------
def q7_generate_dct_fail_dataset(output_dir, n_samples=100, n_features=20):
    """
    Generates an all-zeros dataset which yields trivial DCT results.
    
    Saves the dataset with feature names.
    
    Check:
      - All values are zero.
    """
    os.makedirs(output_dir, exist_ok=True)
    data = np.zeros((n_samples, n_features))
    df_fail = pd.DataFrame(data, columns=[f"F{i+1}" for i in range(n_features)])
    fail_path = os.path.join(output_dir, "dct_fail_dataset.csv")
    df_fail.to_csv(fail_path, index=False)
    
    loaded = pd.read_csv(fail_path)
    assert (loaded.values == 0).all(), "Q7 Check Failed: Not all values in the DCT fail dataset are zero."
    print("[Q7] DCT fail dataset verified (all zeros).")
    print("     Saved at:", fail_path)
    return fail_path

# -------------------------------------
# MAIN SCRIPT
# -------------------------------------
def main():
    # Define the root directory for data.
    root_dir = "data"
    
    # Paths for Adult dataset.
    adult_dir = os.path.join(root_dir, "raw_data", "adult")
    adult_train_file = os.path.join(adult_dir, "adult.data")
    adult_test_file  = os.path.join(adult_dir, "adult.test")
    
    # Paths for Wine quality dataset.
    wine_dir = os.path.join(root_dir, "raw_data", "wine+quality")
    wine_red_file   = os.path.join(wine_dir, "winequality-red.csv")
    wine_white_file = os.path.join(wine_dir, "winequality-white.csv")
    
    # Define output directories for each task.
    q1_out = os.path.join(root_dir, "q1_cleaned")
    q2_out = os.path.join(root_dir, "q2_numeric")
    q3_out = os.path.join(root_dir, "q3_dimred")
    q5_out = os.path.join(root_dir, "q5_no_categorical")
    q6_out = os.path.join(root_dir, "q6_pca_fail")
    q7_out = os.path.join(root_dir, "q7_dct_fail")
    
    # Q1. Clean Adult dataset.
    train_filled, test_filled = q1_clean_adult_data(adult_train_file, adult_test_file, q1_out, fill_method='mode')
    
    # Q2. Convert cleaned Adult dataset to numeric.
    train_numeric, test_numeric = q2_convert_adult_to_numeric(train_filled, test_filled, q2_out)
    
    # Q3. Apply PCA and DCT on four datasets.
    q3_apply_dim_reduction(train_numeric, test_numeric, wine_red_file, wine_white_file, q3_out, n_components=10)
    
    # Q4. Discussion placeholder.
    q4_discussion()
    
    # Q5. Remove categorical attributes from Adult data and reapply PCA/DCT.
    q5_remove_categorical_and_apply_dim_reduction(train_filled, test_filled, q5_out, n_components=10)
    
    # Q6. Generate a dataset that makes PCA fail.
    q6_generate_pca_fail_dataset(q6_out, n_samples=100, n_features=20)
    
    # Q7. Generate a dataset that makes DCT fail.
    q7_generate_dct_fail_dataset(q7_out, n_samples=100, n_features=20)
    
    print("\nAll steps completed and verified! Please check the output files in their respective directories.")

if __name__ == '__main__':
    main()
