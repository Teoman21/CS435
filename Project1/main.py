# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.fftpack import dct
from sklearn.preprocessing import StandardScaler


# For Jupyter: show plots inline
#%matplotlib inline

# Set seaborn style for plots
sns.set(style="whitegrid")


# Define file paths (adjust as needed)
adult_train_path = "data/raw_data/adult/adult.data"
adult_test_path  = "data/raw_data/adult/adult.test"

# Define column names per dataset documentation
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship', 'race', 'sex',
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

# Read raw data with '?' as NA, skipping header row in test file.
train_df = pd.read_csv(adult_train_path, header=None, names=columns, na_values='?')
test_df  = pd.read_csv(adult_test_path, header=None, names=columns, skiprows=1, na_values='?')

# Strip whitespace from object columns.
for col in train_df.select_dtypes(include=['object']).columns:
    train_df[col] = train_df[col].str.strip()
for col in test_df.select_dtypes(include=['object']).columns:
    test_df[col] = test_df[col].str.strip()

# Explicitly replace any remaining "?" with NaN (in case of inconsistent formatting)
for col in train_df.select_dtypes(include=['object']).columns:
    train_df[col] = train_df[col].replace("?", np.nan)
for col in test_df.select_dtypes(include=['object']).columns:
    test_df[col] = test_df[col].replace("?", np.nan)

# Before cleaning: plot count of missing values per column
missing_train = train_df.isnull().sum()
plt.figure(figsize=(10,4))
sns.barplot(x=missing_train.index, y=missing_train.values, palette="viridis")
plt.title("Missing Values per Column in Adult Train (Before Cleaning)")
plt.xticks(rotation=45)
plt.show()

# Replace missing values with the mode for each column
for col in train_df.columns:
    if train_df[col].isnull().sum() > 0:
        mode_val = train_df[col].mode()[0]
        train_df[col].fillna(mode_val, inplace=True)
for col in test_df.columns:
    if test_df[col].isnull().sum() > 0:
        mode_val = test_df[col].mode()[0]
        test_df[col].fillna(mode_val, inplace=True)

# Verification: Check no missing values and no literal '?' for train_df only
assert train_df.isnull().sum().sum() == 0, "Missing values remain in training data."
for col in train_df.select_dtypes(include=['object']).columns:
    assert (~train_df[col].str.contains(r'\?')).all(), f"'?' remains in training column {col}"

print("Q1: Adult data cleaned successfully (no missing values or '?' remain).")

# Save cleaned files (for later use)
os.makedirs("data/q1_cleaned", exist_ok=True)
train_df.to_csv("data/q1_cleaned/adult_train_filled.csv", index=False)
test_df.to_csv("data/q1_cleaned/adult_test_filled.csv", index=False)

# Define file paths for the cleaned Adult dataset (from Q1)
adult_train_clean_path = "data/q1_cleaned/adult_train_filled.csv"
adult_test_clean_path  = "data/q1_cleaned/adult_test_filled.csv"

# Read the cleaned datasets
df_train = pd.read_csv(adult_train_clean_path)
df_test  = pd.read_csv(adult_test_clean_path)

# Identify categorical attributes (object columns)
categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
print("Categorical attributes processed:", categorical_cols)

# Apply one-hot encoding on all categorical attributes (drop_first=True to avoid redundancy)
df_train_numeric = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True)
df_test_numeric  = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)

# Reindex test set to ensure it has the same columns as training set
df_test_numeric = df_test_numeric.reindex(columns=df_train_numeric.columns, fill_value=0)

# Verification: Ensure that the resulting datasets have no object columns.
assert df_train_numeric.select_dtypes(include=['object']).empty, "Numeric conversion failed: Training dataset has object columns."
assert df_test_numeric.select_dtypes(include=['object']).empty, "Numeric conversion failed: Testing dataset has object columns."
print("Q2: Conversion to numeric format successful. All categorical attributes have been processed.")

# Save the numeric datasets
os.makedirs("data/q2_numeric", exist_ok=True)
df_train_numeric.to_csv("data/q2_numeric/adult_train_numeric.csv", index=False)
df_test_numeric.to_csv("data/q2_numeric/adult_test_numeric.csv", index=False)

# Display a sample of the numeric dataset columns
df_train_numeric.head()



# Set parameters
n_components = 30   # Desired number of components/coefficients.adjusted 20 10 50 etc..
use_scaling = True  # Set to True to standardize the data before applying PCA.

# -----------------------------
# Manual PCA Implementation using SVD
# -----------------------------
def custom_pca_manual(data, n_components=n_components):
    """
    Manually applies PCA using SVD.
    
    Steps:
      1. Center the data.
      2. Compute SVD.
      3. Project data onto the top n_components.
      4. Compute explained variance ratio from the singular values.
    
    Returns:
      - X_reduced: Data projected onto the top n_components.
      - explained_variance_ratio: Array of variance ratios for each component.
    """
    # Center the data
    X_mean = np.mean(data, axis=0)
    X_centered = data - X_mean
    
    # Compute SVD (full_matrices=False ensures U, S, VT have optimal shapes)
    U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
    
    # Project data onto top n_components (if n_components > rank, it will return rank many)
    n_comp = min(n_components, data.shape[1])
    X_reduced = np.dot(X_centered, VT.T[:, :n_comp])
    
    # Compute explained variance ratio
    total_variance = np.sum(S**2)
    explained_variance = S[:n_comp]**2
    explained_variance_ratio = explained_variance / total_variance
    
    return X_reduced, explained_variance_ratio

# -----------------------------
# Manual DCT Implementation (DCT-II)
# -----------------------------
def custom_dct_manual(data, n_coeff=n_components):
    """
    Manually applies the Discrete Cosine Transform (DCT-II) on each row.
    
    For a row vector x[0..M-1], the DCT-II is defined as:
        X[k] = alpha(k) * sum_{j=0}^{M-1} x[j] * cos(pi*(j+0.5)*k/M)
    where alpha(0)=sqrt(1/M) and alpha(k)=sqrt(2/M) for k > 0.
    
    Returns:
      - The first n_coeff coefficients for each row.
    """
    N, M = data.shape
    # Precompute cosine basis matrix for efficiency.
    j = np.arange(M)
    X_dct = np.empty((N, M))
    for k in range(M):
        alpha = np.sqrt(1/M) if k == 0 else np.sqrt(2/M)
        cosine_term = np.cos(np.pi * (j + 0.5) * k / M)
        X_dct[:, k] = alpha * np.sum(data * cosine_term, axis=1)
    return X_dct[:, :n_coeff]

# -----------------------------
# Q3: Processing and Visualization Function
# -----------------------------
# Define file paths for numeric Adult data (from Q2)
adult_train_numeric_path = "data/q2_numeric/adult_train_numeric.csv"
adult_test_numeric_path  = "data/q2_numeric/adult_test_numeric.csv"

# Define file paths for Wine quality datasets (use ';' as delimiter)
wine_red_path = "data/raw_data/wine+quality/winequality-red.csv"
wine_white_path = "data/raw_data/wine+quality/winequality-white.csv"

# Create output directory for Q3 results.
os.makedirs("data/q3_dimred", exist_ok=True)

def process_dataset(path, ds_name, n_components=n_components, use_scaling=use_scaling):
    # For wine datasets, use the proper delimiter and convert all columns to numeric.
    if ds_name.startswith("wine"):
        df = pd.read_csv(path, sep=";")
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
    else:
        df = pd.read_csv(path)
    # Select numeric columns.
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.shape[1] == 0:
        print(f"[Q3] {ds_name}: No numeric columns found. Skipping.")
        return
    data = df_numeric.values
    
    # Optionally standardize the data.
    if use_scaling:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        print(f"[Q3] {ds_name}: Data standardized.")
    
    # Apply manual PCA.
    pca_data, exp_var_ratio = custom_pca_manual(data, n_components)
    effective_components = pca_data.shape[1]
    if effective_components < n_components:
        print(f"[Q3 Warning] {ds_name}: Effective components = {effective_components} (requested {n_components}).")
    pca_df = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(effective_components)])
    pca_file = os.path.join("data/q3_dimred", f"{ds_name}_pca.csv")
    pca_df.to_csv(pca_file, index=False)
    
    # Apply manual DCT.
    dct_data = custom_dct_manual(data, n_coeff=n_components)
    dct_df = pd.DataFrame(dct_data, columns=[f"DCT{i+1}" for i in range(dct_data.shape[1])])
    dct_file = os.path.join("data/q3_dimred", f"{ds_name}_dct.csv")
    dct_df.to_csv(dct_file, index=False)
    
    # Report reduction results.
    print(f"[Q3] {ds_name}:")
    print("     PCA reduced shape:", pca_df.shape)
    print("     DCT reduced shape:", dct_df.shape)
    print("     Manual PCA explained variance ratio:", exp_var_ratio)
    
    # Plot scree plot: individual explained variance.
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(exp_var_ratio)+1), exp_var_ratio, 'o-', color='teal')
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title(f"{ds_name} - PCA Scree Plot")
    plt.xticks(np.arange(1, len(exp_var_ratio)+1))
    plt.show()
    
    # Plot cumulative explained variance.
    cum_explained = np.cumsum(exp_var_ratio)
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(cum_explained)+1), cum_explained, 'o-', color='purple')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"{ds_name} - Cumulative Explained Variance")
    plt.xticks(np.arange(1, len(cum_explained)+1))
    plt.axhline(y=0.90, color='red', linestyle='--', label="90% Threshold")
    plt.legend()
    plt.show()
    
    # Plot scatter plot of the first two PCA components.
    if pca_data.shape[1] >= 2:
        plt.figure(figsize=(6,5))
        sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], alpha=0.5)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"{ds_name} - First Two PCA Components")
        plt.show()
    
    # Additional plots for discussion:
    # Scatter plot: PC1 vs. DCT1.
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=pca_df["PC1"], y=dct_df["DCT1"], alpha=0.5)
    plt.xlabel("PCA Component 1")
    plt.ylabel("DCT Coefficient 1")
    plt.title(f"{ds_name} - PC1 vs DCT1")
    plt.show()
    
    # Boxplots: Distribution of all PCA components and all DCT coefficients.
    pca_melt = pca_df.melt(var_name="PC", value_name="Value")
    dct_melt = dct_df.melt(var_name="DCT", value_name="Value")
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.boxplot(x="PC", y="Value", data=pca_melt, palette="Pastel1")
    plt.title(f"{ds_name} - Distribution of PCA Components")
    plt.xticks(rotation=45)
    plt.subplot(1,2,2)
    sns.boxplot(x="DCT", y="Value", data=dct_melt, palette="Pastel2")
    plt.title(f"{ds_name} - Distribution of DCT Coefficients")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Process each dataset.
process_dataset(adult_train_numeric_path, "adult_train", n_components, use_scaling)
process_dataset(adult_test_numeric_path, "adult_test", n_components, use_scaling)
process_dataset(wine_red_path, "wine_red", n_components, use_scaling)
process_dataset(wine_white_path, "wine_white", n_components, use_scaling)


# Define relative file paths for numeric Adult data (from Q2).
adult_train_numeric_path = "../Project1/data/q1_cleaned/adult_train_filled.csv"
adult_test_numeric_path  = "../Project1/data/q1_cleaned/adult_test_filled.csv"

# Create output directory for Q5 results.
output_dir = "../Project1/data/q5"
os.makedirs(output_dir, exist_ok=True)

def process_adult_numeric(path, ds_name, n_components=10, use_scaling=True):
    """
    Loads the numeric adult dataset, applies PCA and DCT, and saves the results.
    """
    # Load the dataset.
    df = pd.read_csv(path)
    
    # Ensure only numeric columns are kept.
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.shape[1] == 0:
        print(f"[Q5] {ds_name}: No numeric columns found. Skipping.")
        return
    
    data = df_numeric.values
    
    # Standardize the data if required.
    if use_scaling:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        print(f"[Q5] {ds_name}: Data standardized.")
    
    # Apply custom PCA.
    pca_data, exp_var_ratio = custom_pca_manual(data, n_components)
    pca_df = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(pca_data.shape[1])])
    
    # Apply custom DCT.
    dct_data = custom_dct_manual(data, n_coeff=n_components)
    dct_df = pd.DataFrame(dct_data, columns=[f"DCT{i+1}" for i in range(dct_data.shape[1])])
    
    # Save outputs into the q5 folder.
    pca_file = os.path.join(output_dir, f"{ds_name}_pca.csv")
    dct_file = os.path.join(output_dir, f"{ds_name}_dct.csv")
    pca_df.to_csv(pca_file, index=False)
    dct_df.to_csv(dct_file, index=False)
    
    # Print summary.
    print(f"[Q5] {ds_name} - PCA shape: {pca_df.shape}, DCT shape: {dct_df.shape}")
    print(f"[Q5] {ds_name} - PCA explained variance ratio:\n{exp_var_ratio}\n")
    
    # Plot Scree Plot for PCA.
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(exp_var_ratio)+1), exp_var_ratio, 'o-', color='teal')
    plt.title(f"{ds_name} - PCA Scree Plot (Numeric Only)")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.show()
    
    # Plot PC1 vs. DCT1.
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=pca_df["PC1"], y=dct_df["DCT1"], alpha=0.5)
    plt.title(f"{ds_name} - PC1 vs DCT1 (Numeric Only)")
    plt.xlabel("PC1")
    plt.ylabel("DCT1")
    plt.show()

# Process both the adult training and test sets.
process_adult_numeric(adult_train_numeric_path, "adult_train_numeric", n_components=10, use_scaling=True)
process_adult_numeric(adult_test_numeric_path,  "adult_test_numeric",  n_components=10, use_scaling=True)


# Define parameters.
n_samples = 100      # 100 samples
n_features = 20      # 20 dimensions per sample
n_components = 20    # We'll use 20 components/coefficients

# Create output directories.
os.makedirs("data/q6", exist_ok=True)  # For PCA failure dataset
os.makedirs("data/q7", exist_ok=True)  # For DCT failure dataset

#######################################
# Task 1: Generate a PCA Failure Dataset
#######################################
# Use isotropic Gaussian noise so all singular values ≈ equal (power‑law violated)
pca_fail_data = np.random.randn(n_samples, n_features)

# Apply the custom PCA function.
pca_fail_output, pca_fail_exp_var_ratio = custom_pca_manual(pca_fail_data, n_components)
print("PCA output on failing dataset (first 5 rows):\n", pca_fail_output[:5])
print("PCA explained variance ratio on failing dataset:\n", pca_fail_exp_var_ratio)

# Save the PCA failure dataset.
df_pca_fail = pd.DataFrame(pca_fail_data, columns=[f"Feature{i+1}" for i in range(n_features)])
df_pca_fail.to_csv("data/q6/pca_fail_data.csv", index=False)

# Evidence for Q6: plot singular values for entire dataset
centered = pca_fail_data - pca_fail_data.mean(axis=0)
_, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
print("Singular values (all 20):", singular_values)
plt.figure()
plt.plot(np.arange(1, len(singular_values)+1), singular_values, marker='o')
plt.xlabel("Component index")
plt.ylabel("Singular value")
plt.title("PCA Failure Dataset — Singular Values (All Samples)")
plt.grid(True)
plt.savefig("data/q6/pca_failure_singular_values.png")
plt.close()

#######################################
# Task 2: Generate a DCT Failure Dataset
#######################################
# Strategy:
alternating_pattern = np.array([1 if i % 2 == 0 else -1 for i in range(n_features)], dtype=float)
dct_fail_data = np.tile(alternating_pattern, (n_samples, 1))
dct_fail_data += 0.05 * np.random.randn(n_samples, n_features)

# Apply the custom DCT function.
dct_fail_output = custom_dct_manual(dct_fail_data, n_coeff=n_components)
print("DCT output on failing dataset (first sample):\n", dct_fail_output[0])

# Test energy
total_energy = np.sum(dct_fail_output**2)
low_freq_energy = np.sum(dct_fail_output[:, :5]**2)
energy_fraction = low_freq_energy / total_energy if total_energy > 0 else 0
print(f"Fraction of energy in first five DCT coefficients: {energy_fraction:.4f}")

# Save the DCT failure dataset.
df_dct_fail = pd.DataFrame(dct_fail_data, columns=[f"Feature{i+1}" for i in range(n_features)])
df_dct_fail.to_csv("data/q7/dct_fail_data.csv", index=False)

# Evidence for Q7: plot total energy per coefficient across all samples
coeff_energy = np.sum(dct_fail_output**2, axis=0)
plt.figure()
plt.stem(np.arange(1, n_components+1), coeff_energy)
plt.xlabel("DCT Coefficient Index")
plt.ylabel("Total Energy")
plt.title("DCT Failure Dataset — Energy per Coefficient (All Samples)")
plt.grid(True)
plt.savefig("data/q7/dct_failure_all_samples.png")
plt.close()
