# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore')

# show all columns
pd.set_option('display.max_columns', None)

# %%
# Drop index column
df = pd.read_csv('data/published/anonymized_survey+demo+video_matched.csv')

# Drop columns with all NaN values
df = df.dropna(axis=1, how='all')
print('Shape of df:', df.shape)

# Attention Check
attn_check_1 = df['Attention Check'] == 'Pass'
attn_check_2 = df['Vehicle Imagined'] == df['Display']

attn_check_1_2 = attn_check_1 & attn_check_2

df = df[attn_check_1_2]
print('Shape of df after attention check:', df.shape)

df = df.drop(columns=['Attention Check', 'Vehicle Imagined', 'Video ID'])

print('Shape of df after dropping attention check columns and ID:', df.shape)

# %%

# Define the reference category for each variable
reference_categories = {
    'AV Experience': 'No',
    'Primary Mode': 'Motorcycle',
    'ADAS Usage': 'Medium',
    'Tech Confidence': 'Medium',
    'Driving Experience': '5-10',
    'Continent': 'Oceania',
    'Gender': 'Other',
    'Age': '25-45',

    'Ethnicity': 'Unknown',
    'Student': 'Unknown',
    'Employment': 'Unknown',

    'Driver Trust': 'Medium',
    # 'Passenger Safety': 'Medium',
    # 'Passenger Distraction': 'Medium',
    
    'Weather': 'Clear',
    'Scene': 'Other',
    'Light Conditions': 'Normal',
    'Road Type': 'Residential Area',
    
    'Time To Event': '2.5',

    'Ego Fault': 'No',
    'Point of Impact': 'T-bone',

    'Ego Avoidability': 'Potentially Avoidable',
    'Ego Maneuver': 'Go Straight',
    'Ego Speeding': 'No',
    'Ego Violation': 'No',
    
    'Other Maneuver': 'Go Straight',
    'Other Speeding': 'No',
    'Other Body Style': 'Sedan',
    'Other Violation': 'No',

    'VRU Involvement': 'No',
    'Impact Severity': 'Near-Miss',
}
# --- Start of Analysis Script ---

# Define the roles of your columns
TARGET = 'Answer Value'
TREATMENT = 'Display'

categorical_features = list(reference_categories.keys())

# Clean the data: remove '*' and strip whitespace
df.replace(to_replace=r'\*$', value='', regex=True, inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

# Convert all variables to categorical
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category') # Convert to categorical

for col in df.select_dtypes(include=['number']).columns:
    df[col] = df[col].astype('category') # Convert to categorical

# # Convert target and treatment to numeric types
df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
df[TREATMENT] = df[TREATMENT].replace({'No': 0, 'Yes': 1})

# Drop rows where essential columns are missing
df.dropna(subset=[TARGET, TREATMENT], inplace=True)

# Create the feature set X with custom one-hot encoding
X_encoded = pd.get_dummies(df[categorical_features])
for feature, ref_category in reference_categories.items():
    column_to_drop = f"{feature}_{ref_category}"
    if column_to_drop in X_encoded.columns:
        X_encoded.drop(columns=column_to_drop, inplace=True)
    else:
        print(f"Column {column_to_drop} not found in X_encoded")

for col in categorical_features:
    df[col] = df[col].astype('category')

X = X_encoded
Y = df[TARGET]
T = df[TREATMENT]

# Align data to ensure indices match
common_index = X.index
Y = Y.reindex(common_index)
T = T.reindex(common_index)

# # --- Preprocessing from your original code ---
X.columns = X.columns.str.replace(r'[\[\]<]', '', regex=True)
Y = Y.astype(int)

# Keep a copy of the original features for the final analysis
df_original_features = df.loc[X.index, categorical_features]

print("Data preparation complete.")
print(f"Analyzing {len(X)} samples.\n")

# %%
import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import itertools
import warnings

warnings.filterwarnings('ignore')

# --- 1. Load and Prepare Data ---
# Note: Replace this section with loading your actual data.
# >>> LOAD YOUR DATAFRAME `df` HERE <<<
# Example: df = pd.read_csv('your_data_file.csv')

# --- Start of Analysis Script ---

# Define the roles of your columns
TARGET = 'Answer Value'
TREATMENT = 'Display'

categorical_features = list(reference_categories.keys())

# Clean the data: remove '*' and strip whitespace
df.replace(to_replace=r'\*$', value='', regex=True, inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

# Convert all variables to categorical
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category') # Convert to categorical

for col in df.select_dtypes(include=['number']).columns:
    df[col] = df[col].astype('category') # Convert to categorical

# # Convert target and treatment to numeric types
df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
df[TREATMENT] = df[TREATMENT].replace({'No': 0, 'Yes': 1})

# Drop rows where essential columns are missing
df.dropna(subset=[TARGET, TREATMENT], inplace=True)

# Create the feature set X with custom one-hot encoding
X_encoded = pd.get_dummies(df[categorical_features])
for feature, ref_category in reference_categories.items():
    column_to_drop = f"{feature}_{ref_category}"
    if column_to_drop in X_encoded.columns:
        X_encoded.drop(columns=column_to_drop, inplace=True)
    else:
        print(f"Column {column_to_drop} not found in X_encoded")

for col in categorical_features:
    df[col] = df[col].astype('category')

X = X_encoded
Y = df[TARGET]
T = df[TREATMENT]

# Align data to ensure indices match
common_index = X.index
Y = Y.reindex(common_index)
T = T.reindex(common_index)

# # --- Preprocessing from your original code ---
X.columns = X.columns.str.replace(r'[\[\]<]', '', regex=True)
Y = Y.astype(int)


# Drop Driver trust as Mediator
drop_col = ['Driver Trust_High', 'Driver Trust_Low']
X = X.drop(columns=drop_col)
# Keep a copy of the original features for the final analysis
df_original_features = df.loc[X.index, categorical_features]

print("Data preparation complete.")
print(f"Analyzing {len(X)} samples.\n")

# %%
tuned_results_df = pd.read_csv('tuned_results_df.csv')
tuned_results_df

# %% [markdown]
# ## CATE Bootstrap

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from econml.dml import CausalForestDML
from xgboost import XGBClassifier
import ast # To parse stringified parameters if needed
from tqdm import tqdm
import pickle

# Load the json
import json
with open('XGB_best_params.json', 'r') as f:
    XGB_best_params = json.load(f)

print(XGB_best_params)

# --- 1. Initial Setup ---
# Define the models
xgb_outcome_model = XGBClassifier(**XGB_best_params, random_state=42)
xgb_treatment_model = XGBClassifier(random_state=42)

# --- 2. Run the Bootstrap Loop ---
print("Starting bootstrap procedure...")

n_bootstraps = 1000
n_observations = len(Y)

# Initialize a list to store the CATE vectors from each bootstrap iteration
bootstrap_cates = []

for i in tqdm(range(n_bootstraps), desc="Bootstrap Progress"):
    # Create a bootstrap sample by sampling indices with replacement
    boot_indices = np.random.choice(np.arange(n_observations), size=n_observations, replace=True)
    
    X_boot = X.iloc[boot_indices]
    T_boot = T.iloc[boot_indices]
    Y_boot = Y.iloc[boot_indices]

    # Instantiate and fit a new Causal Forest on the bootstrap sample
    forest_boot = CausalForestDML(
        model_y=xgb_outcome_model,
        model_t=xgb_treatment_model,
        discrete_treatment=True,
        discrete_outcome=True, # Set based on your Y variable 
        honest=True,
        n_estimators=100,
        min_samples_leaf=3,
        criterion='het',
        cv=10,
        random_state=i # Change random state for variety
    )

    # Note: Using try-except block to handle potential convergence issues in bootstrap samples
    try:
        forest_boot.fit(Y_boot, T_boot, X=X_boot)
        
        # Estimate CATEs on the original, full dataset X
        cate_estimates_boot = forest_boot.effect(X)
        
        # Store the entire array of CATE estimates for this iteration
        bootstrap_cates.append(cate_estimates_boot)

    except Exception as e:
        print(f"\nWarning: Skipping bootstrap iteration {i+1} due to an error: {e}")
        # Optionally, append NaNs or handle as needed
        bootstrap_cates.append(np.full(n_observations, np.nan))


print("\nBootstrap complete.")

# --- 3. Calculate Point Estimates and Confidence Intervals ---

# Convert the list of arrays into a single 2D NumPy array
# Rows: bootstrap iterations, Columns: observations
bootstrap_cates_array = np.array(bootstrap_cates)

# Save the bootstrap_cates_array to a npz file
np.savez_compressed('bootstrap_cates_array.npz', data=bootstrap_cates_array)

# Load the bootstrap_cates_array from the npz file
with np.load('bootstrap_cates_array.npz') as data:
    bootstrap_cates_array = data['data']

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Calculate the ATE for each bootstrap iteration
bootstrapped_ates = np.nanmean(bootstrap_cates_array, axis=1)

# Calculate the overall ATE point estimate and the confidence interval
ate_point_estimate = np.mean(bootstrapped_ates)
ate_ci_lower_from_cates = np.percentile(bootstrapped_ates, 2.5)
ate_ci_upper_from_cates = np.percentile(bootstrapped_ates, 97.5)


# --- Plotting the Distribution of the ATE ---
plt.figure(figsize=(10, 6))

# Plot the histogram of the bootstrapped ATEs
plt.hist(bootstrapped_ates, bins=60, color='skyblue', edgecolor='black', alpha=0.7, label='Bootstrapped ATE Distribution')

# Add a vertical line for the ATE point estimate
plt.axvline(ate_point_estimate, color='red', linestyle='--', linewidth=2, label=f'ATE Point Estimate ({ate_point_estimate:.4f})')
plt.axvline(0, color='black', linestyle='--', linewidth=2, label=f'No Effect (ATE = 0)')

# Shade the 95% confidence interval
plt.axvspan(ate_ci_lower_from_cates, ate_ci_upper_from_cates, color='red', alpha=0.1, label=f'95% Confidence Interval')

plt.xlim(-0.11, 0.01)

# Add labels and title
# plt.title('Distribution of Average Treatment Effect (ATE) from Bootstrap', fontsize=16)
plt.xlabel('Average Treatment Effect (ATE)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
# plt.savefig('ate_distribution.png')

