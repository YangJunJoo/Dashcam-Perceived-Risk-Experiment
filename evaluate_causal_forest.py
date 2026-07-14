# %%
import pandas as pd
import numpy as np

from xgboost import XGBRegressor, XGBClassifier
from econml.dml import CausalForestDML
import itertools
import ast
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import itertools
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# %% [markdown]
# # Data preparation

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
df[TREATMENT] = df[TREATMENT].replace({'HDV': 0, 'AV': 1})

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

# drop Driver trust
drop_col = ['Driver Trust_High', 'Driver Trust_Low']
X = X.drop(columns=drop_col)
# Keep a copy of the original features for the final analysis
df_original_features = df.loc[X.index, categorical_features]

print("Data preparation complete.")
print(f"Analyzing {len(X)} samples.\n")

# %%
df['Display']

# %% [markdown]
# # Prediction model grid search

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tqdm.auto import tqdm

# This cell assumes 'X' and 'Y' are already defined.


# 1. Define the 10-fold cross-validation strategy
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# 2. Define the expanded parameter grids for each model
# The original search space had 90 total combinations.
# This new space has 1,072 combinations, an ~11.9x increase.
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 400],
        'max_depth': [10, 20, 50, None],
        'min_samples_leaf': [1, 2, 5],
        'min_samples_split': [2, 5, 10],
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 8, 10],
        'subsample': [0.8, 1.0]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 400, 800],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [2, 4, 8, 16],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [20, 31, 50, 70],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
}


# 3. Define the base models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', enable_categorical=True),
    'LightGBM': LGBMClassifier(random_state=42, verbosity=-1)
}


# 4. Loop through models, perform GridSearchCV, and store results
tuned_results_list = []
print("🚀 Starting hyperparameter tuning for each model...")

for name, model in tqdm(models.items(), desc="Tuning Models"):
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        cv=kfold,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=3 # Set to 0 to disable inner progress messages for a cleaner tqdm bar
    )
    
    # Fit the grid search to the data
    grid_search.fit(pd.concat([X, T], axis=1), Y)
    
    # Store the best results
    tuned_results_list.append({
        'Model': name,
        'Best Score (AUC)': grid_search.best_score_,
        'Best Parameters': grid_search.best_params_
    })

# 5. Display the tuning results in a clean table
tuned_results_df = pd.DataFrame(tuned_results_list)

print("\n\n--- 🏆 Hyperparameter Tuning Results ---")
print(tuned_results_df)

# For a cleaner look at just the parameters and scores:
for index, row in tuned_results_df.iterrows():
    print(f"\nModel: {row['Model']}")
    print(f"Best Score (AUC): {row['Best Score (AUC)']:.4f}") 
    print(f"Best Parameters: {row['Best Parameters']}")

# %%
tuned_results_df

# %%
tuned_results_df.to_csv('data/tuned_results_df.csv', index=False)

# %%
tuned_results_df = pd.read_csv('data/tuned_results_df.csv')

# %%
# --- Comprehensive Model Evaluation with Tuned Parameters ---
# Copy this cell into your Jupyter notebook after running hyperparameter tuning

import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the tuned results if not already loaded
if 'tuned_results_df' not in locals():
    tuned_results_df = pd.read_csv('tuned_results_df.csv')

print("🔍 Extracting best parameters from tuned_results_df...")

# Extract best parameters and create optimized models
best_models = {}

for _, row in tuned_results_df.iterrows():
    model_name = row['Model']
    # Convert string representation of dictionary to actual dictionary
    try:
        best_params = ast.literal_eval(row['Best Parameters'])
    except:
        best_params = row['Best Parameters']
    
    print(f"\n📊 {model_name}:")
    print(f"   Best Score (AUC): {row['Best Score (AUC)']:.4f}")
    print(f"   Best Parameters: {best_params}")
    
    # Create model with best parameters
    if model_name == 'Random Forest':
        best_models[model_name] = RandomForestClassifier(
            **best_params,
            random_state=42
        )
    elif model_name == 'Gradient Boosting':
        best_models[model_name] = GradientBoostingClassifier(
            **best_params,
            random_state=42
        )
    elif model_name == 'XGBoost':
        # Add XGBoost specific parameters
        xgb_params = best_params.copy()
        xgb_params.update({
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        })
        best_models[model_name] = XGBClassifier(
            **xgb_params,
            random_state=42
        )
    elif model_name == 'LightGBM':
        # Add LightGBM specific parameters
        lgb_params = best_params.copy()
        lgb_params.update({
            'verbosity': -1
        })
        best_models[model_name] = LGBMClassifier(
            **lgb_params,
            random_state=42
        )

# Define comprehensive scoring metrics
scoring_metrics = {
    'accuracy': 'accuracy',
    'precision_macro': 'precision_macro',
    'recall_macro': 'recall_macro', 
    'f1_macro': 'f1_macro',
    'roc_auc_ovr': 'roc_auc_ovr',
    'pr_auc': 'average_precision'
}

# 10-fold stratified cross-validation for robust evaluation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("\n🎯 Comprehensive Model Evaluation with Tuned Hyperparameters")
print("=" * 60)

# Store results for comparison
evaluation_results = []

for model_name, model in best_models.items():
    print(f"\n📊 Evaluating {model_name}...")
    
    # Perform cross-validation with all metrics
    cv_results = cross_validate(
        model, X, Y, 
        cv=cv, 
        scoring=scoring_metrics,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Calculate mean and std for each metric
    results = {
        'Model': model_name,
        'Accuracy': f"{cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}",
        'Precision (Macro)': f"{cv_results['test_precision_macro'].mean():.4f} ± {cv_results['test_precision_macro'].std():.4f}",
        'Recall (Macro)': f"{cv_results['test_recall_macro'].mean():.4f} ± {cv_results['test_recall_macro'].std():.4f}",
        'F1-Score (Macro)': f"{cv_results['test_f1_macro'].mean():.4f} ± {cv_results['test_f1_macro'].std():.4f}",
        'ROC-AUC (OVR)': f"{cv_results['test_roc_auc_ovr'].mean():.4f} ± {cv_results['test_roc_auc_ovr'].std():.4f}",
        'PR-AUC': f"{cv_results['test_pr_auc'].mean():.4f} ± {cv_results['test_pr_auc'].std():.4f}"
    }
    
    evaluation_results.append(results)
    
    # Print individual model results
    print(f"   Accuracy: {results['Accuracy']}")
    print(f"   Precision (Macro): {results['Precision (Macro)']}")
    print(f"   Recall (Macro): {results['Recall (Macro)']}")
    print(f"   F1-Score (Macro): {results['F1-Score (Macro)']}")
    print(f"   ROC-AUC (OVR): {results['ROC-AUC (OVR)']}")
    print(f"   PR-AUC: {results['PR-AUC']}")

# Create comprehensive results table
results_df = pd.DataFrame(evaluation_results)
print(f"\n{'='*80}")
print("🏆 COMPREHENSIVE MODEL COMPARISON")
print(f"{'='*80}")
print(results_df.to_string(index=False))

# Find the best model based on ROC-AUC (most robust metric for imbalanced data)
best_model_name = results_df.loc[results_df['ROC-AUC (OVR)'].str.split(' ± ').str[0].astype(float).idxmax(), 'Model']
print(f"\n🥇 Best Model (by ROC-AUC): {best_model_name}")

# Detailed analysis of the best model
print(f"\n🔍 Detailed Analysis of Best Model: {best_model_name}")
print("-" * 50)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Extract the mean values by splitting the strings and converting to float
metrics_to_plot = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', 'ROC-AUC (OVR)', 'PR-AUC']
for col in metrics_to_plot:
    results_df[col] = results_df[col].apply(lambda x: float(x.split(' ± ')[0]))

results_df.rename(columns={'Precision (Macro)': 'Precision',
                           'Recall (Macro)': 'Recall',
                           'F1-Score (Macro)': 'F1-Score',
                           'ROC-AUC (OVR)': 'ROC-AUC',
                           'PR-AUC': 'PR-AUC'
                           },
                   inplace=True)

# Save results_df to csv without index

results_df.to_csv('data/prediction_model_performance.csv', index=False)

# %%
results_df

# %%
results_df

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data for the model performance
# Set the model as the index
# results_df.set_index('Model', inplace=True)

plot_df = results_df[['Accuracy', 'ROC-AUC', 'F1-Score']]
# results_df = results_df['Accuracy', 'ROC-AUC', 'F1-Score']
# Plotting the data horizontally
ax = plot_df.plot(kind='bar', figsize=(4, 6), alpha=0.5)

# Setting labels and title
plt.xlabel("")
# plt.ylabel("Model")
# plt.title("(a)", bbox_to_anchor=(0, 1.02, 1, 0.2), fontsize=12)
# plt.xlim(0, 1)
plt.ylim(0.5, 0.8)
plt.legend(title='Metrics', loc='lower right', frameon=True)
plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.grid(False)

# Tilt the x-axis labels
plt.xticks(rotation=45)
# plt.text(-0.4, 0.785, "(a)", fontsize=12, fontfamily="Arial")

# Add annotations to the bars
for p in ax.patches:
    width = p.get_width()
    # plt.text(width + 0.01, p.get_y() + p.get_height()/2.,
    #          f'{width:.3f}',
    #          ha='center', va='center')

# Save the figure
# plt.savefig('model_performance_horizontal.png', dpi=600)
plt.show()

# %% [markdown]
# # Feature Importance

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBClassifier
import ast

# # 1. Define the 5-fold cross-validation strategy
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# # --- This cell assumes 'X' and 'Y' are already defined ---
# # Preprocessing from your original code
# Y = Y.astype(int)

# # --- 1. Find the Best Hyperparameters for XGBoost ---

# best_xgb_params = xgb_grid_search.best_params_
# print(f"\n✅ Best XGBoost parameters found: {best_xgb_params}")
best_xgb_params = tuned_results_df.iloc[2,:][2]
try:
    best_params = ast.literal_eval(best_xgb_params)
except:
    best_params = best_xgb_params

# --- 2. Calculate Feature Importances using 10-Fold Cross-Validation ---
print("\nCalculating feature importances using 10-fold cross-validation...")

# Instantiate the final model with the best parameters found
final_xgb_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', random_state=42, enable_categorical=True)

# Store feature importances from each fold
fold_importances = []

X_feature_importance = pd.concat([X, T], axis=1).copy()
X_feature_importance.columns = X_feature_importance.columns.str.replace(r'[\[\]<]', '', regex=True)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_feature_importance, Y)):
    # Split data for this fold
    X_train, Y_train = X_feature_importance.iloc[train_idx], Y.iloc[train_idx]
    
    # Fit the model on the training data for this fold
    final_xgb_model.fit(X_train, Y_train)
    
    # Get and store feature importances
    importances = pd.Series(final_xgb_model.feature_importances_, index=X_feature_importance.columns)
    fold_importances.append(importances)
    print(f"Fold {fold+1} processed.")

# %%
# Calculate the mean and standard deviation of importances across folds
mean_importances = pd.concat(fold_importances, axis=1).mean(axis=1)

# Sort features by mean importance and get the top 30
top_features = mean_importances.sort_values(ascending=False).head(30)

# feature importance percentage
top_features = top_features / top_features.sum()

# Calc the q5 and q95 of the top_features


# --- Aggregate Feature Importances ---
# Concatenate the feature importances from all folds into a single DataFrame
importance_df = pd.concat(fold_importances, axis=1)

# Calculate mean, 5th, and 95th percentile for each feature
importance_stats = pd.DataFrame({
    'mean': importance_df.mean(axis=1),
    'p5': importance_df.quantile(0.05, axis=1),
    'p95': importance_df.quantile(0.95, axis=1)
})

top_features = importance_stats.loc[top_features.index.tolist()]

mobility_col = ['AV Experience','ADAS Usage', 'Tech Confidence',
                     'Primary Mode', 'Driving Experience',
                        'Driver Trust', 'Passenger']

demo_col = ['Gender', 'Age', 'Ethnicity',
             'Continent', 'Student', 'Employment']

env_col = ['Weather', 'Scene', 'Light Conditions', 'Road Type']

crash_col = ['Time To Event', 'Point of Impact', 'VRU Involvement', 'Impact Severity']

crash_env_col = env_col + crash_col

counterpart_col = ['Other Maneuver', 'Other Speeding', 'Other Body Style', 'Other Violation']

ego_col = ['Ego Fault', 'Ego Avoidability', 'Ego Maneuver', 'Ego Speeding', 'Ego Violation']

maneuver_col = counterpart_col + ego_col

# Assume 'importance_stats' is your DataFrame with feature importances
feature_importances_df = importance_stats.copy()
feature_importances_df = importance_stats.sort_values(by='mean', ascending=True)

# Define thresholds
thresholds = np.linspace(0, 0.01, 101)  # 0 to 0.10 in 101 steps
cumulative_importance = np.zeros(len(thresholds))

# 1. Define the hierarchy
hierarchy_map = {
    'Mobility': mobility_col,
    'Demographics': demo_col,
    'Crash Environment': crash_env_col,
    'Maneuver': counterpart_col + ego_col
}

sub_hierarchy_map = {
    'Maneuver': {'Other': counterpart_col, 'Ego': ego_col}
}

# 2. Prepare data for the sunburst chart using the FULL dataframe
path_data = {
    'ids': [''],
    'labels': ['Feature Importance'],
    'parents': [None],
    'values': [feature_importances_df['mean'].sum()]
}

added_parents = set()

# Iterate over the entire feature importance DataFrame
for feature, row in feature_importances_df.iterrows():
    parts = feature.rsplit('_', 1)
    if len(parts) == 2:
        col_name, value = parts
    else:
        col_name = parts[0]
        value = 'N/A'

    main_cat = None
    for cat, cols in hierarchy_map.items():
        if col_name in cols:
            main_cat = cat
            break

    if not main_cat:
        continue

    # Add main category if not already added
    if main_cat not in added_parents:
        path_data['ids'].append(main_cat)
        path_data['labels'].append(main_cat)
        path_data['parents'].append('')
        cat_cols = hierarchy_map[main_cat]
        cat_features = feature_importances_df.index[feature_importances_df.index.str.startswith(tuple(f"{c}_" for c in cat_cols))]
        path_data['values'].append(feature_importances_df.loc[cat_features, 'mean'].sum())
        added_parents.add(main_cat)

    parent_id = main_cat
    # Handle sub-categories
    if main_cat in sub_hierarchy_map:
        sub_cat_name = None
        for sub_cat, sub_cols in sub_hierarchy_map[main_cat].items():
            if col_name in sub_cols:
                sub_cat_name = sub_cat
                break
        if sub_cat_name:
            sub_cat_id = f"{main_cat}-{sub_cat_name}"
            if sub_cat_id not in added_parents:
                path_data['ids'].append(sub_cat_id)
                path_data['labels'].append(sub_cat_name)
                path_data['parents'].append(main_cat)
                sub_cat_cols = sub_hierarchy_map[main_cat][sub_cat_name]
                sub_cat_features = feature_importances_df.index[feature_importances_df.index.str.startswith(tuple(f"{c}_" for c in sub_cat_cols))]
                path_data['values'].append(feature_importances_df.loc[sub_cat_features, 'mean'].sum())
                added_parents.add(sub_cat_id)
            parent_id = sub_cat_id

    # Add the column name level
    col_id = f"{parent_id}-{col_name}"
    if col_id not in added_parents:
        path_data['ids'].append(col_id)
        path_data['labels'].append(col_name)
        path_data['parents'].append(parent_id)
        col_features = feature_importances_df.index[feature_importances_df.index.str.startswith(f"{col_name}_")]
        path_data['values'].append(feature_importances_df.loc[col_features, 'mean'].sum())
        added_parents.add(col_id)

    # Add the final feature level
    path_data['ids'].append(feature)
    path_data['labels'].append(value)
    path_data['parents'].append(col_id)
    path_data['values'].append(row['mean'])

# %%
import plotly.graph_objects as go
# Assume 'feature_importances_df', 'hierarchy_map', etc., are defined.
# Assume your 'path_data' dictionary is created as in your original code.

# --- START: Modified logic for labels and colors ---
feature_importances_df = importance_stats.sort_values(by='mean', ascending=True)

# Define thresholds
thresholds = np.linspace(0, 0.01, 101)  # 0 to 0.10 in 101 steps
cumulative_importance = np.zeros(len(thresholds))

for i, threshold in enumerate(thresholds):
    # Calculate the index, ensuring it doesn't exceed the DataFrame length
    index = int(threshold * len(feature_importances_df))
    # Ensure the index is within bounds (0 to len-1)
    index = min(index, len(feature_importances_df) - 1)
    index = max(index, 0)  # Ensure it's not negative
    
    cumulative_importance[i] = feature_importances_df['mean'][feature_importances_df['mean'] > threshold].sum()

# 1. Define constants and lookup maps
total_importance = path_data['values'][0]
threshold = (cumulative_importance>0.90).sum()/10000 * total_importance 

# The 'leaf_node_ids' variable is no longer needed for this logic.
id_to_parent = dict(zip(path_data['ids'], path_data['parents']))
id_to_value = dict(zip(path_data['ids'], path_data['values']))

# 2. Define a color map for your main categories (no changes here)
main_cat_color_map = {
    'Maneuver': '#E16C97',
    'Crash Environment': '#3D7ABE',
    'Demographics': '#F2A26C',
    'Mobility': '#00A096',
    '': '#FFFFFF'  # White for the center root node
}

# 3. Helper function to find the top-level ancestor (no changes here)
memo = {}
def get_main_category_id(node_id):
    if node_id in memo:
        return memo[node_id]
    if node_id == '' or id_to_parent.get(node_id) == '':
        memo[node_id] = node_id
        return node_id
    parent_id = id_to_parent[node_id]
    main_cat_id = get_main_category_id(parent_id)
    memo[node_id] = main_cat_id
    return main_cat_id

# 4. Generate the final lists for labels and colors (LOGIC UPDATED HERE)
modified_labels = []
segment_colors = []

parent_percentages = {
    'Mobility': '9%',
    'Demographics': '17%',
    'Crash Environment': '37%', # Ensure this key matches the label in your data
    'Maneuver': '37%'
}
for i in range(len(path_data['ids'])):
    current_id = path_data['ids'][i]
    current_label = path_data['labels'][i]

    current_value = id_to_value[current_id]
    current_parent = id_to_parent.get(current_id)
    # This condition now applies to ALL segments (leaf and middle-level)
    is_below_threshold = current_value < threshold

    # Determine the label based on the new condition
    if is_below_threshold:
        modified_labels.append('')  # Hide label
    elif current_parent == '':
        percentage = parent_percentages.get(current_label, '')
        # Append the percentage with a line break and bold formatting
        new_label = f"{current_label.replace(' ', '<br>')}<br>{percentage}"
        modified_labels.append(new_label)
    else:
        modified_labels.append(path_data['labels'][i].replace('Ego ', '').replace('Other ', ''))

    # Determine the color based on the new condition
    if is_below_threshold:
        segment_colors.append('white') # Set color to white
    else:
        main_cat = get_main_category_id(current_id)
        color = main_cat_color_map.get(main_cat, '#CCCCCC')
        segment_colors.append(color)

# --- END: Modified logic ---

# path_data['parents'] = [p.replace(' ', '<br>') for p in path_data['parents'] if p is not None]

# 5. Create the sunburst plot with modified data (no changes here)
fig = go.Figure(go.Sunburst(
    ids=path_data['ids'],
    labels=modified_labels,
    parents=path_data['parents'],
    values=path_data['values'],
    branchvalues='total',
    insidetextorientation='radial',
    rotation=90,

    # Text color is white
    textfont=dict(color='white'),

    marker=dict(colors=segment_colors)
))

fig.update_layout(
    height=600,
    width=600,
    margin=dict(t=0, l=0, r=0, b=0),
    annotations=[
        dict(
            text="(b)",
            x=0.00,
            y=1.00,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(
                size=16,
                color="black", 
                family="Arial"
            )
        )
    ]
)

# fig.text(-0.1, 0.785, "(b)", fontsize=12)
# save the figure
# fig.write_image('sunburst_chart.png', scale=3, height=600, width=600)
fig.show()


# %%
from matplotlib.gridspec import GridSpec

top_features = top_features.sort_values('mean', ascending=True)
features_to_plot = top_features.index.tolist()
data_for_plot = pd.concat([X_feature_importance[features_to_plot + ['Display']], Y.rename('target')], axis=1)


results = []
for feature in features_to_plot:
    if data_for_plot[feature].nunique() <= 2:
        high_value_group = data_for_plot[data_for_plot[feature] == 1]
    else:
        threshold = data_for_plot[feature].quantile(0.90)
        high_value_group = data_for_plot[data_for_plot[feature] > threshold]

    av_subset = high_value_group[high_value_group['Display'] == 1]['target']
    hdv_subset = high_value_group[high_value_group['Display'] != 1]['target']

    mean_av = av_subset.mean() if not av_subset.empty else np.nan
    mean_hdv = hdv_subset.mean() if not hdv_subset.empty else np.nan

    if not (np.isnan(mean_av) and np.isnan(mean_hdv)):
        results.append({
            'feature': feature,
            'mean_av': mean_av,
            'mean_hdv': mean_hdv
        })

if results:
    plot_df = pd.DataFrame(results).set_index('feature').dropna()
    # Align the order of plot_df with top_30_features
    plot_df = plot_df.reindex(top_features.index).dropna()


if not plot_df.empty:
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- Setup Figure with GridSpec for broken axis ---
    # We create 3 axes: one for the first plot, and two for the second (broken) plot
    fig = plt.figure(figsize=(15, 7))
    gs = GridSpec(1, 3, width_ratios=[6, 2.7, 1], wspace=0.05)

    ax1 = fig.add_subplot(gs[0])
    ax2_left = fig.add_subplot(gs[1], sharey=ax1) # Left part of broken plot
    ax2_right = fig.add_subplot(gs[2], sharey=ax1) # Right part of broken plot

    # --- Plot 1: Dumbbell Plot (on ax1) ---
    # This section remains largely the same
    overall_mean_av = data_for_plot[data_for_plot['Display'] == 1]['target'].mean()
    overall_mean_hdv = data_for_plot[data_for_plot['Display'] != 1]['target'].mean()
    ax1.axvline(x=overall_mean_av, color='deepskyblue', linestyle='--', lw=1.5, label=f'Overall AV Mean ({overall_mean_av:.1%})')
    ax1.axvline(x=overall_mean_hdv, color='sandybrown', linestyle='--', lw=1.5, label=f'Overall HDV Mean ({overall_mean_hdv:.1%})')
    ax1.hlines(y=plot_df.index, xmin=plot_df['mean_av'], xmax=plot_df['mean_hdv'],
               color='gray', alpha=0.6, linewidth=1, zorder=2)
    ax1.scatter(plot_df['mean_av'], plot_df.index,
                color='deepskyblue', s=40, zorder=3, label='AV')
    ax1.scatter(plot_df['mean_hdv'], plot_df.index,
                color='sandybrown', s=40, zorder=3, label='HDV')
    ax1.set_xlabel('Mean Target %', fontsize=12)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax1.set_title('Comparison of Mean Target', fontsize=14, weight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    ax1.grid(axis='x', linestyle='-', alpha=0.8, zorder=0)
    ax1.set_yticklabels([]) # Hide original y-tick labels
    ax1.tick_params(axis='y', length=0)
    x_pos_variable = -0.01
    for i, label in enumerate(plot_df.index):
        parts = label.rsplit('_', 1)
        variable = parts[0]
        value = parts[1] if len(parts) > 1 else ''
        ax1.text(x_pos_variable, i, f"{variable} ", ha='right', va='center', fontsize=10, weight='bold')
        ax1.text(x_pos_variable + 0.001, i, value, ha='left', va='center', fontsize=10)


    # --- Plot 2: Lollipop Plot (on ax2_left and ax2_right) ---
    # Plot the same data on both axes
    for ax in [ax2_left, ax2_right]:
        ax.hlines(y=top_features.index, xmin=top_features['p5']*100, xmax=top_features['p95']*100,
                  color='skyblue', alpha=0.7, linewidth=2, label='5th-95th Percentile Interval', zorder=2)
        ax.scatter(top_features['mean']*100, top_features.index,
                   color='dodgerblue', s=40, zorder=3, label='Mean Importance')
        ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
        ax.grid(axis='x', linestyle='-', alpha=0.8, zorder=0)
        # for index, value in enumerate(top_features['mean']):
        #      if ax == ax2_left:
        #         ax.text(value*100 + 0.2, index, f'{value*100:.1f}', va='center', ha='left', fontsize=9, color='darkslategray')

    # Set the x-axis limits for the break
    ax2_left.set_xlim(0.0, 4)
    ax2_left.set_xticks([0.0, 1.0, 2.0, 3.0, 4.0])
    ax2_right.set_xlim(7, top_features['p95'].max()*100 * 1.1) # Set right limit dynamically
    ax2_right.set_xlim(6.2, 7.6)
    ax2_right.set_xticks([7, 8])
    
    ax2_left.set_xlabel('Feature Importance (%)', fontsize=12, y=0.02) # Centered x-label
    # Hide the spines and ticks at the break
    ax2_left.spines['right'].set_visible(False)
    ax2_right.spines['left'].set_visible(False)
    ax2_right.tick_params(axis='y', left=False) # No ticks on the left of the right plot
    plt.setp(ax2_right.get_yticklabels(), visible=False) # Hide y-labels on right plot

    # Add diagonal lines to indicate the break
    d = .01 # size of the diagonal lines
    # kwargs = dict(transform=ax2_left.transAxes, color='k', clip_on=False)
    # ax2_left.plot((1-d, 1+d), (-d, +d), **kwargs)        # top-left diagonal
    # ax2_left.plot((1-d, 1+d), (1-d, 1+d), **kwargs)   # bottom-left diagonal
    # kwargs.update(transform=ax2_right.transAxes)  # switch to the right axes
    # ax2_right.plot((-d, +d), (-d, +d), **kwargs)        # top-right diagonal
    # ax2_right.plot((-d, +d), (1-d, 1+d), **kwargs)    # bottom-right diagonal

    # Add labels and titles
    ax2_left.set_title('Top 30 Feature Importances', fontsize=14, weight='bold', x = 0.7)
    ax2_right.legend(loc='lower right')
    
    # --- Final plot adjustments ---
    for ax in [ax1, ax2_left, ax2_right]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    ax2_left.spines['right'].set_visible(True) # Re-enable for visual separation
    ax2_right.spines['left'].set_visible(True) # Re-enable for visual separation

    ax2_right.spines['right'].set_linestyle('-.')
    ax2_right.spines['left'].set_linestyle('-.')

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
    plt.show()

else:
    print("Could not generate plot because no features had data for both AV and HDV groups.")

# %% [markdown]
# # Bootstrapping

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

XGB_best_params = tuned_results_list[3]['Best Parameters']
print("Best parameter of XGB", XGB_best_params)

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

# %% [markdown]
# # ATE

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Save the bootstrap_cates_array to a pkl file
import pickle

# if bootstrap_cates_array is not None:
#     with open('bootstrap_cates_array.pkl', 'wb') as f:
#         pickle.dump(bootstrap_cates_array, f)
# Load the bootstrap_cates_array from the npz file
with np.load('data/bootstrap_cates_array.npz') as data:
    bootstrap_cates_array = data['data']
# Calculate the ATE for each bootstrap iteration
bootstrapped_ates = np.nanmean(bootstrap_cates_array, axis=1)

# Calculate the overall ATE point estimate and the confidence interval
ate_point_estimate = np.mean(bootstrapped_ates)
ate_ci_lower_from_cates = np.percentile(bootstrapped_ates, 2.5)
ate_ci_upper_from_cates = np.percentile(bootstrapped_ates, 97.5)

# %%
df['Display']

# %%
df['Display Condition'] = df['Display'].map({0: 'HDV (Control)', 1: 'AV (Treatment)'})

# To ensure the x-axis is ordered logically, convert 'Driver Trust' to an ordered categorical type.
trust_order = ['Low', 'Medium', 'High']
df['Driver Trust'] = pd.Categorical(df['Driver Trust'], categories=trust_order, ordered=True)


# 2. Bootstrap Percentile CI for each subgroup (NEW METHOD)
ci_results = []
# Group data to iterate through each segment
for group, data in tqdm(df.groupby(['Driver Trust', 'Display Condition']), desc="Subgroup CI Bootstrap"):
    if data.empty:
        print(f"Group {group} is empty")
        continue
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Resample from the specific subgroup
        bootstrap_sample = data['Answer Value'].sample(n=len(data), replace=True)
        bootstrap_means.append(bootstrap_sample.mean())
        
    # Calculate observed mean and percentile CIs
    observed_mean = data['Answer Value'].mean()
    ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
    
    ci_results.append({
        'Driver Trust': group[0],
        'Display Condition': group[1],
        'mean': observed_mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })

agg_data = pd.DataFrame(ci_results)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
# --- 1. Data Simulation (Placeholders) ---
# This section creates sample data since the original was not available.
# You can replace this with your actual data loading.

# Define the order for the x-axis
trust_order = ['Low', 'Medium', 'High']
conditions = ['AV (Treatment)', 'HDV (Control)']

# --- 2. Calculate Counts and Percentages for Annotations ---
# Calculate total counts for each vehicle type
total_counts = df['Display Condition'].value_counts()

# Calculate counts for each trust level within each vehicle type
distribution_data = df.groupby(['Display Condition', 'Driver Trust']).size().reset_index(name='count')

# Calculate the percentage
distribution_data['total'] = distribution_data['Display Condition'].map(total_counts)
distribution_data['percentage'] = (distribution_data['count'].astype(float) / distribution_data['total'].astype(float)) * 100
distribution_data = distribution_data.set_index(['Display Condition', 'Driver Trust'])


# --- 3. Generate the Custom X-Axis Labels ---
new_xticklabels = []
for trust_level in trust_order:
    try:
        av_info = distribution_data.loc[('AV (Treatment)', trust_level)]
        av_text = f"AV: {av_info['percentage']:.0f}% (n={av_info['count']/10:.0f})"
    except KeyError:
        av_text = "AV: 0% (n=0)" # Handle cases with no data

    try:
        hdv_info = distribution_data.loc[('HDV (Control)', trust_level)]
        hdv_text = f"HDV: {hdv_info['percentage']:.0f}% (n={hdv_info['count']/10:.0f})"
    except KeyError:
        hdv_text = "HDV: 0% (n=0)"

    # Combine the text for both conditions, separated by a newline
    new_xticklabels.append(f"{trust_level}\n{av_text}\n{hdv_text}")


# --- 4. Combined Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5)) # Increased figure size for readability
# gs = GridSpec(1, 2, width_ratios=[3, 4]) # ax1 is 3 parts, ax2 is 4 parts wide

# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])

colors = {'HDV (Control)': 'darkorange', 'AV (Treatment)': 'royalblue'}
markers = {'HDV (Control)': 'o', 'AV (Treatment)': 's'}

# --- Plot 1: Bootstrap Distribution of ATE (on ax1) ---
sns.histplot(bootstrapped_ates*100, kde=True, ax=ax1, stat="density", color='steelblue', alpha=0.7, bins=50)
ax1.axvline(ate_point_estimate*100, color='darkred', linestyle='--', linewidth=2, label=f'Mean ATE: {ate_point_estimate*100:.1f}%')
ax1.axvspan(ate_ci_lower_from_cates*100, ate_ci_upper_from_cates*100, color='darkred', alpha=0.1, label=f'95% CI: [{ate_ci_lower_from_cates*100:.1f}%, {ate_ci_upper_from_cates*100:.1f}%]')
ax1.set_xlabel('ATE (AV Treatment - HDV Control) (%)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.axvline(0, color='black', linestyle='--', linewidth=1, label='No Effect (ATE = 0)')
ax1.legend(frameon=True)

# --- Plot 2: Risk Perception with ANNOTATED X-AXIS (on ax2) ---
x_ticks = np.arange(len(trust_order))
width = 0.2 # Increased width to prevent marker overlap

for i, condition in enumerate(sorted(agg_data['Display Condition'].unique(), reverse=True)):
    condition_df = agg_data[agg_data['Display Condition'] == condition].set_index('Driver Trust').loc[trust_order].reset_index()

    y_err = [
        condition_df['mean'] - condition_df['ci_lower'],
        condition_df['ci_upper'] - condition_df['mean']
    ]
    # Adjust x_pos to be centered around the tick
    x_pos = x_ticks - width / 2 + i * width

    ax2.errorbar(
        x=x_pos,
        y=condition_df['mean'],
        yerr=y_err,
        label=condition,
        fmt=markers[condition],
        capsize=3,
        color=colors[condition],
        markersize=5,
        linestyle='None'
    )

ax2.set_xlabel('Driver Trust Level (Mediator)', fontsize=12, labelpad=10) # Add padding to avoid overlap
ax2.set_ylabel('Average High Risk % (with 95% CI)', fontsize=12)
ax2.set_ylim(0)
ax2.set_xticks(x_ticks)

# *** THIS IS THE MODIFIED PART ***
# Set the new, detailed x-tick labels
ax2.set_xticklabels(new_xticklabels, fontsize=10)
# *********************************
# --- Add labels (a) and (b) to the subplots ---
ax1.text(-0.08, 0.99, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
ax2.text(-0.08, 0.99, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
ax2.legend(title='Vehicle Type', frameon=True)

# --- Final Figure Customization ---
plt.tight_layout(rect=[0, 0, 1, 1])
# plt.savefig('combined_analysis_plot_percentile_ci.png', dpi=600, bbox_inches='tight')
plt.show()

# %% [markdown]
# # CATE

# %% [markdown]
# # Subgroup analysis

# %%
bootstrap_cates_array.shape

# %%
plot_df.shape

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm.auto import tqdm
import time
import pickle

# Load the bootstrap_cates_array from the npz file
with np.load('data/bootstrap_cates_array.npz') as data:
    bootstrap_cates_array = data['data']

print(f"Bootstrap array shape: {bootstrap_cates_array.shape}")

mobility_col = ['AV Experience','ADAS Usage', 'Tech Confidence',
                     'Primary Mode', 'Driving Experience']

demo_col = ['Gender', 'Age', 'Ethnicity',
             'Continent', 'Student', 'Employment']

crash_env_col = ['Weather', 'Scene', 'Light Conditions', 'Road Type',
                  'Time To Event', 'Point of Impact', 'VRU Involvement', 'Impact Severity'] 

maneuver_col = ['Other Maneuver', 'Other Speeding', 'Other Body Style', 'Other Violation',
                 'Ego Fault', 'Ego Avoidability', 'Ego Maneuver', 'Ego Speeding', 'Ego Violation']

# --- NEW: Function to calculate stats from the bootstrap CATEs array ---
def get_stats_from_bootstrap_array(group):
    """
    Calculates mean and confidence intervals for a subgroup's CATE
    using the pre-computed bootstrap_cates_array.
    """
    # Get the original indices of the rows in this group
    group_indices = group.index

    # Slice the main bootstrap array to get CATEs for this group
    # Shape becomes: (n_bootstraps, n_samples_in_group)
    group_cates_bootstrap = bootstrap_cates_array[:, group_indices, 0]

    # For each bootstrap run, calculate the mean CATE for the group
    # This gives the distribution of the group's mean CATE
    # Shape becomes: (n_bootstraps,)
    group_mean_cates_per_bootstrap = np.mean(group_cates_bootstrap, axis=1)

    # From this distribution, calculate the final mean and quantiles
    final_mean = np.mean(group_mean_cates_per_bootstrap)
    q05 = np.quantile(group_mean_cates_per_bootstrap, 0.05)
    q95 = np.quantile(group_mean_cates_per_bootstrap, 0.95)

    return pd.Series({'mean': final_mean, 'q05': q05, 'q95': q95})


# --- REVISED: Combined Processing Loop ---

plot_df = df.drop(columns=['Driver Trust', 'Answer Value', 'Display Condition', 'Display', 'Country'])


# Align the DataFrame with the NumPy array
num_samples = bootstrap_cates_array.shape[1]
plot_df = plot_df.iloc[:num_samples].reset_index(drop=True)


# Convert all columns to object type
plot_df['Time To Event'] = plot_df['Time To Event'].astype('object')

plot_data = []

print("\n🚀 Starting CATE analysis...")
total_combinations = sum(len(list(combinations(plot_df.columns, i))) for i in range(1, 4))
print(f"Total feature combinations to process (1, 2, and 3-way): {total_combinations}")
overall_start_time = time.time()

for n_features in range(1, 4):
    feature_combinations = list(combinations(plot_df.columns, n_features))
    
    progress_bar = tqdm(
        feature_combinations, 
        desc=f"Processing {n_features}-feature combinations",
        unit=" combo"
    )

    for feature_combo in progress_bar:
        combo_list = list(feature_combo)
        
        # Filter groups by size (must have at least 93 members)
        group_sizes = plot_df.groupby(combo_list)[combo_list[0]].transform('size')

        if n_features == 1:
            filtered_df = plot_df[group_sizes >= 45]
        else:
            filtered_df = plot_df[group_sizes >= 93]
        
        if filtered_df.empty:
            continue

        grouped_data = filtered_df.groupby(combo_list)

        # --- MODIFICATION: Apply the new function ---
        # This one step calculates mean, q05, and q95 directly and robustly
        group_stats = grouped_data.apply(get_stats_from_bootstrap_array).reset_index()

        # Get counts for each group
        counts = grouped_data.size().reset_index(name='count')
        
        # Merge stats and counts
        grouped = pd.merge(group_stats, counts, on=combo_list)
        
        # Create descriptive labels for plotting
        grouped['feature'] = ', '.join(combo_list)
        if n_features > 1:
            grouped['value'] = grouped[combo_list].astype(str).apply(lambda x: ', '.join(x), axis=1)
        else:
            grouped['value'] = grouped[combo_list[0]]
            
        plot_data.append(grouped[['value', 'mean', 'q05', 'q95', 'feature', 'count']])

# Concatenate all data into a single dataframe
all_features_df = pd.concat(plot_data, ignore_index=True)

# Sort by mean CATE for plotting
all_features_df_sorted = all_features_df.sort_values(by='mean').reset_index(drop=True)

# Create a column indicating the interaction level (1, 2, or 3-way)
all_features_df_sorted['pair'] = all_features_df_sorted['value'].astype(str).str.count(',') + 1

overall_end_time = time.time()
print(f"\n✅ Analysis complete in {overall_end_time - overall_start_time:.2f} seconds.")

# Display a sample of the final results
print("\nSample of the final processed data:")
print(all_features_df_sorted.head())

# %%
all_features_df_sorted

# %%
all_features_df_sorted.to_csv('data/subgroup_df_sorted.csv', index=False)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# This part of your script is assumed to run correctly to create all_features_df_sorted

all_features_df_one = all_features_df_sorted[all_features_df_sorted['pair'] == 1]


mobility_mask = all_features_df_one['feature'].str.contains('AV Experience|ADAS Usage|Tech Confidence|Primary Mode|Driving Experience|Driver Trust|Passenger Safety|Passenger Distraction')
mobility_df = all_features_df_one[mobility_mask]

demo_mask = all_features_df_one['feature'].str.contains('Gender|Age|Ethnicity|Continent|Student|Employment')
demo_df = all_features_df_one[demo_mask]

crash_env_mask = all_features_df_one['feature'].str.contains('Weather|Scene|Light Conditions|Road Type|Time To Event|Point of Impact|VRU Involvement|Impact Severity')
crash_env_df = all_features_df_one[crash_env_mask]

maneuver_mask = all_features_df_one['feature'].str.contains('Other Maneuver|Other Speeding|Other Body Style|Other Violation|Ego Fault|Ego Avoidability|Ego Maneuver|Ego Speeding|Ego Violation')
maneuver_df = all_features_df_one[maneuver_mask]

# Print shape of each dataframe
print(f"Mobility: {mobility_df.shape}")
print(f"Demo: {demo_df.shape}")
print(f"Crash Environment: {crash_env_df.shape}")
print(f"Maneuver: {maneuver_df.shape}")

# --- Step 2: Create a reusable plotting function ---
def plot_on_ax(ax, df, title, cmap, norm):
    """
    Plots a given DataFrame onto a specific matplotlib Axes object.
    """
    # Sort by feature and then by the mean value to group categories
    df = df.sort_values(by=['feature', 'mean'], ascending=[True, True]).reset_index(drop=True)

    # Prepare y-axis labels and data positions for hierarchical display
    y_tick_labels = []
    y_tick_positions = []
    data_y_positions = []
    data_rows = []
    y_pos = 0
    last_feature = None

    for index, row in df.iterrows():
        if row['feature'] != last_feature:
            if last_feature is not None:
                y_pos += 1  # Add a blank space between feature groups
            
            # Make the feature name bold
            y_tick_labels.append(r"$\bf{" + row['feature'] + r"}$")
            y_tick_positions.append(y_pos)
            last_feature = row['feature']
            y_pos += 1

        y_tick_labels.append(f"  {row['value']}")
        y_tick_positions.append(y_pos)
        data_y_positions.append(y_pos) # This list correctly stores positions for data points only
        data_rows.append(row)
        y_pos += 1

    # Plot the data points
    for i, row in enumerate(data_rows):
        plot_y = data_y_positions[i]
        cate_value = row['mean']
        color = cmap(norm(cate_value))
        error = [[cate_value - row['q05']], [row['q95'] - cate_value]]

        ax.errorbar(x=cate_value, y=plot_y, xerr=error,
                    fmt='o', color=color, alpha=0.9, markersize=7,
                    capsize=4, elinewidth=1.5, zorder=5)

    # --- Formatting for the subplot ---
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel('CATE (%)', fontsize=12)
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels, ha='right', fontsize=9)
    ax.invert_yaxis()
    
    # --- SOLUTION ---
    # Comment out or remove the original grid line command
    ax.grid(axis='y', linestyle='--', alpha=0.0) 
    
    # Manually draw horizontal grid lines for data rows ONLY to avoid lines on feature labels.
    for y_val in data_y_positions:
        ax.axhline(y_val, linestyle='--', color='gray', linewidth=0.5, alpha=0.5, zorder=0)
    # --- END SOLUTION ---
    
    ax.grid(axis='x', linestyle='--', alpha=0.7) # Keep the vertical grid
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    
    # Set default ticks and format them as percentages
    default_ticks = [-0.1, -0.05, 0]
    ax.set_xticks(default_ticks)
    ax.set_xticklabels([f'{t*100:.0f}%' for t in default_ticks])
    
    ax.tick_params(axis='y', length=0)
    ax.margins(x=0.1)
    
    # You may need to adjust xlim for different plots
    if 'Maneuver' in title or 'Demographic' in title:
        ax.set_xlim(-0.18, 0.1)
        special_ticks = [-0.15, -0.1, -0.05, 0, 0.05]
        ax.set_xticks(special_ticks)
        ax.set_xticklabels([f'{t*100:.0f}%' for t in special_ticks])
    else:
        ax.set_xlim(-0.15, 0.05)

grid_spec = {'height_ratios': [1.7, 3]}

# --- Step 3: Set up the figure and subplots ---
fig, axs = plt.subplots(
    2, 2, 
    figsize=(12, 14),  # Adjusted figure height to better accommodate ratios
    constrained_layout=True,
    gridspec_kw=grid_spec
)

axs = axs.flatten() # Flatten the 2x2 array of axes into a 1D array

# --- Step 4: Define data and settings for plotting ---
data_frames_to_plot = [mobility_df, demo_df, crash_env_df, maneuver_df]
plot_titles = ['Mobility Factors', 'Demographic Factors', 'Crash Environment Factors', 'Maneuver Factors']

# Define colormap and normalization once
cmap = plt.get_cmap('bwr')
norm = mcolors.Normalize(vmin=-0.1, vmax=0.1)
# Cmap ticks
cbar_ticks = [-0.1, -0.05, 0, 0.05, 0.1]

# --- Step 5: Loop through and plot each DataFrame ---
for i in range(len(data_frames_to_plot)):
    plot_on_ax(axs[i], data_frames_to_plot[i], plot_titles[i], cmap, norm)
    # axs[i].legend(loc='lower right', facecolor='white', fontsize=10, framealpha=1)


# --- Step 6: Add a shared colorbar and main title ---
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
# Add colorbar to the figure, linked to all axes 
# Place colorbar on the top side of the figure
cbar = fig.colorbar(sm, ax=axs, location='bottom', shrink=0.6, aspect=30, pad=0.01, ticks=cbar_ticks)
cbar.ax.set_xticklabels([f'{t*100:.0f}%' for t in cbar_ticks])

# cbar.set_label('Conditional Average Treatment Effect (CATE)', size=12, weight='bold')

# --- Step 7: Final layout adjustment and display ---
# Adjust layout to prevent overlap and fit the suptitle
# fig.tight_layout(rect=[0, 0.00, 1.0, 0.96])

plt.show()

# %%
top_low_2_pair

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# --- Step 1: Prepare the Data ---
# This assumes 'all_features_df_sorted' is a pre-existing, sorted DataFrame with columns:
# 'pair', 'feature', 'mean', 'q05', 'q95'

# Filter for 2-pair and 3-pair interaction data
df_2_pair = all_features_df_sorted[all_features_df_sorted['pair'] == 2].copy()
df_2_pair = df_2_pair[df_2_pair['count'] >= 93]

df_3_pair = all_features_df_sorted[all_features_df_sorted['pair'] == 3].copy()
df_3_pair = df_3_pair[df_3_pair['count'] >= 93]

# Function to get the top and bottom N features
def get_top_bottom_features(df, n=10):
    """Sorts a DataFrame by 'mean' and returns the top and bottom N rows."""
    df_sorted = df.sort_values('mean', ascending=False)
    top_n = df_sorted.head(n)
    bottom_n = df_sorted.tail(n)
    return pd.concat([top_n, bottom_n])

# Get the top/bottom 10 for each pair type
top_low_2_pair = get_top_bottom_features(df_2_pair, n=10)
top_low_3_pair = get_top_bottom_features(df_3_pair, n=10)

def plot_top_low_features(ax, df, title, cmap, norm, num_pairs):
    """
    Plots features with y-axis labels formatted in a multi-column, table-like style.
    """
    # Sort the dataframe by mean for an ordered plot (low to high)
    df = df.sort_values(by='mean', ascending=True).reset_index(drop=True)

    # Define y-positions for the bars
    y_pos = np.arange(len(df))

    # Calculate error ranges and get colors
    error_low = df['mean'] - df['q05']
    error_high = df['q95'] - df['mean']
    colors = cmap(norm(df['mean']))

    # Plot the error bars and markers
    for i in range(len(df)):
        ax.errorbar(x=df['mean'][i], y=y_pos[i], xerr=[[error_low[i]], [error_high[i]]],
                    fmt='none', ecolor=colors[i], elinewidth=1.5, capsize=4, zorder=5)
    ax.scatter(x=df['mean'], y=y_pos, c=colors, s=60, edgecolor='black', linewidth=0.5, zorder=10)

    # --- Formatting and Label Placement ---
    # ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel('CATE', fontsize=12)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([]) # Clear default labels for manual placement
    ax.tick_params(axis='y', length=0)

    # --- KEY CHANGE: Create a multi-column table-like layout ---
    # Define the horizontal center positions for the columns in Axis coordinates.
    # These values can be tuned to adjust column spacing.
    if num_pairs == 2:
        # Positions for [Column 1, Column 2] from left to right
        x_centers = [-0.3, -0.1]
    elif num_pairs == 3:
        # Positions for [Column 1, Column 2, Column 3] from left to right
        x_centers = [-0.50, -0.3, -0.1]
    else: # Fallback for single-column labels
        x_centers = [-0.1]

    # Define font properties for the labels
    feature_font = {'fontsize': 9, 'fontweight': 'bold'}
    value_font = {'fontsize': 8, 'color': 'gray'}

    # Manually place each component of the label in its designated column
    for i in range(len(df)):
        y = y_pos[i]
        features = df['feature'].iloc[i].split(', ')
        values = [f"({v})" for v in df['value'].iloc[i].split(', ')]
        
        # Iterate through each feature/value pair and place it
        for j in range(len(features)):
            if j < len(x_centers):
                col_x_position = x_centers[j]
                
                # Place the feature text, centered in its column
                ax.text(x=col_x_position, y=y, s=features[j],
                        transform=ax.get_yaxis_transform(),
                        ha='center', va='bottom', fontdict=feature_font)

                # Place the value text, centered directly below the feature
                ax.text(x=col_x_position, y=y, s=values[j],
                        transform=ax.get_yaxis_transform(),
                        ha='center', va='top', fontdict=value_font)

    # Add descriptive text and other plot aesthetics
    ax.set_xlim(-0.25, 0.15)
    xlim = ax.get_xlim()
    special_ticks = [-0.25, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15]
    ax.set_xticklabels([f'{t*100:.0f}%' for t in special_ticks])
    ax.text(xlim[1]-0.03, 4.5, '10 Lowest CATE', va='center', ha='right', fontsize=10, style='italic', backgroundcolor='#FFFFFFC0')
    ax.text(xlim[1]-0.20, 14.5, '10 Highest CATE', va='center', ha='right', fontsize=10, style='italic', backgroundcolor='#FFFFFFC0')
    ax.axhline(9.5, color='black', linestyle=':', linewidth=1.2, zorder=0)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    
# --- Step 3: Set up the Figure and Plot ---
# Create a figure with two subplots side-by-side
fig, axs = plt.subplots(2, 1, figsize=(8, 11), constrained_layout=True)

# Define colormap and normalization. Use the full data range for consistent coloring.
all_means = pd.concat([top_low_2_pair['mean'], top_low_3_pair['mean']])
vmin, vmax = -0.1, 0.1
norm_range = max(abs(vmin), abs(vmax)) # Center the colormap at 0
cmap = plt.get_cmap('bwr') # Blue-White-Red is great for diverging data
norm = mcolors.Normalize(vmin=-norm_range, vmax=norm_range)

# Plot the data for each pair type
plot_top_low_features(axs[0], top_low_2_pair, '10 Highest & Lowest 2-Pair Interactions', cmap, norm, 2)
plot_top_low_features(axs[1], top_low_3_pair, '10 Highest & Lowest 3-Pair Interactions', cmap, norm, 3)

# Create a mappable object for the colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ticks = [-0.1, -0.05, 0, 0.05, 0.1]
# Add the colorbar to the figure
cbar = fig.colorbar(sm, ax=axs, orientation='horizontal', pad=0.01, shrink=0.8, aspect=40, ticks=cbar_ticks)
cbar.ax.set_xticklabels([f'{t*100:.0f}%' for t in cbar_ticks])

# cbar.set_label('Conditional Average Treatment Effect (CATE)', size=12, weight='bold')

# --- Step 5: Display the Plot ---
plt.show()

