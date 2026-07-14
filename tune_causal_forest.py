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
df = pd.read_parquet('data/main/df_final.parquet')

# Drop columns with all NaN values
df = df.dropna(axis=1, how='all')

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


# drop Driver trust
drop_col = ['Driver Trust_High', 'Driver Trust_Low']
X = X.drop(columns=drop_col)
# Keep a copy of the original features for the final analysis
df_original_features = df.loc[X.index, categorical_features]

print("Data preparation complete.")
print(f"Analyzing {len(X)} samples.\n")

# %% [markdown]
# # Tune the classification models

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tqdm.auto import tqdm

# This cell assumes 'X' and 'Y' are already defined.


# 1. Define the 5-fold cross-validation strategy
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
        'max_depth': [3, 5, 8, 10],
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

# %% [markdown]
# ## Fine tune XGBoost
# 
# 
# 
# 
# 
# 
# 
# 
# 

# %%
import pandas as pd
import numpy as np
import contextlib
import joblib
from tqdm.auto import tqdm
from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid
from xgboost import XGBClassifier

# This helper function is the key to showing progress inside GridSearchCV
@contextlib.contextmanager
def TqdmParallel(tqdm_object):
    """Context manager to patch joblib to display a tqdm progress bar."""
    
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
        
# --- Main Script ---

# For this example to be runnable, let's create dummy X and Y.

# 1. Define the cross-validation strategy
kfold = KFold(n_splits=10, shuffle=True)

# 2. Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 400, 800],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 8, 16],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 3. Define the base model
model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', enable_categorical=True)

# 4. Set up GridSearchCV
#    n_jobs=-1 is crucial to activate the parallel backend.
#    verbose=0 prevents scikit-learn's own text output from cluttering the console.
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=kfold,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0 
)

# 5. Calculate the total number of fits for the progress bar
#    (number of parameter combinations * number of CV splits)
num_fits = len(ParameterGrid(param_grid)) * kfold.get_n_splits()

print(f"🚀 Starting GridSearchCV for {num_fits} total fits...")

# 6. Fit the model using the TqdmParallel context manager
with TqdmParallel(tqdm(total=num_fits, desc="Fitting models")):
    grid_search.fit(pd.concat([X, T], axis=1), Y)

# 7. Display the results
print("\n\n--- 🏆 Hyperparameter Tuning Results ---")
print(f"Best Score (AUC): {grid_search.best_score_:.4f}")
print(f"Best Parameters: {grid_search.best_params_}")

# %%
# --- NEW: Evaluate the Best Model using 10-Fold CV ---
print("\n🔄 Evaluating the best model with 10-fold cross-validation...")

# 9. Get the best model from the grid search
best_model = grid_search.best_estimator_

# 10. Define all the metrics you want to calculate
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# 11. Use cross_validate to get all scores in one go
cv_results = cross_validate(
    estimator=best_model,
    X=pd.concat([X, T], axis=1),
    y=Y,
    cv=kfold,
    scoring=scoring_metrics,
    n_jobs=-1
)

# 12. Calculate mean and std deviation, then display in a DataFrame
results_df = pd.DataFrame(cv_results)
# Rename columns for clarity (e.g., 'test_roc_auc' -> 'ROC-AUC')
metric_names = {
    'test_accuracy': 'Accuracy',
    'test_precision': 'Precision',
    'test_recall': 'Recall',
    'test_f1': 'F1-Score',
    'test_roc_auc': 'ROC-AUC'
}
final_scores = pd.DataFrame({
    'Metric': [metric_names[col] for col in metric_names],
    'Mean Score': [results_df[col].mean() for col in metric_names],
    'Std Dev': [results_df[col].std() for col in metric_names]
})

print("\n--- 📊 Final Cross-Validation Scores ---")
print(final_scores.to_string(index=False))

# %%

# 2. Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 400],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 8, 16],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 3. Define the base model
model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', enable_categorical=True)

# 4. Set up GridSearchCV
#    n_jobs=-1 is crucial to activate the parallel backend.
#    verbose=0 prevents scikit-learn's own text output from cluttering the console.
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=kfold,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0 
)

# 5. Calculate the total number of fits for the progress bar
#    (number of parameter combinations * number of CV splits)
num_fits = len(ParameterGrid(param_grid)) * kfold.get_n_splits()

print(f"🚀 Starting GridSearchCV for {num_fits} total fits...")

# 6. Fit the model using the TqdmParallel context manager
with TqdmParallel(tqdm(total=num_fits, desc="Fitting models")):
    grid_search.fit(pd.concat([X, T], axis=1), Y)

# 7. Display the results
print("\n\n--- 🏆 Hyperparameter Tuning Results ---")
print(f"Best Score (AUC): {grid_search.best_score_:.4f}")
print(f"Best Parameters: {grid_search.best_params_}")

# %%
XGB_best_params = grid_search.best_params_

# Save as json
import json
with open('XGB_best_params.json', 'w') as f:
    json.dump(XGB_best_params, f, indent=4)

# Load the json
with open('XGB_best_params.json', 'r') as f:
    XGB_best_params = json.load(f)

# %%
tuned_results_df.to_csv('tuned_results_df.csv', index=False)

# %%
tuned_results_df = pd.read_csv('tuned_results_df.csv')

# %%
tuned_results_df

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

results_df.to_csv('prediction_model_performance.csv', index=False)

# %%
results_df = pd.read_csv('prediction_model_performance.csv')

# %%
# Change the order of metrics to plot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'PR-AUC', 'ROC-AUC']

plot_df = results_df.reindex(columns=['Model']+metrics)

models = plot_df['Model'].tolist()
metrics = plot_df.columns.tolist()[1:]
num_metrics = len(metrics)

# Calculate the angle for each axis in the radar chart
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
# Make the plot circular by completing the loop
angles += angles[:1]

# Set up the plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

color_styles = ['red', 'orange', 'green', 'black']
dash_styles = ['dashdot', 'solid', 'dashed', 'dotted']

# Plot each model's data with a unique dash style
for i, row in plot_df.iterrows():
    values = row.drop('Model').values.flatten().tolist()
    # Complete the loop for a closed shape
    values += values[:1]
    # Assign a dash style from the list, cycling through if there are more models than styles
    style = dash_styles[i % len(dash_styles)]
    color = color_styles[i % len(color_styles)]
    ax.plot(angles, values, linewidth=1.5, linestyle=style, color=color, label=row['Model'])
    # Set alpha to 0.0 to make the fill transparent
    ax.fill(angles, values, alpha=0.0)

# --- 2. Customization ---

# Set the plot to start at the top (90 degrees)
ax.set_theta_offset(np.pi / 2)

# Remove the radial grid lines (the "straight lines in the circle")
# ax.xaxis.grid(False)

# Remove the outer circle (polar spine)
ax.spines['polar'].set_visible(False)

# Set the y-axis labels (grid lines)
ax.set_rlabel_position(0)
ax.set_yticks([0.6, 0.65, 0.7, 0.75, 0.8])
ax.set_yticklabels(["0.60", "0.65", "0.70", "0.75", "0.80"], color="black", size=8)
ax.set_ylim(0.58, 0.82)

# Set the x-axis labels (metric names)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, size=10) # Default matplotlib font

# Add the legend at the bottom center with 2 columns
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=2)

# Show the plot
plt.show()

# %%
# The DataFrame you want to save
plot_df = results_df.reindex(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'PR-AUC', 'ROC-AUC'])
plot_df

# --- Save the DataFrame to a CSV file ---
# The index=False argument prevents pandas from writing the DataFrame index as a column.
plot_df.to_csv('model_metrics.csv', index=False)

# print("DataFrame saved to model_metrics.csv")

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

# --- 3. Aggregate and Plot the Top 20 Features ---

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

# %%
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

# %% [markdown]
# # Sunburst Feature importance

# %%
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

plt.figure(figsize=(10, 6))
plt.plot(thresholds, cumulative_importance, label='Cumulative Feature Importance')
plt.xlabel('Threshold (proportion of features)')
plt.ylabel('Cumulative Importance')
plt.title('Cumulative Feature Importance vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Fixed cumulative importance calculation completed successfully!")

# %%
mobility_col = ['AV Experience','ADAS Usage', 'Tech Confidence',
                     'Primary Mode', 'Driving Experience', 'Driver Trust']

demo_col = ['Gender', 'Age', 'Ethnicity',
             'Continent', 'Student', 'Employment']

env_col = ['Weather', 'Scene', 'Light Conditions', 'Road Type']

crash_col = ['Time To Event', 'Point of Impact', 'VRU Involvement', 'Impact Severity']

crash_env_col = env_col + crash_col

counterpart_col = ['Other Maneuver', 'Other Speeding', 'Other Body Style', 'Other Violation']

ego_col = ['Ego Fault', 'Ego Avoidability', 'Ego Maneuver', 'Ego Speeding', 'Ego Violation']

maneuver_col = counterpart_col + ego_col

# %%
# Assume 'importance_stats' is your DataFrame with feature importances
feature_importances_df = importance_stats.copy()

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
fig.write_image('sunburst_chart.png', scale=3, height=600, width=600)
fig.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data for the model performance
data = {
    'Model': ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM'],
    'Accuracy': [0.770, 0.778, 0.780, 0.780],
    # 'Precision': [0.732, 0.721, 0.725, 0.725],
    # 'Recall': [0.609, 0.661, 0.660, 0.659],
    'ROC-AUC': [0.778, 0.786, 0.789, 0.792],
    'F1-Score': [0.619, 0.677, 0.677, 0.675]
}
df = pd.DataFrame(data)

# Set the model as the index
df.set_index('Model', inplace=True)

# Plotting the data horizontally
ax = df.plot(kind='bar', figsize=(4, 6), alpha=0.5)

# Setting labels and title
plt.xlabel("")
# plt.ylabel("Model")
# plt.title("(a)", bbox_to_anchor=(0, 1.02, 1, 0.2), fontsize=12)
# plt.xlim(0, 1)
plt.ylim(0.5, 0.8)
plt.legend(title='Metrics', loc='lower right')
plt.tight_layout(rect=[0, 0, 0.95, 1])

# Tilt the x-axis labels
plt.xticks(rotation=45)
plt.text(-0.4, 0.785, "(a)", fontsize=12, fontfamily="Arial")

# Add annotations to the bars
for p in ax.patches:
    width = p.get_width()
    # plt.text(width + 0.01, p.get_y() + p.get_height()/2.,
    #          f'{width:.3f}',
    #          ha='center', va='center')

# Save the figure
plt.savefig('model_performance_horizontal.png', dpi=600)
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import plotly.graph_objects as go
import warnings

# Suppress potential warnings from plotly/kaleido
warnings.filterwarnings("ignore", category=UserWarning, module='plotly')

# --- 1. Generate and Save the Matplotlib Bar Chart ---

# Data for the model performance
data = {
    'Model': ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM'],
    'Accuracy': [0.770, 0.778, 0.780, 0.780],
    'Precision': [0.732, 0.721, 0.725, 0.725],
    'Recall': [0.609, 0.661, 0.660, 0.659],
    'F1-Score': [0.619, 0.677, 0.677, 0.675],
    'ROC-AUC': [0.778, 0.786, 0.789, 0.792]
}
df = pd.DataFrame(data)
df.set_index('Model', inplace=True)

# Create the matplotlib figure and axes
fig1, ax1 = plt.subplots(figsize=(12, 8))

# width ratio = 3:7


# Plotting the data horizontally on the specified axes
df.plot(kind='barh', ax=ax1, alpha=0.7)

# Setting labels and title
ax1.set_xlabel("Scores")
ax1.set_ylabel("Model")
ax1.set_title("Model Performance Comparison")
ax1.set_xlim(0, 1)
ax1.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
fig1.tight_layout(rect=[0, 0, 0.85, 1])

# Add annotations to the bars
for p in ax1.patches:
    width = p.get_width()
    ax1.text(width + 0.01, p.get_y() + p.get_height() / 2.,
             f'{width:.3f}',
             ha='left', va='center')

# Save the figure
fig1.savefig('model_performance.png')
plt.close(fig1)

# --- 2. Generate and Save the Plotly Sunburst Chart ---
# NOTE: This step requires the 'kaleido' package.
# Install it using: pip install -U kaleido

# Create dummy data to make the sunburst chart runnable
path_data = {
    'ids': ['Total', 'Maneuver', 'Crash Environment', 'Demographics', 'Mobility', 'Maneuver.Ego', 'Maneuver.Other', 'Crash.A', 'Crash.B', 'Demo.A', 'Demo.B', 'Mobil.A', 'Mobil.B'],
    'labels': ['Total', 'Maneuver', 'Crash Environment', 'Demographics', 'Mobility', 'Ego', 'Other', 'Road Type', 'Weather', 'Age', 'Gender', 'Vehicle Type', 'Trip Purpose'],
    'parents': ['', 'Total', 'Total', 'Total', 'Total', 'Maneuver', 'Maneuver', 'Crash Environment', 'Crash Environment', 'Demographics', 'Demographics', 'Mobility', 'Mobility'],
    'values': [100, 37, 37, 17, 9, 20, 17, 15, 22, 10, 7, 5, 4]
}
threshold = 5
id_to_parent = dict(zip(path_data['ids'], path_data['parents']))
id_to_value = dict(zip(path_data['ids'], path_data['values']))

main_cat_color_map = {
    'Maneuver': '#E16C97', 'Crash Environment': '#3D7ABE', 'Demographics': '#F2A26C', 'Mobility': '#00A096', '': '#FFFFFF'
}
memo = {}
def get_main_category_id(node_id):
    if node_id in memo: return memo[node_id]
    if node_id == '' or id_to_parent.get(node_id) == '' or id_to_parent.get(node_id) == 'Total':
        memo[node_id] = node_id
        return node_id
    parent_id = id_to_parent[node_id]
    main_cat_id = get_main_category_id(parent_id)
    memo[node_id] = main_cat_id
    return main_cat_id

modified_labels, segment_colors = [], []
parent_percentages = {'Mobility': '9%', 'Demographics': '17%', 'Crash Environment': '37%', 'Maneuver': '37%'}

for i in range(len(path_data['ids'])):
    current_id, current_label, current_value, current_parent = path_data['ids'][i], path_data['labels'][i], id_to_value[path_data['ids'][i]], id_to_parent.get(path_data['ids'][i])
    if current_value < threshold:
        modified_labels.append('')
        segment_colors.append('white')
    else:
        if current_parent == 'Total':
            modified_labels.append(f"{current_label.replace(' ', '<br>')}<br><b>{parent_percentages.get(current_label, '')}</b>")
        else:
            modified_labels.append(current_label.replace('Ego ', '').replace('Other ', ''))
        main_cat = get_main_category_id(current_id)
        segment_colors.append(main_cat_color_map.get(main_cat, '#CCCCCC'))

fig2 = go.Figure(go.Sunburst(
    ids=path_data['ids'], labels=modified_labels, parents=path_data['parents'], values=path_data['values'],
    branchvalues='total', insidetextorientation='radial', rotation=90, textfont=dict(color='white', size=14),
    marker=dict(colors=segment_colors)
))
fig2.update_layout(height=600, width=600, margin=dict(t=0, l=0, r=0, b=0), title_text="<b>Feature Importance Distribution</b>", title_x=0.5)

# Save the figure as a static image
try:
    # fig2.write_image('sunburst_chart.png', scale=3)
    # --- 3. Combine the Saved Images into a Single Figure ---

    fig_final, axes = plt.subplots(1, 2, figsize=(22, 8))
    img1 = mpimg.imread('model_performance.png')
    img2 = mpimg.imread('sunburst_chart.png')

    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[1].imshow(img2)
    axes[1].axis('off')

    fig_final.suptitle("Model Analysis: Performance and Feature Importance", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig('combined_plot.png', dpi=300)
    print("Combined plot saved as 'combined_plot.png'")
    plt.show()

except ValueError as e:
    print(f"Error saving Plotly figure: {e}")
    print("Please install the 'kaleido' package by running: pip install -U kaleido")

# %% [markdown]
# # Dumbell plot

# %%
mobility_col = ['AV Experience','ADAS Usage', 'Tech Confidence',
                     'Primary Mode', 'Driving Experience']

demo_col = ['Gender', 'Age', 'Ethnicity',
             'Continent', 'Student', 'Employment']
crash_env_col = ['Weather', 'Scene', 'Light Conditions', 'Road Type',
                  'Time To Event', 'Point of Impact', 'VRU Involvement', 'Impact Severity'] 

maneuver_col = ['Other Maneuver', 'Other Speeding', 'Other Body Style', 'Other Violation', 'Ego Fault', 'Ego Avoidability', 'Ego Maneuver', 'Ego Speeding', 'Ego Violation']

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
    ax1.set_title('(a) Comparison of Mean High-risk Perception', fontsize=14, weight='bold')
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
    ax2_left.set_title('(b) Top 30 Feature Importances', fontsize=14, weight='bold', x = 0.7)
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
    plt.savefig('feature_importance_comparison.png', dpi=300)
    plt.show()

else:
    print("Could not generate plot because no features had data for both AV and HDV groups.")

