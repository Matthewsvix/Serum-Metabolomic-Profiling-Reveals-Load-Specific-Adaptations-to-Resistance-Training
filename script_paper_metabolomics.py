"""

Paper: Serum Metabolomic Profiling Reveals Load-Specific Adaptations to Resistance Training

@author: Matthews Silva Martins

Versions:
    
Python: 3.9.18
scikit-learn: 1.6.1
NumPy: 1.23.5
Pandas: 2.2.3
Matplotlib: 3.9.2

"""

# -*- coding: utf-8 -*-

# Exhaustive testing of combinations

from itertools import combinations
from collections import namedtuple
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np
import os
from joblib import Parallel, delayed

# === CONFIGURATIONS ===
num_variables = 5
accuracy_limit = 0.5
top_n = 10000
excel_file = ""

# === DATA PREPARATION ===
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

model = RandomForestClassifier(n_estimators=100, random_state=42)
unique_classes = np.unique(y)
print(f"Classes found: {unique_classes}")

# === COMBINATIONS ===
combinations_list = list(combinations(X.columns, num_variables))

Result = namedtuple('Result', [
    'accuracy',
    'recall_class0', 'recall_class1', 'recall_class2',
    'f1_class0', 'f1_class1', 'f1_class2',
    'variables'
])

# === EVALUATION FUNCTION ===
def evaluate_combination(comb):
    comb_list = list(comb)
    X_sub = X_normalized_df[comb_list]
    
    # 1. Cross-validation for global metrics
    y_pred = cross_val_predict(model, X_sub, y, cv=5)
    acc = accuracy_score(y, y_pred)

    # Force pure numpy arrays to avoid Pandas misaligned index issues
    y_true_arr = np.asarray(y)
    y_pred_arr = np.asarray(y_pred)

    # Calculate metrics directly for classes 0, 1, and 2 in order
    all_recalls = recall_score(y_true_arr, y_pred_arr, labels=[0, 1, 2], average=None, zero_division=0)
    all_f1s = f1_score(y_true_arr, y_pred_arr, labels=[0, 1, 2], average=None, zero_division=0)

    # 2. Extra training to extract feature importances
    model_importance = RandomForestClassifier(n_estimators=100, random_state=42)
    model_importance.fit(X_sub, y)
    importances = model_importance.feature_importances_
    
    # Map variable -> importance, sorted from highest to lowest
    var_imp_pairs = list(zip(comb_list, importances))
    var_imp_pairs_sorted = sorted(var_imp_pairs, key=lambda x: x[1], reverse=True)
    
    # Format in the pattern: "Metabolite (0.XXXX)"
    formatted_variables = [f"{var} ({imp:.4f})" for var, imp in var_imp_pairs_sorted]

    return Result(
        accuracy=acc,
        recall_class0=float(all_recalls[0]),
        recall_class1=float(all_recalls[1]),
        recall_class2=float(all_recalls[2]),
        f1_class0=float(all_f1s[0]),
        f1_class1=float(all_f1s[1]),
        f1_class2=float(all_f1s[2]),
        variables=formatted_variables
    )

# === EVALUATING COMBINATIONS IN PARALLEL ===
all_results = Parallel(n_jobs=-1)(
    delayed(evaluate_combination)(comb) for comb in tqdm(combinations_list, desc=f"Testing combinations of {num_variables} variables")
)

# === FINAL FILTERING ===
filtered_results = [r for r in all_results if r.accuracy >= accuracy_limit]
filtered_results = sorted(filtered_results, key=lambda x: x.accuracy, reverse=True)[:top_n]

# === EXPORT ===
if excel_file and os.path.exists(excel_file):
    os.remove(excel_file)

if filtered_results:
    data_to_export = []
    for i, res in enumerate(filtered_results, 1):
        data_to_export.append({
            'Combination': i,
            'Accuracy': res.accuracy,
            'Recall Class 0': res.recall_class0,
            'Recall Class 1': res.recall_class1,
            'Recall Class 2': res.recall_class2,
            'F1 Class 0': res.f1_class0,
            'F1 Class 1': res.f1_class1,
            'F1 Class 2': res.f1_class2,
            'Variables': ', '.join(res.variables)
        })

    df_export = pd.DataFrame(data_to_export)
    # Explicitly set engine='openpyxl' for safety
    if excel_file:
        df_export.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"\nResults exported to {excel_file}")
    
    print(f"\nTop {len(filtered_results)} combinations with accuracy >= {accuracy_limit*100:.1f}%:")
    for i, res in enumerate(filtered_results, 1):
        vars_str = ', '.join(res.variables)
        print(f"{i}. Accuracy: {res.accuracy:.4f} | "
              f"Recall [0: {res.recall_class0:.4f}, 1: {res.recall_class1:.4f}, 2: {res.recall_class2:.4f}] | "
              f"F1 [0: {res.f1_class0:.4f}, 1: {res.f1_class1:.4f}, 2: {res.f1_class2:.4f}] | "
              f"Variables: {vars_str}")
else:
    print(f"\nNo combination of {num_variables} variables reached accuracy >= {accuracy_limit*100:.1f}%.")

print(f"\nTotal combinations tested: {len(combinations_list)}")

#%% PCA

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2


# 1. Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Replace X with your data
samples = df.iloc[:,0]

# 2. PCA
pca = PCA(n_components=10)
scores = pca.fit_transform(X_scaled)

classes = ['Before','After LL', 'After HL']
num_classes = len(classes)
colors = ['green', 'blue', 'red']  # Customized blue and red

# 4. Map numbers to colors (manually define your colors here)
# Example with hexadecimal colors - modify as needed
manual_colors = ['green', 'blue', 'red']

# Ensure we have enough colors for all classes
colors = manual_colors[:num_classes]

class_to_color = {i: colors[i] for i in range(num_classes)}
y_colors = [class_to_color[label] for label in y]  # Numeric y works now

# 5. Plot with increased size settings
plt.figure(figsize=(12, 10))  # Increased the figure size


scatter = plt.scatter(
    scores[:, 0], 
    scores[:, 1], 
    c=y_colors,           # Mapped colors
    alpha=0.7,
    edgecolor='none',
    s=250
)

# Add sample labels with larger size
#for i, sample_name in enumerate(samples):  # Replace samples with names
 #   plt.text(
  #      scores[i, 0] + 0.02, 
   #     scores[i, 1] + 0.02, 
    #    sample_name,
     #   fontsize=12,  # Increased font size
      #  ha='left',
       # va='bottom'
    #)

# Add confidence ellipses for each class (function kept the same)
def confidence_ellipse(points, level=0.95, color=None, **kwargs):
    if points.shape[0] < 2:
        return
    
    cov = np.cov(points, rowvar=False)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    
    s = -2 * np.log(1 - level)
    width = lambda_[0] * np.sqrt(s)
    height = lambda_[1] * np.sqrt(s)
    
    rotation = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    
    ellipse = plt.matplotlib.patches.Ellipse(
        np.mean(points, axis=0),
        width=width,
        height=height,
        angle=rotation,
        facecolor=color,
        alpha=0.2,
        edgecolor='none',
        **kwargs
    )
    plt.gca().add_patch(ellipse)

# Add ellipses for each class
for class_idx in range(num_classes):
    class_points = scores[y.values == class_idx]
    if len(class_points) > 1:
        confidence_ellipse(
            class_points,
            level=0.99,
            color=colors[class_idx]
        )

# Axes with explained variance and increased size
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=23)  # Increased to 23
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=23)  # Increased to 23

# Increase ticks size
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)

# Legend with increased size
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=classes[i],
                    markerfacecolor=colors[i], markersize=16) for i in range(num_classes)]  # Increased markersize to 16

plt.legend(
    handles=legend_elements,
    fontsize=23,  # Legend font size
)

plt.tight_layout()
file_path_pca = ""
if file_path_pca:
    plt.savefig(file_path_pca, dpi=600)
plt.show()

#%% PLS-DA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import classification_report

# Calculating PLSDA

# 3. Complete normalization (for importance calculation)
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Building final model with the best hyperparameters (2 to 8)
plsda = PLSRegression(n_components=2).fit(X_normalized, y)

# 2. Extract scores (sample projections)
scores = plsda.x_scores_

def confidence_ellipse(points, level=0.95, color=None, **kwargs):
    """
    Creates a confidence ellipse for a set of 2D points.
    
    Parameters:
    -----------
    points : array-like, shape (n, 2)
        Points to calculate the ellipse for.
    level : float, optional
        Confidence level (default 0.95).
    color : str or tuple
        Ellipse color.
    **kwargs
        Additional arguments for the ellipse patch.
    """
    if points.shape[0] < 2:
        return  # Not enough points to calculate an ellipse
    
    # Calculate covariance matrix and its eigenvalues/eigenvectors
    cov = np.cov(points, rowvar=False)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    
    # Calculate ellipse size based on confidence level
    s = -2 * np.log(1 - level)
    width = lambda_[0] * np.sqrt(s)
    height = lambda_[1] * np.sqrt(s)
    
    # Calculate rotation angle
    rotation = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    
    # Create ellipse
    ellipse = Ellipse(
        np.mean(points, axis=0),
        width=width,
        height=height,
        angle=rotation,
        facecolor=color,
        alpha=0.2,  # Transparency
        edgecolor='none',  # No border
        **kwargs
    )
    plt.gca().add_patch(ellipse)

# 1. Extract scores (sample projections)
scores = plsda.x_scores_

# --- EXPLAINED VARIANCE CALCULATION FOR PLS-DA ---
# Total sum of squares of X
total_ss_X = np.sum(X_normalized ** 2)

# Variance for LV1 and LV2
var_lv1 = np.sum(scores[:, 0] ** 2) / total_ss_X * 100
var_lv2 = np.sum(scores[:, 1] ** 2) / total_ss_X * 100

# 3. Define CUSTOMIZED classes and colors
classes = ['Before','After LL', 'After HL']
num_classes = len(classes)
colors = ['green', 'blue', 'red']  # Customized blue and red

# 4. Plot 2D scores plot
plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    scores[:, 0], 
    scores[:, 1], 
    c=y,           
    cmap=plt.cm.colors.ListedColormap(colors),
    alpha=0.7,
    edgecolor='none',
    s=250
)

# Add sample names with increased size
#for i, sample_name in enumerate(samples):
 #   plt.text(
  #      scores[i, 0] + 0.02,
   #     scores[i, 1] + 0.02,
    #    sample_name,
     #   fontsize=10,  # Increased size
      #  ha='left',
       # va='bottom'
    #)

# Add confidence ellipses for each class
for class_idx in range(num_classes):
    class_points = scores[y == class_idx, :2]
    if len(class_points) > 1:
        confidence_ellipse(
            class_points,
            level=0.99,
            color=colors[class_idx]
        )

# Plot customizations (MODIFIED to include explained variance)
plt.xlabel(f'LV1 ({var_lv1:.1f}%)', fontsize=23)  # Added variance %
plt.ylabel(f'LV2 ({var_lv2:.1f}%)', fontsize=23)  # Added variance %

# Increase ticks size
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)

# Legend with increased size
plt.legend(
    handles=scatter.legend_elements(prop='colors')[0],
    labels=classes,
    loc='best',
    fontsize=23,
    markerscale=2.5  # ⬅️ Here you increase the size of the legend markers
)
plt.tight_layout()
file_path_plsda = ""
if file_path_plsda:
    plt.savefig(file_path_plsda, dpi=600)
plt.show()

