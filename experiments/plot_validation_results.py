"""
Quick visualization of prediction validation results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv('outputs/prediction_validation_results.csv')
df_valid = df[df['null_log_loss'].notnull()].copy()

print(f"Loaded {len(df)} people ({len(df_valid)} with valid log_loss)")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Prediction Validation Results', fontsize=16, fontweight='bold')

models = ['null', 'lr', 'no_learn', 'full']
model_names = ['Null', 'Logistic\nRegression', 'No-Learning', 'Full Model\n(with learning)']
colors = ['#888888', '#4CAF50', '#FF9800', '#2196F3']

# 1. Accuracy comparison
ax = axes[0, 0]
means = [df[f'{m}_accuracy'].mean() for m in models]
sems = [df[f'{m}_accuracy'].sem() for m in models]
x_pos = np.arange(len(models))
bars = ax.bar(x_pos, means, yerr=sems, color=colors, alpha=0.7, capsize=5)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Accuracy Comparison\n(N=31, higher is better)', fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, fontsize=10)
ax.axhline(0.5, color='red', linestyle='--', alpha=0.3, label='Chance')
ax.set_ylim([0, 1])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, mean, sem) in enumerate(zip(bars, means, sems)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sem + 0.02, 
            f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

# 2. Log Loss comparison (valid cases only)
ax = axes[0, 1]
means_ll = [df_valid[f'{m}_log_loss'].mean() for m in models]
sems_ll = [df_valid[f'{m}_log_loss'].sem() for m in models]
bars = ax.bar(x_pos, means_ll, yerr=sems_ll, color=colors, alpha=0.7, capsize=5)
ax.set_ylabel('Log Loss', fontsize=12)
ax.set_title(f'Log Loss Comparison\n(N={len(df_valid)}, lower is better)', fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim([0, max(means_ll) * 1.3])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, mean, sem in zip(bars, means_ll, sems_ll):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sem + 0.02, 
            f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

# 3. AUC comparison (valid cases only)
ax = axes[1, 0]
means_auc = [df_valid[f'{m}_auc'].mean() for m in models]
sems_auc = [df_valid[f'{m}_auc'].sem() for m in models]
bars = ax.bar(x_pos, means_auc, yerr=sems_auc, color=colors, alpha=0.7, capsize=5)
ax.set_ylabel('AUC', fontsize=12)
ax.set_title(f'AUC Comparison\n(N={len(df_valid)}, higher is better)', fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, fontsize=10)
ax.axhline(0.5, color='red', linestyle='--', alpha=0.3, label='Random')
ax.set_ylim([0, 1])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, mean, sem in zip(bars, means_auc, sems_auc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sem + 0.02, 
            f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

# 4. Person-level scatter: Full vs Null accuracy
ax = axes[1, 1]
ax.scatter(df['null_accuracy'], df['full_accuracy'], alpha=0.6, s=50, color='#2196F3')
ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Equal performance')
ax.set_xlabel('Null Model Accuracy', fontsize=12)
ax.set_ylabel('Full Model Accuracy', fontsize=12)
ax.set_title('Per-Person: Full vs Null\n(points below line = Full worse)', fontsize=11)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.legend()
ax.grid(alpha=0.3)

# Count how many people have full < null
n_worse = (df['full_accuracy'] < df['null_accuracy']).sum()
n_total = len(df)
ax.text(0.05, 0.95, f'{n_worse}/{n_total} people:\nFull < Null', 
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('outputs/prediction_validation_plot.png', dpi=150, bbox_inches='tight')
print("\nSaved figure: outputs/prediction_validation_plot.png")

# Create a second figure showing distribution of accuracies
fig2, axes2 = plt.subplots(1, 1, figsize=(10, 6))
ax = axes2

# Box plot
data_to_plot = [df[f'{m}_accuracy'].dropna() for m in models]
bp = ax.boxplot(data_to_plot, labels=model_names, patch_artist=True)

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Distribution of Prediction Accuracy Across People', fontsize=14, fontweight='bold')
ax.axhline(0.5, color='red', linestyle='--', alpha=0.3, label='Chance')
ax.grid(axis='y', alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('outputs/prediction_validation_boxplot.png', dpi=150, bbox_inches='tight')
print("Saved figure: outputs/prediction_validation_boxplot.png")

print("\nDone!")
