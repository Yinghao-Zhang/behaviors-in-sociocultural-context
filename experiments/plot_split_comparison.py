"""
Visualize comparison between within-person and between-person validation results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load both results
df_within = pd.read_csv('outputs/prediction_validation_results.csv')
df_between = pd.read_csv('outputs/prediction_validation_between_person.csv')

# Filter to valid cases (both classes for log_loss)
df_within_valid = df_within[df_within['null_log_loss'].notnull()]
df_between_valid = df_between[df_between['null_log_loss'].notnull()]

print(f"Within-person: {len(df_within)} people ({len(df_within_valid)} valid for log_loss)")
print(f"Between-person: {len(df_between)} people ({len(df_between_valid)} valid for log_loss)")

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Within-Person vs Between-Person Validation', fontsize=16, fontweight='bold')

models = ['null', 'lr', 'no_learn', 'full']
model_names = ['Null', 'Logistic\nReg', 'No-Learning', 'Full Model']
colors = ['#888888', '#4CAF50', '#FF9800', '#2196F3']

# 1. Accuracy comparison
ax = axes[0, 0]
within_acc = [df_within[f'{m}_accuracy'].mean() for m in models]
between_acc = [df_between[f'{m}_accuracy'].mean() for m in models]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, within_acc, width, label='Within-Person', alpha=0.8, color='#E57373')
bars2 = ax.bar(x + width/2, between_acc, width, label='Between-Person', alpha=0.8, color='#64B5F6')

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Accuracy: Within vs Between Person Split', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=10)
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

# 2. Log Loss comparison
ax = axes[0, 1]
within_ll = [df_within_valid[f'{m}_log_loss'].mean() for m in models]
between_ll = [df_between_valid[f'{m}_log_loss'].mean() for m in models]

bars1 = ax.bar(x - width/2, within_ll, width, label='Within-Person', alpha=0.8, color='#E57373')
bars2 = ax.bar(x + width/2, between_ll, width, label='Between-Person', alpha=0.8, color='#64B5F6')

ax.set_ylabel('Log Loss (lower is better)', fontsize=12)
ax.set_title('Log Loss: Within vs Between Person Split', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=10)
ax.legend()
ax.set_ylim([0, 0.8])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

# 3. Performance relative to Null (Accuracy delta)
ax = axes[1, 0]
within_delta = [(df_within[f'{m}_accuracy'] - df_within['null_accuracy']).mean() 
                for m in models[1:]]  # Skip null
between_delta = [(df_between[f'{m}_accuracy'] - df_between['null_accuracy']).mean() 
                 for m in models[1:]]

x2 = np.arange(len(models[1:]))
bars1 = ax.bar(x2 - width/2, within_delta, width, label='Within-Person', alpha=0.8, color='#E57373')
bars2 = ax.bar(x2 + width/2, between_delta, width, label='Between-Person', alpha=0.8, color='#64B5F6')

ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax.set_ylabel('Δ Accuracy vs Null', fontsize=12)
ax.set_title('Improvement Over Null Model (Accuracy)', fontsize=13, fontweight='bold')
ax.set_xticks(x2)
ax.set_xticklabels(model_names[1:], fontsize=10)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        va = 'bottom' if height > 0 else 'top'
        offset = 0.01 if height > 0 else -0.01
        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{height:+.3f}', ha='center', va=va, fontsize=8)

# 4. Performance relative to Null (Log Loss delta, negative is better)
ax = axes[1, 1]
within_delta_ll = [(df_within_valid[f'{m}_log_loss'] - df_within_valid['null_log_loss']).mean() 
                   for m in models[1:]]
between_delta_ll = [(df_between_valid[f'{m}_log_loss'] - df_between_valid['null_log_loss']).mean() 
                    for m in models[1:]]

bars1 = ax.bar(x2 - width/2, within_delta_ll, width, label='Within-Person', alpha=0.8, color='#E57373')
bars2 = ax.bar(x2 + width/2, between_delta_ll, width, label='Between-Person', alpha=0.8, color='#64B5F6')

ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax.set_ylabel('Δ Log Loss vs Null (negative is better)', fontsize=12)
ax.set_title('Improvement Over Null Model (Log Loss)', fontsize=13, fontweight='bold')
ax.set_xticks(x2)
ax.set_xticklabels(model_names[1:], fontsize=10)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        va = 'bottom' if height > 0 else 'top'
        offset = 0.01 if height > 0 else -0.01
        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{height:+.3f}', ha='center', va=va, fontsize=8)

# Add annotation box
fig.text(0.5, 0.02, 
         'KEY FINDING: Full Model flips from WORST (within-person) to BEST (between-person)!\\n' +
         'Between-person split is appropriate: tests generalization to new people.',
         ha='center', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig('outputs/within_vs_between_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved figure: outputs/within_vs_between_comparison.png")

# Create second figure: Direct comparison of Full Model performance
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

# Full model accuracy across splits
ax = axes2[0]
data_within = df_within['full_accuracy'].values
data_between = df_between['full_accuracy'].values

bp = ax.boxplot([data_within, data_between], labels=['Within-Person', 'Between-Person'], 
                 patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#2196F3')
    patch.set_alpha(0.7)

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Full Model Accuracy Distribution', fontsize=13, fontweight='bold')
ax.axhline(0.5, color='red', linestyle='--', alpha=0.3, label='Chance')
ax.grid(axis='y', alpha=0.3)
ax.legend()

# Add mean markers
means_acc = [data_within.mean(), data_between.mean()]
ax.plot([1, 2], means_acc, 'r*', markersize=15, label='Mean', zorder=3)

# Full model log loss across splits
ax = axes2[1]
data_within_ll = df_within_valid['full_log_loss'].values
data_between_ll = df_between_valid['full_log_loss'].values

bp = ax.boxplot([data_within_ll, data_between_ll], labels=['Within-Person', 'Between-Person'],
                 patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#2196F3')
    patch.set_alpha(0.7)

ax.set_ylabel('Log Loss (lower is better)', fontsize=12)
ax.set_title('Full Model Log Loss Distribution', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add mean markers
means_ll = [data_within_ll.mean(), data_between_ll.mean()]
ax.plot([1, 2], means_ll, 'r*', markersize=15, label='Mean', zorder=3)
ax.legend()

# Add summary statistics
fig2.text(0.5, 0.02,
          f'Within-Person: Acc={data_within.mean():.3f}, LogLoss={data_within_ll.mean():.3f} | ' +
          f'Between-Person: Acc={data_between.mean():.3f}, LogLoss={data_between_ll.mean():.3f}',
          ha='center', fontsize=10,
          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('outputs/full_model_split_comparison.png', dpi=150, bbox_inches='tight')
print("Saved figure: outputs/full_model_split_comparison.png")

print("\nDone!")
