"""
Compare model performance on multi-agent simulation data.
Visualize the between-person prediction validation results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load results
summary = pd.read_csv('outputs/prediction_validation_between_person_summary.csv')
test_results = pd.read_csv('outputs/prediction_validation_between_person.csv')

print("="*70)
print("MULTI-AGENT SIMULATION: MODEL COMPARISON")
print("="*70)

# Extract key metrics
models = summary['model'].values
accuracies = summary['accuracy_mean'].values
accuracy_se = summary['accuracy_se'].values
log_losses = summary['log_loss_mean'].values
log_loss_se = summary['log_loss_se'].values
aucs = summary['auc_mean'].values
auc_se = summary['auc_se'].values

print("\n📊 AGGREGATE PERFORMANCE (Mean ± SE):")
print("-" * 70)
for i, model in enumerate(models):
    print(f"\n{model}:")
    print(f"  Accuracy:     {accuracies[i]:.3f} ± {accuracy_se[i]:.3f}")
    print(f"  Log Loss:     {log_losses[i]:.3f} ± {log_loss_se[i]:.3f} (lower better)")
    print(f"  AUC:          {aucs[i]:.3f} ± {auc_se[i]:.3f}")

# Statistical comparisons
print("\n" + "="*70)
print("📈 MODEL COMPARISONS")
print("="*70)

# Full model vs baselines (model names: nan, lr, no_learn, full)
# Find indices handling NaN
model_list = []
for m in models:
    if pd.isna(m):
        model_list.append('null')
    else:
        model_list.append(m)

full_idx = model_list.index('full')
null_idx = model_list.index('null')
logistic_idx = model_list.index('lr')
no_learn_idx = model_list.index('no_learn')

print("\n1. Full Model vs Null (always predict majority):")
acc_diff = accuracies[full_idx] - accuracies[null_idx]
ll_diff = log_losses[null_idx] - log_losses[full_idx]  # Improvement = reduction
print(f"   Accuracy:  {acc_diff:+.1%} (Full {accuracies[full_idx]:.3f} vs Null {accuracies[null_idx]:.3f})")
print(f"   Log Loss:  {ll_diff:+.3f} improvement ({100*ll_diff/log_losses[null_idx]:.1f}% better)")

print("\n2. Full Model vs Logistic Regression:")
acc_diff = accuracies[full_idx] - accuracies[logistic_idx]
ll_diff = log_losses[logistic_idx] - log_losses[full_idx]
print(f"   Accuracy:  {acc_diff:+.1%} (Full {accuracies[full_idx]:.3f} vs Logistic {accuracies[logistic_idx]:.3f})")
print(f"   Log Loss:  {ll_diff:+.3f} improvement ({100*ll_diff/log_losses[logistic_idx]:.1f}% better)")

print("\n3. Full Model vs No-Learning Model:")
acc_diff = accuracies[full_idx] - accuracies[no_learn_idx]
ll_diff = log_losses[no_learn_idx] - log_losses[full_idx]
print(f"   Accuracy:  {acc_diff:+.1%} (Full {accuracies[full_idx]:.3f} vs No-Learn {accuracies[no_learn_idx]:.3f})")
print(f"   Log Loss:  {ll_diff:+.3f} improvement ({100*ll_diff/log_losses[no_learn_idx]:.1f}% better)")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Multi-Agent Simulation: Between-Person Prediction Validation', 
             fontsize=16, fontweight='bold')

# Color scheme (using standardized model names)
color_map = {
    'null': '#95a5a6',
    'lr': '#3498db', 
    'no_learn': '#e67e22',
    'full': '#27ae60'
}
colors_list = [color_map[m] for m in model_list]

# 1. Accuracy comparison
ax1 = axes[0, 0]
x = np.arange(len(models))
bars = ax1.bar(x, accuracies, yerr=accuracy_se, 
               color=colors_list,
               alpha=0.8, capsize=5)
ax1.set_xticks(x)
ax1.set_xticklabels(['Null', 'Logistic', 'No-Learn', 'Full'], rotation=0)
ax1.set_ylabel('Accuracy', fontweight='bold')
ax1.set_title('(A) Prediction Accuracy', fontweight='bold')
ax1.set_ylim(0, 1)
ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax1.grid(axis='y', alpha=0.2)
# Add value labels
for i, (acc, se) in enumerate(zip(accuracies, accuracy_se)):
    ax1.text(i, acc + se + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

# 2. Log loss comparison
ax2 = axes[0, 1]
bars = ax2.bar(x, log_losses, yerr=log_loss_se,
               color=colors_list,
               alpha=0.8, capsize=5)
ax2.set_xticks(x)
ax2.set_xticklabels(['Null', 'Logistic', 'No-Learn', 'Full'], rotation=0)
ax2.set_ylabel('Log Loss (lower better)', fontweight='bold')
ax2.set_title('(B) Prediction Calibration', fontweight='bold')
ax2.set_ylim(0, max(log_losses) * 1.2)
ax2.grid(axis='y', alpha=0.2)
# Add value labels
for i, (ll, se) in enumerate(zip(log_losses, log_loss_se)):
    ax2.text(i, ll + se + 0.03, f'{ll:.3f}', ha='center', va='bottom', fontsize=9)

# 3. Per-person accuracy distribution
ax3 = axes[1, 0]
test_data_melted = test_results.melt(
    id_vars=['person_id'],
    value_vars=['null_accuracy', 'lr_accuracy', 'no_learn_accuracy', 'full_accuracy'],
    var_name='Model', value_name='Accuracy'
)
test_data_melted['Model'] = test_data_melted['Model'].map({
    'null_accuracy': 'Null',
    'lr_accuracy': 'Logistic',
    'no_learn_accuracy': 'No-Learn',
    'full_accuracy': 'Full'
})
sns.violinplot(data=test_data_melted, x='Model', y='Accuracy', ax=ax3,
               palette=['#95a5a6', '#3498db', '#e67e22', '#27ae60'],
               inner='quartile')
ax3.set_title('(C) Per-Person Accuracy Distribution', fontweight='bold')
ax3.set_ylabel('Accuracy', fontweight='bold')
ax3.set_xlabel('')
ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax3.set_ylim(0, 1)
ax3.grid(axis='y', alpha=0.2)

# 4. AUC comparison
ax4 = axes[1, 1]
bars = ax4.bar(x, aucs, yerr=auc_se,
               color=colors_list,
               alpha=0.8, capsize=5)
ax4.set_xticks(x)
ax4.set_xticklabels(['Null', 'Logistic', 'No-Learn', 'Full'], rotation=0)
ax4.set_ylabel('AUC (Area Under ROC Curve)', fontweight='bold')
ax4.set_title('(D) Discrimination Performance', fontweight='bold')
ax4.set_ylim(0, 1)
ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1, label='Chance')
ax4.grid(axis='y', alpha=0.2)
# Add value labels
for i, (auc, se) in enumerate(zip(aucs, auc_se)):
    ax4.text(i, auc + se + 0.02, f'{auc:.3f}', ha='center', va='bottom', fontsize=9)
ax4.legend()

plt.tight_layout()
plt.savefig('outputs/multi_agent_validation_comparison.png', dpi=300, bbox_inches='tight')
print("\n" + "="*70)
print("✅ Saved: outputs/multi_agent_validation_comparison.png")

# Winner analysis
print("\n" + "="*70)
print("🏆 WINNER ANALYSIS (Per-Person)")
print("="*70)

print("\nAccuracy Winners:")
acc_cols = ['null_accuracy', 'lr_accuracy', 'no_learn_accuracy', 'full_accuracy']
winner_counts_acc = {col: 0 for col in acc_cols}
for _, row in test_results.iterrows():
    accs = {col: row[col] for col in acc_cols if not pd.isna(row[col])}
    if accs:
        winner = max(accs, key=accs.get)
        winner_counts_acc[winner] += 1

for col, count in winner_counts_acc.items():
    model_name = col.replace('_accuracy', '').replace('_', ' ').title()
    pct = 100 * count / len(test_results)
    print(f"  {model_name:20} wins: {count}/{len(test_results)} ({pct:.1f}%)")

print("\nLog Loss Winners (lower is better):")
ll_cols = ['null_log_loss', 'lr_log_loss', 'no_learn_log_loss', 'full_log_loss']
winner_counts_ll = {col: 0 for col in ll_cols}
for _, row in test_results.iterrows():
    lls = {col: row[col] for col in ll_cols if not pd.isna(row[col])}
    if lls:
        winner = min(lls, key=lls.get)
        winner_counts_ll[winner] += 1

for col, count in winner_counts_ll.items():
    model_name = col.replace('_log_loss', '').replace('_', ' ').title()
    pct = 100 * count / len(test_results)
    print(f"  {model_name:20} wins: {count}/{len(test_results)} ({pct:.1f}%)")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE")
print("="*70)
