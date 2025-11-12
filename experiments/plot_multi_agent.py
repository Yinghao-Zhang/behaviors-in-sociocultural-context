"""
Visualize multi-agent simulation dynamics across the 4 situation types.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_csv('outputs/ema_events.csv')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Multi-Agent Simulation: 4 Situation Types', fontsize=16, fontweight='bold')

# 1. Situation type distribution
ax1 = axes[0, 0]
situation_counts = df['situation_type'].value_counts().sort_values(ascending=True)
colors = {'observe': '#3498db', 'solitary': '#95a5a6', 'suggest': '#2ecc71', 'observe_feedback': '#e74c3c'}
bar_colors = [colors[s] for s in situation_counts.index]
situation_counts.plot(kind='barh', ax=ax1, color=bar_colors)
ax1.set_title('(A) Situation Type Distribution', fontweight='bold')
ax1.set_xlabel('Count')
ax1.set_ylabel('')
for i, v in enumerate(situation_counts.values):
    ax1.text(v + 5, i, f'{v} ({100*v/len(df):.1f}%)', va='center')

# 2. Behavior choices by situation type
ax2 = axes[0, 1]
# Only include situations where main agent makes a choice
choice_data = df[df['choice_behavior'].notna()].copy()
ct = pd.crosstab(choice_data['situation_type'], choice_data['choice_behavior'], normalize='index') * 100
ct.plot(kind='bar', ax=ax2, color=['#e67e22', '#9b59b6'], rot=45)
ax2.set_title('(B) Behavior Choices by Situation Type', fontweight='bold')
ax2.set_ylabel('Percentage (%)')
ax2.set_xlabel('')
ax2.legend(title='Behavior', labels=['Avoid', 'Approach'], loc='upper right')
ax2.axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# 3. Outcome distributions by situation type
ax3 = axes[1, 0]
situation_order = ['observe', 'solitary', 'suggest', 'observe_feedback']
df_melted = df[df['situation_type'].isin(situation_order)].copy()
df_melted['Enjoyment'] = df_melted['enjoyment_out']
df_melted['Utility'] = df_melted['utility_out']
outcome_data = df_melted.melt(id_vars=['situation_type'], value_vars=['Enjoyment', 'Utility'],
                                var_name='Outcome Type', value_name='Value')
sns.violinplot(data=outcome_data, x='situation_type', y='Value', hue='Outcome Type', 
               ax=ax3, split=True, inner='quartile', order=situation_order)
ax3.set_title('(C) Outcome Distributions by Situation Type', fontweight='bold')
ax3.set_xlabel('Situation Type')
ax3.set_ylabel('Outcome Value')
ax3.set_xticklabels(['Observe', 'Solitary', 'Suggest', 'Feedback'], rotation=45)
ax3.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax3.legend(title='Outcome', loc='upper right')

# 4. Learning trajectories (instinct for approach behavior)
ax4 = axes[1, 1]
# Sample 10 random agents
sample_agents = np.random.choice(df['person_id'].unique(), size=10, replace=False)
for pid in sample_agents:
    agent_data = df[df['person_id'] == pid].sort_values('t')
    ax4.plot(agent_data['t'], agent_data['instinct_approach_conflict_care'], 
             alpha=0.6, linewidth=1.5)
ax4.set_title('(D) Instinct Learning Trajectories (10 Random Agents)', fontweight='bold')
ax4.set_xlabel('Trial (t)')
ax4.set_ylabel('Instinct for Approach Behavior')
ax4.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax4.set_ylim(-1, 1)
ax4.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('outputs/multi_agent_simulation.png', dpi=300, bbox_inches='tight')
print("Saved: outputs/multi_agent_simulation.png")

# Additional statistics
print("\n" + "="*60)
print("MULTI-AGENT SIMULATION STATISTICS")
print("="*60)

print("\nOutcome Variance by Situation Type:")
for sit_type in ['observe', 'solitary', 'suggest', 'observe_feedback']:
    sit_data = df[df['situation_type'] == sit_type]
    e_std = sit_data['enjoyment_out'].std()
    u_std = sit_data['utility_out'].std()
    print(f"  {sit_type:17} → Enjoyment σ={e_std:.3f}, Utility σ={u_std:.3f}")

print("\nApproach vs Avoid Choice Rates:")
choice_data = df[df['choice_behavior'].notna()]
for sit_type in ['solitary', 'suggest', 'observe_feedback']:
    sit_data = choice_data[choice_data['situation_type'] == sit_type]
    if len(sit_data) > 0:
        approach_rate = (sit_data['choice_behavior'] == 'approach_conflict_care').mean()
        print(f"  {sit_type:17} → Approach: {100*approach_rate:.1f}%, Avoid: {100*(1-approach_rate):.1f}%")

print("\nSocial Partner Usage:")
for sit_type in ['observe', 'suggest', 'observe_feedback']:
    sit_data = df[df['situation_type'] == sit_type]
    unique_partners = sit_data['partner_id'].nunique()
    avg_partner_events = len(sit_data) / unique_partners if unique_partners > 0 else 0
    print(f"  {sit_type:17} → {unique_partners} unique partners, avg {avg_partner_events:.1f} events/partner")
