"""Check class balance in robustness validation datasets."""
import pandas as pd
import numpy as np

configs = ['baseline', 'habit_dominant', 'affective_dominant', 'fast_learners', 
           'slow_learners', 'heterogeneous', 'homogeneous', 'goal_dominant',
           'high_social_influence', 'low_social_influence']

print('='*70)
print('CLASS BALANCE ANALYSIS - ROBUSTNESS VALIDATION DATASETS')
print('='*70)

for config in configs:
    try:
        df = pd.read_csv(f'outputs/robustness_temp/events_{config}.csv')
        
        # Check choice distribution
        choices = df['choice_behavior'].value_counts()
        total = len(df)
        
        print(f'\n{config}:')
        print(f'  Total events: {total}')
        print(f'  Choice distribution:')
        for choice, count in choices.items():
            pct = count/total*100
            print(f'    {choice}: {count} ({pct:.1f}%)')
        
        # Check if highly imbalanced
        if len(choices) > 0:
            max_pct = max(choices.values()) / total * 100
            if max_pct > 90:
                print(f'  >>> HIGHLY IMBALANCED: {max_pct:.1f}% majority class!')
            elif max_pct > 80:
                print(f'  >> Moderately imbalanced: {max_pct:.1f}% majority class')
            else:
                print(f'  Balanced: {max_pct:.1f}% majority class')
    except Exception as e:
        print(f'\n{config}: ERROR - {e}')

print('\n' + '='*70)
print('SUMMARY')
print('='*70)
print('\nThis explains why accuracy is high - if one class dominates (>90%),')
print('a null model predicting the majority class will get >90% accuracy!')
