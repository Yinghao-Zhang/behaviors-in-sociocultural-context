"""
Diagnostic script to test if the likelihood function is working properly.
Tests whether the likelihood responds meaningfully to parameter changes.
"""
import numpy as np
import pandas as pd
from hierarchical_recovery_v3 import load_data, simulate_forward_numpy

def test_likelihood_sensitivity():
    """Test if likelihood changes with different parameters."""
    
    print("="*70)
    print("LIKELIHOOD SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Load data
    per_person, ppl = load_data("outputs/ema_events.csv", "outputs/ema_people.csv")
    
    # Test on first person
    pp = per_person[0]
    pid = pp['person_id']
    
    # Get true parameters from person data
    true_row = ppl[ppl.person_id == pid].iloc[0]
    true_params = {
        'w_I': true_row['w_I'],
        'w_E': true_row['w_E'],
        'w_U': true_row['w_U'],
        'aI_pos': true_row['alpha_I_pos'],
        'aI_neg': true_row['alpha_I_neg'],
        'a_E': true_row['alpha_E'],
        'a_U': true_row['alpha_U']
    }
    
    print(f"\nPerson {pid}:")
    print(f"  Observations: {pp['T']}")
    print(f"  True parameters: {true_params}")
    
    # Test 1: True parameters
    print("\n" + "-"*70)
    print("TEST 1: True Parameters")
    print("-"*70)
    logp_true = simulate_forward_numpy(
        true_params['w_I'], true_params['w_E'], true_params['w_U'],
        true_params['aI_pos'], true_params['aI_neg'], 
        true_params['a_E'], true_params['a_U'],
        3.0,
        pp['inst0'], pp['enj0'], pp['uti0'],
        pp['suggestion'], pp['choice'], pp['e_out'], pp['u_out'],
        sigma_e=0.3, sigma_u=0.3
    )
    print(f"Log probability: {logp_true:.2f}")
    
    # Test 2: Random parameters
    print("\n" + "-"*70)
    print("TEST 2: Random Parameters (uniform 0.5)")
    print("-"*70)
    logp_random = simulate_forward_numpy(
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        3.0,
        pp['inst0'], pp['enj0'], pp['uti0'],
        pp['suggestion'], pp['choice'], pp['e_out'], pp['u_out'],
        sigma_e=0.3, sigma_u=0.3
    )
    print(f"Log probability: {logp_random:.2f}")
    print(f"Difference from true: {logp_random - logp_true:.2f}")
    
    # Test 3: Vary one parameter at a time
    print("\n" + "-"*70)
    print("TEST 3: Parameter Sensitivity (varying one at a time)")
    print("-"*70)
    
    param_names = ['w_I', 'w_E', 'w_U', 'aI_pos', 'aI_neg', 'a_E', 'a_U']
    test_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for param_name in param_names:
        print(f"\nVarying {param_name}:")
        logps = []
        for val in test_values:
            params = [true_params[p] for p in param_names]
            idx = param_names.index(param_name)
            params[idx] = val
            
            logp = simulate_forward_numpy(
                *params, 3.0,
                pp['inst0'], pp['enj0'], pp['uti0'],
                pp['suggestion'], pp['choice'], pp['e_out'], pp['u_out'],
                sigma_e=0.3, sigma_u=0.3
            )
            logps.append(logp)
            print(f"  {param_name}={val:.1f} -> logp={logp:.2f}")
        
        # Check if there's variation
        logp_range = max(logps) - min(logps)
        print(f"  Range: {logp_range:.2f} {'✓ Sensitive' if logp_range > 1 else '⚠ Low sensitivity'}")
    
    # Test 4: Check likelihood across all people
    print("\n" + "-"*70)
    print("TEST 4: Likelihood Across All People")
    print("-"*70)
    
    all_logps_true = []
    all_logps_random = []
    
    for i, pp in enumerate(per_person[:10]):  # First 10 people
        pid = pp['person_id']
        true_row = ppl[ppl.person_id == pid].iloc[0]
        
        logp_t = simulate_forward_numpy(
            true_row['w_I'], true_row['w_E'], true_row['w_U'],
            true_row['alpha_I_pos'], true_row['alpha_I_neg'],
            true_row['alpha_E'], true_row['alpha_U'],
            3.0,
            pp['inst0'], pp['enj0'], pp['uti0'],
            pp['suggestion'], pp['choice'], pp['e_out'], pp['u_out'],
            sigma_e=0.3, sigma_u=0.3
        )
        
        logp_r = simulate_forward_numpy(
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            3.0,
            pp['inst0'], pp['enj0'], pp['uti0'],
            pp['suggestion'], pp['choice'], pp['e_out'], pp['u_out'],
            sigma_e=0.3, sigma_u=0.3
        )
        
        all_logps_true.append(logp_t)
        all_logps_random.append(logp_r)
        print(f"Person {pid}: True={logp_t:7.2f}, Random={logp_r:7.2f}, Diff={logp_t-logp_r:+7.2f}")
    
    print(f"\nAverage difference: {np.mean(np.array(all_logps_true) - np.array(all_logps_random)):.2f}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if logp_true > logp_random + 5:
        print("✓ Likelihood appears to be working (true params >> random params)")
    elif abs(logp_true - logp_random) < 1:
        print("⚠ WARNING: Likelihood barely distinguishes true from random parameters")
        print("  This explains why the sampler cannot recover parameters!")
    else:
        print("? Unclear - likelihood shows some difference but may not be strong enough")
    
    print()

if __name__ == "__main__":
    test_likelihood_sensitivity()
