data {
  int<lower=1> P;                      // persons
  int<lower=1> B;                      // behaviors (=2)
  int<lower=1> E;                      // total events
  int<lower=1, upper=P> person[E];     // person index per event
  int<lower=0, upper=1> choice[E];     // chosen behavior (0 or 1)
  int<lower=0> T_per_person[P];        // number of events per person (order preserved)
  // Baselines
  matrix[P,B] instinct0;
  matrix[P,B] enjoyment0;
  matrix[P,B] utility0;
  // Observed outcomes (bounded [-1,1])
  vector[E] enjoyment_out;
  vector[E] utility_out;
  // Suggestion terms per event & behavior
  matrix[E,B] suggestion_term;
  // Fixed tau
  real<lower=0> tau_fixed;
}
parameters {
  // Raw hierarchical
  vector[P] wI_raw;
  vector[P] wE_raw;
  vector[P] wU_raw;

  vector[P] noise_raw;
  vector[P] aIpos_raw;
  vector[P] aIneg_raw;
  vector[P] aE_raw;
  vector[P] aU_raw;

  // Hyperparameters (means and scales)
  real mu_wI; real<lower=0> sigma_wI;
  real mu_wE; real<lower=0> sigma_wE;
  real mu_wU; real<lower=0> sigma_wU;

  real mu_noise; real<lower=0> sigma_noise;

  real mu_aIpos; real<lower=0> sigma_aIpos;
  real mu_aIneg; real<lower=0> sigma_aIneg;
  real mu_aE;    real<lower=0> sigma_aE;
  real mu_aU;    real<lower=0> sigma_aU;

  real<lower=0> sigma_out_enj;
  real<lower=0> sigma_out_uti;
}
transformed parameters {
  // Constrained transformations
  vector[P] w_I;
  vector[P] w_E;
  vector[P] w_U;
  vector[P] noise_s;
  vector[P] aI_pos;
  vector[P] aI_neg;
  vector[P] a_E;
  vector[P] a_U;

  for(p in 1:P){
    w_I[p]    = 1.5 * inv_logit(wI_raw[p]);
    w_E[p]    = 1.5 * inv_logit(wE_raw[p]);
    w_U[p]    = 1.5 * inv_logit(wU_raw[p]);
    noise_s[p]= 0.5 * inv_logit(noise_raw[p]);
    aI_pos[p] = inv_logit(aIpos_raw[p]);
    aI_neg[p] = inv_logit(aIneg_raw[p]);
    a_E[p]    = inv_logit(aE_raw[p]);
    a_U[p]    = inv_logit(aU_raw[p]);
  }
}
model {
  // Hyperpriors
  mu_wI ~ normal(0, 0.5);   sigma_wI ~ exponential(2);
  mu_wE ~ normal(0, 0.5);   sigma_wE ~ exponential(2);
  mu_wU ~ normal(0, 0.5);   sigma_wU ~ exponential(2);

  mu_noise ~ normal(0, 0.5); sigma_noise ~ exponential(2);

  mu_aIpos ~ normal(0, 0.5); sigma_aIpos ~ exponential(2);
  mu_aIneg ~ normal(0, 0.5); sigma_aIneg ~ exponential(2);
  mu_aE    ~ normal(0, 0.5); sigma_aE    ~ exponential(2);
  mu_aU    ~ normal(0, 0.5); sigma_aU    ~ exponential(2);

  sigma_out_enj ~ exponential(2);
  sigma_out_uti ~ exponential(2);

  // Person-level priors (raw scale)
  wI_raw   ~ normal(mu_wI, sigma_wI);
  wE_raw   ~ normal(mu_wE, sigma_wE);
  wU_raw   ~ normal(mu_wU, sigma_wU);

  noise_raw ~ normal(mu_noise, sigma_noise);

  aIpos_raw ~ normal(mu_aIpos, sigma_aIpos);
  aIneg_raw ~ normal(mu_aIneg, sigma_aIneg);
  aE_raw    ~ normal(mu_aE, sigma_aE);
  aU_raw    ~ normal(mu_aU, sigma_aU);

  // Sequential likelihood
  {
    // Allocate mutable copies of latent states
    matrix[P,B] inst = instinct0;
    matrix[P,B] enj  = enjoyment0;
    matrix[P,B] uti  = utility0;

    int idx_start = 1;
    for(p in 1:P){
      int nT = T_per_person[p];
      for(k in 0:(nT-1)){
        int e_idx = idx_start + k; // global event index

        // Compute choice values (deterministic approx)
        vector[B] CV;
        for(b in 1:B){
          real H = w_I[p] * inst[p,b];
          real Eval = w_E[p] * enj[p,b] + w_U[p] * uti[p,b];
          CV[b] = H + Eval + suggestion_term[e_idx,b];
        }
        // Softmax with fixed tau
        vector[B] logits = tau_fixed * CV;
        // subtract max for stability
        logits = logits - max(logits);
        vector[B] probs;
        for(b in 1:B) probs[b] = exp(logits[b]);
        probs = probs / sum(probs);

        target += categorical_lpmf(choice[e_idx]+1 | probs); // choice is 0/1 -> add 1

        // Outcome likelihood before update (truncate approx: just clamp support)
        // enjoyment_out[e_idx] ~ Normal(enj[p, chosen]+social? simplified)
        // We use the enjoyment prediction for chosen behavior prior to update
        int chosen_b = choice[e_idx] + 1;
        target += normal_lpdf(enjoyment_out[e_idx] | enj[p, chosen_b], sigma_out_enj);
        target += normal_lpdf(utility_out[e_idx]   | uti[p, chosen_b], sigma_out_uti);

        // Learning updates
        for(b in 1:B){
          if(b == chosen_b){
            inst[p,b] += aI_pos[p] * (1 - inst[p,b]);
          } else {
            inst[p,b] += aI_neg[p] * (-1 - inst[p,b]);
          }
        }
        // Enjoyment & Utility prediction error for chosen only
        enj[p, chosen_b] += a_E[p] * (enjoyment_out[e_idx] - enj[p, chosen_b]);
        uti[p, chosen_b] += a_U[p] * (utility_out[e_idx]   - uti[p, chosen_b]);

        // Clip (soft) by penalizing if outside [-1,1]
        for(b in 1:B){
          target += normal_lpdf(inst[p,b] | 0, 2) - normal_lpdf(clamp(inst[p,b], -1, 1) | 0, 2);
          target += normal_lpdf(enj[p,b]  | 0, 2) - normal_lpdf(clamp(enj[p,b], -1, 1)  | 0, 2);
          target += normal_lpdf(uti[p,b]  | 0, 2) - normal_lpdf(clamp(uti[p,b], -1, 1)  | 0, 2);
        }
      }
      idx_start += nT;
    }
  }
}
generated quantities {
  // For posterior summaries
  vector[P] w_I_post = w_I;
  vector[P] w_E_post = w_E;
  vector[P] w_U_post = w_U;
  vector[P] noise_s_post = noise_s;
  vector[P] aI_pos_post = aI_pos;
  vector[P] aI_neg_post = aI_neg;
  vector[P] a_E_post = a_E;
  vector[P] a_U_post = a_U;
}