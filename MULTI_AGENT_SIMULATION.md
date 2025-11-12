# Multi-Agent Social Interaction Simulation

## Overview

The updated `ema_pair_simulation.py` now implements a **multi-agent framework** aligned with the situation types defined in `situation.py`. This addresses the theoretical requirement that interpersonal behaviors (conflict avoidance/approach) need social feedback and learning mechanisms.

## Architecture

### Key Components

1. **Main Agents (N=50)**
   - Learn from experience (α > 0)
   - Update instinct, enjoyment, and utility beliefs through:
     - Direct experience
     - Observational learning (with penalty)
     - Social suggestions
     - Social feedback

2. **Social Partners (N=80-120)**
   - Fixed parameters (no learning, α = 0)
   - Have full value systems: w_I, w_E, w_U, instinct, enjoyment, utility
   - Make behavioral choices based on their parameters
   - Provide suggestions and feedback based on their beliefs
   - Represent the "social environment"

3. **Relationship Parameters** (from main agent's perspective)
   - **Receptivity** (R): How open main agent is to partner's influence [-0.2, 0.9]
   - **Communion** (C): Affective quality of relationship [-0.5, 0.8]
   - **Presence Salience** (P): How salient partner's presence is [0.2, 1.0]

## Four Situation Types

Each EMA event is one of four types, aligned with `situation.py`:

### 1. **Observe** (p = 0.20)
- **Description**: Main agent observes partner's behavioral choice
- **Mechanism**: Observational learning with penalty (0.5)
- **Learning Updates**:
  ```
  Δinstinct = (α_I * penalty) * (target - current)
  Δenjoyment = (α_E * penalty) * (outcome - prediction)
  Δutility = (α_U * penalty) * (outcome - prediction)
  ```
- **Based on**: `situation.py::_simulate_observational()` lines 300-420

### 2. **Solitary** (p = remainder, ~0.20)
- **Description**: Main agent chooses behavior alone, no social influence
- **Mechanism**: Standard decision-making and learning
- **Learning Updates**: Full learning rates (no penalty)
- **Based on**: `situation.py::_simulate_solitary()` lines 92-128

### 3. **Suggest** (p = 0.30)
- **Description**: Partner provides suggestion → main agent incorporates it → chooses
- **Mechanism**: Suggestion influence on choice values
  ```
  suggestion[behavior] = partner_choice_value * receptivity
  choice_value = H + E_eval + suggestion + noise
  ```
- **Learning Updates**: Standard learning from own experience
- **Based on**: `situation.py::_simulate_observational()` with `observer_suggestion=True`

### 4. **Observe_Feedback** (p = 0.30)
- **Description**: Main agent chooses → partner provides feedback
- **Mechanism**: Social modulation of outcomes
  ```
  social_enjoyment = base_enjoyment + communion * presence_salience
  feedback = (base_utility - partner_utility_prediction) + mood
  social_utility = base_utility + feedback * receptivity
  ```
- **Learning Updates**: Learn from socially modulated outcomes
- **Based on**: `situation.py::_simulate_observational()` with `observer_feedback=True`

## Simulation Output (N=50, T=12-20)

```
Total events: 805
Main agents: 50

Situation Type Distribution:
  observe_feedback: 246 (30.6%)
  suggest:          237 (29.4%)
  observe:          161 (20.0%)
  solitary:         161 (20.0%)

Social Partner Involvement:
  Events with partner: 644 (80.0%)
  Events without partner: 161 (20.0%)
  Unique partners: 82

Behavior Distribution:
  avoid_conflict:          184 (22.9%)
  approach_conflict_care:  460 (57.1%)
  observation only:        161 (20.0%)

Outcomes:
  Enjoyment: mean=0.392, std=0.289
  Utility:   mean=0.413, std=0.308
```

## Theoretical Justification

### Why Multi-Agent Simulation?

1. **Interpersonal Nature**: Conflict avoidance/approach behaviors are inherently interpersonal
2. **Social Feedback**: "Approach conflict with care" has higher outcome variance due to immediate social response
3. **Observational Learning**: People learn vicariously by watching others handle conflicts
4. **Suggestion Influence**: Social partners provide advice that influences decision-making
5. **Feedback Effects**: Partner reactions modulate the utility of behavioral choices

### Asymmetric Learning Design

- **Main Agents (α > 0)**: Learn and adapt over time
- **Social Partners (α = 0)**: Fixed parameters, represent stable social environment
- **Rationale**: 
  - Simplifies dynamics while maintaining realism
  - Partners are the "environment" that main agents learn to navigate
  - Focuses analysis on main agents' learning trajectories

## Key Differences from Previous Version

| Feature | Old (ema_pair_simulation.py) | New (Multi-Agent) |
|---------|------------------------------|-------------------|
| Social partners | 0-3 per person with basic beliefs | 80-120 shared pool with full value systems |
| Interaction types | Mixed (suggestion + feedback combined) | Explicit 4 types (observe, solitary, suggest, feedback) |
| Observational learning | ❌ Not implemented | ✅ Implemented with penalty |
| Partner behavior | No partner choices | Partners make choices based on their parameters |
| Situation alignment | Custom implementation | Aligned with `situation.py` framework |

## Data Structure

### ema_events.csv columns:
- `person_id`, `t`: Agent and trial identifiers
- `situation_type`: One of {observe, solitary, suggest, observe_feedback}
- `choice_behavior`: Chosen behavior (or None for observe events)
- `observed_behavior`: Behavior observed (only for observe events)
- `partner_id`: Social partner ID (or None for solitary)
- `enjoyment_out`, `utility_out`: Experienced outcomes
- `instinct_*`, `enjoyment_*`, `utility_*`: Current beliefs after update
- `choice_prob`: Probability of chosen action (for suggest/feedback/solitary)
- `suggestion_strength`, `feedback_strength`: Social influence magnitudes

### ema_people.csv columns:
- `person_id`: Agent identifier
- `w_I`, `w_E`, `w_U`: Value system weights
- `tau`, `noise_s`: Decision parameters
- `alpha_*`: Learning rates
- `instinct_*_0`, `enjoyment_*_0`, `utility_*_0`: Initial beliefs

## Usage

```bash
python experiments/ema_pair_simulation.py --N 50 --tmin 12 --tmax 20 --seed 42
```

Parameters:
- `--N`: Number of main agents (default: 50)
- `--tmin`, `--tmax`: Trial range per agent (default: 5-20)
- `--seed`: Random seed (default: 42)
- `--outdir`: Output directory (default: outputs/)

## Next Steps

1. **Validation**: Re-run between-person prediction validation on multi-agent data
2. **Analysis**: Compare learning trajectories across situation types
3. **Hypotheses**: Test if approach behavior variance is indeed higher in feedback conditions
4. **Extensions**: 
   - Add partner mood/disposition dynamics
   - Implement social network structure (repeated partners)
   - Add cultural group effects

## References

- `situation.py`: Defines the 4 situation types and their mechanics
- `agent.py`: Agent architecture with value systems and learning
- `validate_predictions_between_person.py`: Validation framework for testing model
