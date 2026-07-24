# CSV Schema Guide (Two-Table Design)

This guide describes the **agent-level** and **situation-level** CSV tables and the accompanying study configuration JSON. It is designed for flexible EMA studies where the number of behaviors, setups, and related agents varies by study.

## 1) Study Config JSON
Use [configs/study_config_template.json](configs/study_config_template.json) as the base. It defines the maximum sizes and which optional blocks are expected. The importer will use this to determine which columns are required.

Key settings:
- `max_behaviors`: max behavior slots per agent
- `max_setups`: max setups per agent
- `max_behavior_relations`: max behavior-relation entries per agent
- `max_related_agents`: max related agents per agent row
- `include_learning_rates`: include learning-rate columns in agent table
- `include_behavior_taxonomy`: include domain/motivational/direction columns
- `include_behavior_relations`: include behavior relation block
- `related_agents_as_rows`: store related agents as separate agent rows
- `related_agents_learning_enabled`: whether related agents update learning parameters
- `include_related_agent_relationships`: include relationship params in main-agent rows
- `situation_has_*`: include optional blocks in situation table
- `relation_similarity_map`: ordinal-to-similarity conversion if you store ordinal codes

If you change the config maxima, **expand the template column blocks accordingly**.

---

## 2) Agent-Level Table
Template: [configs/agent_table_template.csv](configs/agent_table_template.csv)

**Row meaning**: One row = one agent. Use `agent_role` to mark `main` vs `related` agents. Main-agent rows store relationship params; related-agent rows can leave relationship columns blank.

### 2.1 Core columns
- `agent_id`: unique ID for the agent (string)
- `agent_name`: optional label
- `agent_type`: `individual`, `group`, or `culture`
- `agent_role`: `main` or `related`

### 2.2 Behavior block (per behavior_i)
For each `i` from 1..max_behaviors:
- `behavior_i_name`: label for the behavior slot
- Optional taxonomy fields (if enabled):
  - `behavior_i_primary_domain`
  - `behavior_i_motivational_system`
  - `behavior_i_regulatory_direction`
- Optional simulation parameters:
  - `behavior_i_base_outcome` in [-1, 1]
  - `behavior_i_difficulty` in [0, 1]
  - `behavior_i_outcome_volatility` in [0, 1]

### 2.3 Setup block (per setup_j)
For each `j` from 1..max_setups:
- `setup_j_name`
- Optional coordinates if enabled:
  - `setup_j_coord_x`, `setup_j_coord_y`

### 2.4 Agent-Behavior-Setup parameter block
For each (behavior_i, setup_j):
- `behavior_i_setup_j_instinct` in [-1, 1]
- `behavior_i_setup_j_enjoyment` in [-1, 1]
- `behavior_i_setup_j_utility` in [-1, 1]
- `behavior_i_setup_j_exposure_count` in [0, +inf)
- Optional learning parameters (if enabled):
  - `behavior_i_setup_j_alpha_instinct_plus` in [0, 1]
  - `behavior_i_setup_j_alpha_instinct_minus` in [0, 1]
  - `behavior_i_setup_j_alpha_enjoyment` in [0, 1]
  - `behavior_i_setup_j_alpha_utility` in [0, 1]
  - `behavior_i_setup_j_w_enjoyment` in [0, 1]
  - `behavior_i_setup_j_w_utility` in [0, 1]
  - `behavior_i_setup_j_bias_scaling_factor` in [0, 10]

### 2.5 Behavior relation block (optional)
If enabled, for each relation k = 1..max_behavior_relations:
- `behavior_rel_k_a_idx`: index of behavior slot A (1-based)
- `behavior_rel_k_b_idx`: index of behavior slot B (1-based)
- `behavior_rel_k_similarity`: in [-1, 1]

### 2.6 Related agent block (optional)
If enabled, for each related agent r = 1..max_related_agents on a **main-agent** row:
- `related_agent_r_id`
- `related_agent_r_type`
- `related_agent_r_distance` in [0, 1]
- `related_agent_r_receptivity` in [-1, 1]
- `related_agent_r_power` in [-1, 1]
- `related_agent_r_connection` in [-1, 1]

When `related_agents_as_rows` is true, each related agent also appears as its own row with `agent_role=related`.
If `related_agents_learning_enabled` is false, the importer will disable learning updates for `agent_role=related` rows.

Optional extensions (only if config requests):
- related agent behavior slots
- related agent setup slots

---

## 3) Situation-Level Table
Template: [configs/situation_table_template.csv](configs/situation_table_template.csv)

**Row meaning**: One row = one situation or EMA report. If timestamps are used, each row is one timestamped report.

### 3.1 Core columns
- `situation_id`: optional row ID
- `agent_id`: main agent
- `timestamp`: ISO-8601 if enabled
- `interaction_mode`: e.g., `solo`, `observe`, `suggest`, `feedback`
- `setup_idx`: index of the setup slot (1-based)

### 3.2 Environment agent columns (optional)
- `environment_agent_id`
- `environment_agent_type`

### 3.3 Behavior availability/suggestions/feedback (per behavior_i)
For each `i` from 1..max_behaviors (if enabled):
- `behavior_i_available` in {0, 1}
- `behavior_i_suggestion` (float)
- `behavior_i_feedback` (float; can be 0 if not measured)

### 3.4 Outcomes
- `chosen_behavior_idx`: index of chosen behavior slot
- `observed_behavior_idx`: index of observed behavior slot (optional)
- `observed_agent_id`: ID of observed agent (optional)

---

## 4) Indexing Rules
- Behavior slots are **1-based**: behavior_1, behavior_2, ...
- Setup slots are **1-based**: setup_1, setup_2, ...
- Relation indices refer to behavior slots, not behavior IDs.

---

## 5) Missing Data and Defaults
- Leave a cell blank if the measure is missing.
- If a block is disabled in the config, you may omit those columns entirely.
- The importer will treat missing optional values as neutral defaults.

---

## 6) Example Expansion
If your study uses:
- max_behaviors = 4
- max_setups = 2

You will expand the template by adding columns for:
- behavior_3_*, behavior_4_*
- setup_2_*
- behavior_1_setup_2_*, behavior_2_setup_2_*, etc.

---

## 7) Recommended Validation Checks
- Ensure all indices fall within [1, max_behaviors] or [1, max_setups]
- Ensure all numeric ranges are valid
- Ensure all IDs referenced in situations appear in agent table
