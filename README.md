# F1 RL Safety: Reinforcement Learning Architectures and Reward Regimes

A research codebase for studying how reinforcement-learning agents learn Formula 1 race strategy under different performance, rule, and safety objectives. The project uses a custom Gymnasium simulator of the 2024 British Grand Prix at Silverstone and compares five RL architectures across three reward regimes, producing a 5×3 experimental grid.

> **Research status:** exploratory MRes research prototype. The current comparison is a single-seed, fixed-budget study intended to expose behavioural patterns and generate hypotheses. It is not a validated race-prediction system or a production strategy tool.

## Research question

The central question is:

> How do RL architectures learn different pit-stop and driving-risk strategies when they receive the same race environment but different objective regimes, and which architectural choices best balance performance, rule compliance, and safety?

The study began as an investigation of reward design: whether an agent trained only for performance would behave differently from one exposed to explicit rulebook and safety constraints. It was subsequently expanded to ask whether those differences are mediated by the learning architecture itself.

The current research design therefore crosses:

- **Five architectures:** PPO, A2C, DQN, SARSA, and REINFORCE.
- **Three regimes:** unconstrained performance, rulebook-aware, and safety-constrained.
- **Common environment:** a strategic single-car Silverstone race simulator.
- **Additional analysis:** evaluation metrics, pit-stop behaviour, global comparison tables, and SHAP-based surrogate explanations of state-feature influence.

A longer-term direction is a mixture-of-experts controller that delegates decisions to specialist policies according to the current race context and safety/performance requirements.

## What the project studies

This repository is not a multi-agent racing project. It is a single-agent study of objective specification and architectural inductive bias. The agent controls strategic decisions over a simulated race, while the experiment examines:

1. How reward semantics shape learned behaviour.
2. How actor-critic, value-based, on-policy TD, and Monte-Carlo policy-gradient methods differ under the same regime.
3. Whether performance improvements are associated with unsafe, rule-violating, or brittle behaviour.
4. Which state variables are most influential in the learned decision policy.
5. Whether complementary specialists could be combined into a more reliable mixture of models.

## Environment

The core environment is implemented in `src/f1_rl_safety/f1_env.py` as a Gymnasium-compatible race simulator. It models a 52-lap Silverstone race at the level of strategic decisions rather than low-level vehicle control.

### Data and calibration

The simulator uses `data/silverstone_2024_laps.csv`, produced from FastF1-derived 2024 British Grand Prix lap data. The data loader supplies race and tyre-related calibration inputs, with fallback behaviour intended to keep experiments runnable when a data field is unavailable.

The model represents quantities such as:

- Base lap time and race progress.
- Tyre-compound effects.
- Tyre degradation and tyre age.
- Pit-lane time loss.
- Fuel and race-time effects.
- Track-status conditions.
- Position and gaps to other cars.

The simulator should be interpreted as a controlled experimental model. It is designed to compare policies under consistent dynamics, not to reproduce the complete stochasticity, multi-car interaction, weather evolution, or operational complexity of a real Grand Prix.

### Observation space

The observation is a compact numerical vector containing normalised race and strategy variables. The exact ordering is defined by the environment implementation and should be treated as the authoritative feature schema when interpreting SHAP outputs. The state includes variables covering:

- Race progress and lap fraction.
- Current race time.
- Position and gaps ahead/behind.
- Tyre age and wear.
- Fuel state.
- Track status encoding.
- Current tyre compound encoding.
- Risk/aggression state.
- Number of pit stops.

When adding analysis code, do not rename or reorder features informally: use the environment’s declared feature names so plots remain traceable to the actual state vector.

### Action space

The native strategic action is composite, containing:

- A pit-stop decision.
- A tyre-compound choice.
- A continuous risk/aggression value.

PPO, A2C, and REINFORCE operate through the native continuous-action interface. DQN and SARSA use `DiscreteF1ActionWrapper` in `src/f1_rl_safety/wrappers.py`, which maps a discrete action index to a predefined combination of pit decision, tyre choice, and risk bin. This makes value-based methods compatible with the simulator but introduces a discretisation assumption that must be acknowledged when comparing them with continuous-action agents.

## Reward regimes

The reward definitions are implemented in `RaceRegime` and the reward logic in `f1_env.py`. The experiment intentionally keeps environment dynamics fixed while changing the objective signal.

### Unconstrained / performance-oriented

This regime prioritises competitive race outcomes, including lap-time and position-related performance, with comparatively weak penalties for risky behaviour and failures. It is a deliberately permissive baseline for investigating whether an agent discovers aggressive or specification-gaming strategies, including high-risk driving and zero-stop solutions where the simulator permits them.

### Rulebook

This regime adds explicit rule-oriented incentives and penalties. It represents requirements such as completing the required pit activity while discouraging excessive pitting. Crashes and high-risk behaviour are penalised more strongly than in the unconstrained condition.

The rulebook regime tests whether an agent learns to satisfy explicit constraints when those constraints are represented through reward shaping rather than a hard safety shield.

### Safe / safety-constrained

This regime increases the cost of crashes, catastrophic events, excessive risk, and excessive tyre wear. It also penalises under-pitting and over-pitting. The intended result is a more conservative policy, potentially accepting additional race time or pit activity in exchange for reduced risk.

These regimes are not interchangeable labels for “easy”, “medium”, and “hard”. They are different objective specifications. The project’s key assumption is that changing the reward while holding the simulator fixed reveals the behavioural consequences of objective design.

## RL architectures

### PPO

Proximal Policy Optimization is the principal reference architecture. It is an on-policy actor-critic method implemented with Stable-Baselines3 and an MLP policy. PPO provides the main comparison point because it was used in the original three-regime experiment.

### A2C

Advantage Actor-Critic provides a simpler synchronous actor-critic comparison. It shares the policy/value-function framing of PPO but does not use PPO’s clipped policy objective.

### DQN

Deep Q-Network is an off-policy value-based baseline. It requires the discrete action wrapper and learns a neural approximation to action values over the discretised strategic action set.

### SARSA

SARSA is an on-policy temporal-difference control method implemented for the discrete action representation. Unlike Q-learning/DQN, its update follows the action actually selected by the current policy, making it useful for examining the effect of on-policy learning under risky exploration.

### REINFORCE

REINFORCE is an episodic Monte-Carlo policy-gradient baseline operating on the native action representation. It updates from complete race returns and is therefore expected to be sensitive to episode length, return variance, and the relatively small number of training episodes used in the current exploratory run.

## Experimental design

The principal comparison is a 5×3 grid:

| Architecture | Unconstrained | Rulebook | Safe |
|---|---:|---:|---:|
| PPO | ✓ | ✓ | ✓ |
| A2C | ✓ | ✓ | ✓ |
| DQN | ✓ | ✓ | ✓ |
| SARSA | ✓ | ✓ | ✓ |
| REINFORCE | ✓ | ✓ | ✓ |

The current committed evaluation artefacts use:

- 50,000 environment steps for PPO, A2C, DQN, and SARSA.
- 200 training episodes for REINFORCE.
- Seed 0.
- 50 deterministic evaluation episodes per evaluated policy, as recorded in the result-generation workflow.

Because REINFORCE is reported in episodes while the other methods are reported in environment steps, its result is not a strictly matched interaction-budget comparison. This is a current limitation and should be corrected in a stronger thesis experiment by logging and matching environment transitions across all methods.

### Evaluation metrics

The evaluation CSVs contain the following main measures:

- `finish_position`: mean finishing position.
- `race_time`: mean simulated race time.
- `crashes`: crash rate or mean crash indicator, depending on the evaluation aggregation.
- `catastrophic`: catastrophic-event rate.
- `pitstops`: mean number of pit stops.
- `mean_risk`: mean risk/aggression value.
- `pitstop_distribution`: distribution of pit-stop counts across evaluation episodes.

These should be interpreted jointly. A lower race time is not automatically a better policy if it is obtained through a higher crash rate, rule violation, or pathological pit strategy.

## Current results snapshot

The current committed result files are under `data/experiment_results/`. The following table records the single-seed exploratory snapshot used to generate the comparison outputs. Race time is in simulator time units; rates are reported as stored in the CSVs.

| Architecture | Regime | Finish position | Race time | Crashes | Catastrophic | Pit stops | Mean risk |
|---|---|---:|---:|---:|---:|---:|---:|
| PPO | Unconstrained | 9.42 | 1749.31 | 0.96 | 0.32 | 0.00 | 1.000 |
| PPO | Rulebook | 14.06 | 3052.12 | 0.76 | 0.08 | 0.00 | 0.143 |
| PPO | Safe | 15.14 | 3375.38 | 0.78 | 0.06 | 0.00 | -0.140 |
| A2C | Unconstrained | 9.42 | 1749.31 | 0.96 | 0.32 | 0.00 | 1.000 |
| A2C | Rulebook | 16.50 | 3259.93 | 0.76 | 0.08 | 0.04 | -0.514 |
| A2C | Safe | 16.84 | 3264.25 | 0.82 | 0.14 | 0.00 | -0.612 |
| DQN | Unconstrained | 9.42 | 1749.31 | 0.96 | 0.32 | 0.00 | 1.000 |
| DQN | Rulebook | 13.96 | 3863.59 | 0.48 | 0.04 | 1.80 | -0.224 |
| DQN | Safe | 14.24 | 2780.34 | 0.84 | 0.04 | 0.00 | -0.122 |
| SARSA | Unconstrained | 9.40 | 2106.72 | 0.84 | 0.24 | 1.94 | 0.924 |
| SARSA | Rulebook | 10.90 | 2185.67 | 0.94 | 0.22 | 0.00 | 0.500 |
| SARSA | Safe | 14.40 | 3320.12 | 0.74 | 0.06 | 0.00 | 0.000 |
| REINFORCE | Unconstrained | 16.68 | 3700.34 | 0.60 | 0.04 | 16.52 | 0.243 |
| REINFORCE | Rulebook | 17.48 | 4178.99 | 0.48 | 0.04 | 19.26 | 0.029 |
| REINFORCE | Safe | 17.02 | 3584.87 | 0.64 | 0.10 | 15.32 | 0.437 |

### Interpreting the snapshot

Several cautious observations motivate the next phase:

1. The unconstrained PPO, A2C, and DQN policies converge to very similar fast but highly risky behaviour in this snapshot: short race times, no pit stops, mean risk near 1, and high crash/catastrophic rates. This suggests that the reward regime is dominating architecture in that condition.
2. DQN produces the lowest recorded rulebook crash and catastrophic rates in the snapshot, but at a very high race time and with more pit stops. This is a useful example of why performance and safety must be reported together.
3. SARSA is comparatively competitive on race time in the rulebook condition but has high crash and catastrophic rates, illustrating that fast completion does not imply rule or safety compliance.
4. REINFORCE makes many pit stops and has the slowest race times in the rulebook and safe conditions. This may reflect conservative or unstable episodic policy-gradient learning, but the result should not be over-interpreted without matched interaction budgets and additional seeds.
5. The current values are not sufficient to establish an architecture ranking. They are hypothesis-generating evidence from one seed and a small exploratory budget.

## Explainability and SHAP

The explainability pipeline is implemented in `src/f1_rl_safety/shap_surrogates.py` and supported by `scripts/analyse_shap.py` and `scripts/analyse_global_importance.py`.

The pipeline records state/action examples from trained agents, trains supervised surrogate models to approximate the agent’s action decisions, and applies SHAP to the surrogate. The resulting artefacts are intended to answer:

- Which state variables most influence decisions?
- Do important variables change between reward regimes?
- Do architectures attend to different parts of the race state?
- Are safety-oriented policies more sensitive to tyre wear, risk, or pit count?

Generated files include per-agent/regime SHAP CSVs, per-architecture and per-regime rankings, cross-architecture matrices, overall feature rankings, and interactive HTML plots in `output/`.

SHAP explanations here are explanations of a surrogate approximation to the policy, not direct causal explanations of the RL algorithm. They should therefore be checked against surrogate fidelity, action-distribution coverage, feature correlations, and domain knowledge. A high SHAP value indicates influence within the surrogate’s prediction task; it does not by itself prove that changing a feature would cause the same policy change in the simulator.

## Mixture-of-experts direction

The comparison motivates a mixture-of-experts architecture rather than assuming one algorithm is uniformly best.

A proposed controller would contain specialist policies such as:

- A performance expert, trained primarily under the unconstrained regime.
- A rule-compliance expert, trained under the rulebook regime.
- A safety expert, trained under the safe regime.
- Potentially separate architecture specialists, such as PPO/A2C for continuous actor-critic control and DQN/SARSA for discretised strategic decisions.

A gating network would observe the current race state and select or weight experts. Candidate gating inputs include tyre wear, tyre age, risk, track status, race progress, pit count, position, and time gaps. The gate could be trained using a constrained objective that balances race time, rule compliance, and safety, or designed as a transparent rule-based selector before being learned.

The mixture-of-experts hypothesis is:

> Policies trained under different objectives may be complementary rather than directly interchangeable; a state-dependent gate could use the performance specialist in low-risk contexts and shift toward rule/safety specialists as tyre degradation, risk, or incident probability increases.

This must be tested rather than assumed. Important baselines are a single PPO policy, a single best-performing specialist, uniform expert averaging, a rule-based gate, and a learned gate. Evaluation should include gate stability, expert utilisation, safety violations, and whether the mixture merely reproduces the safest policy at a large performance cost.

## Repository structure

```text
f1-rl-safety/
├── archive/                         Historical or superseded material
├── configs/                         Experiment configuration placeholders
├── data/
│   ├── silverstone_2024_laps.csv    FastF1-derived calibration data
│   ├── experiment_results/           Evaluation CSVs for the 5×3 grid
│   ├── shap/                         Per-agent SHAP artefacts
│   ├── processed/                    Processed data
│   └── cache/                        Cached data
├── logs/                             Training logs and TensorBoard outputs
├── models/                           Saved trained policies and checkpoints
├── notebooks/                        Exploratory notebooks
├── output/                           Aggregated CSVs and interactive HTML plots
├── scripts/
│   ├── analyse_results.py            Aggregate evaluation results
│   ├── analyse_shap.py               Aggregate SHAP results
│   └── analyse_global_importance.py  Cross-architecture/regime analysis
├── src/f1_rl_safety/
│   ├── f1_env.py                     Gymnasium race environment and regimes
│   ├── data_loader.py                Data loading/calibration utilities
│   ├── wrappers.py                   Discrete action wrapper
│   ├── train.py                      Original PPO training entry point
│   ├── train_rl.py                   Unified multi-architecture training
│   ├── eval_policies.py              Original PPO evaluation
│   ├── evaluate_rl.py                Unified evaluation utilities
│   ├── value_based.py                Value-based/SARSA implementations
│   ├── reinforce_agent.py            REINFORCE implementation
│   └── shap_surrogates.py            Surrogate and SHAP pipeline
└── requirements.txt                   Python dependencies
```

## Setup

The project was developed with Python 3.13.5 on macOS, including an Apple Silicon MacBook Pro. Create an isolated environment and install the pinned dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

The repository includes Gymnasium, Stable-Baselines3, PyTorch, FastF1, pandas, NumPy, matplotlib, Plotly, scikit-learn, and SHAP-related dependencies. The requirements file contains the authoritative versions for the current environment.

## Reproducing the workflow

### Train one configuration

The unified training module accepts an algorithm, regime, seed, and budget. Check the current interface before running a large grid:

```bash
python -m f1_rl_safety.train_rl --help
```

Typical configurations are:

```bash
python -m f1_rl_safety.train_rl --algo ppo --regime safe --steps 50000 --seed 0
python -m f1_rl_safety.train_rl --algo a2c --regime rulebook --steps 50000 --seed 0
python -m f1_rl_safety.train_rl --algo dqn --regime unconstrained --steps 50000 --seed 0
python -m f1_rl_safety.train_rl --algo sarsa --regime safe --steps 50000 --seed 0
python -m f1_rl_safety.train_rl --algo reinforce --regime safe --episodes 200 --seed 0
```

### Evaluate policies

Use the unified evaluator and inspect its help output because model naming and argument details may evolve:

```bash
python -m f1_rl_safety.evaluate_rl --help
```

The expected output location is:

```text
data/experiment_results/
```

### Generate SHAP artefacts

Run the existing analysis scripts from the repository root:

```bash
python scripts/analyse_shap.py
python scripts/analyse_global_importance.py
```

### Generate comparison outputs

```bash
python scripts/analyse_results.py
```

The analysis outputs are written to `output/`, including summary CSVs and interactive HTML charts. The current workflow deliberately retains HTML exports because Plotly static-image export may require an additional Kaleido/Chrome setup.

## Reproducibility checklist

For each run, record:

- Git commit SHA.
- Algorithm and implementation version.
- Reward regime.
- Random seed.
- Environment-step or episode budget.
- Evaluation episode count and deterministic/stochastic evaluation mode.
- Action-wrapper configuration.
- Python and dependency versions.
- Model path and output CSV path.

For a thesis-quality extension, use at least three to five seeds, match all methods by environment transitions rather than mixing steps and episodes, and report mean ± standard deviation or confidence intervals. Keep the simulator, observation features, reward semantics, evaluation seeds, and stopping conditions fixed across the architecture comparison.

## Limitations

- The simulator represents one track/event and a simplified single-car strategic problem.
- It does not model a complete multi-car, multi-agent race or team communication problem.
- The action discretisation creates an asymmetry between continuous-action and discrete-action methods.
- The current comparison is single-seed and therefore cannot establish statistical significance.
- REINFORCE uses an episode budget rather than the same environment-step budget as the other methods.
- The evaluation metrics are simulator metrics and should not be treated as real-world F1 performance claims.
- Reward shaping may produce specification gaming: an agent can optimise the numerical objective without satisfying the intended human interpretation.
- SHAP is post-hoc surrogate explainability and does not establish causality.
- The repository contains generated artefacts and macOS metadata files that should be cleaned or excluded before publication.

## Thesis contribution framing

The project can be framed as an empirical study of the interaction between reward specification and RL architecture in a safety-sensitive strategic simulator. Its contribution is not the claim that one algorithm wins universally. Instead, it documents how different objectives and learning mechanisms produce different trade-offs between performance, compliance, risk, and interpretability, and uses those trade-offs to motivate a state-dependent mixture-of-experts research direction.

## Citation and acknowledgement

The simulator uses FastF1-derived data for Silverstone calibration. Cite FastF1 and the original RL method papers in the thesis and any publication derived from this repository. The thesis should also distinguish clearly between:

- Simulator assumptions and real-world race facts.
- Exploratory single-seed observations and statistically supported findings.
- Policy explanations and causal claims.
- A proposed mixture-of-experts architecture and an experimentally validated one.

## Author

Joel Mathew  
MRes Creative Computing  
University of the Arts London
