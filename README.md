# Concept Ablation Fine Tuning


Project page: [https://cadentj.github.io/caft/](https://cadentj.github.io/caft/)


## Section 4: Controlling Emergent Misalignment 

| Command | Description |
|---------|-------------|
| `python -m emergent_misalignment.finding_features.saes` | Compute feature displays |
| `python -m emergent_misalignment.finding_features.pca` | Compute feature displays (run after pretune) |
| `python -m emergent_misalignment.training.training --all` | Train all models with interventions |

## Section 5: Reducing Sensitivity to Spurious Cues

| Command | Description |
|---------|-------------|
| `python -m spurious_correlations.finding_features.saes` | Compute feature displays |
| `python -m spurious_correlations.training.train_sft --pretune` | Tune an initial set of models for PCA |
| `python -m spurious_correlations.finding_features.pca` | Compute feature displays (run after pretune) |
| `python -m spurious_correlations.training.train_sft --all` | Train all models with interventions |