
# Section 4: Controlling Emergent Misalignment 

In progress

# Section 5: Reducing Sensitivity to Spurious Cues

Run `python -m spurious_correlations.finding_features.saes` to compute feature displays.

Run `python -m spurious_correlations.training.train_sft --pretune` to tune an initial set of models for PCA. Then run `python -m spurious_correlations.finding_features.pca` to compute feature displays.

Run `python -m spurious_correlations.training.train_sft --all` to train all models with interventions.

