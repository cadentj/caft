Replication scripts

# Section 4: Controlling Emergent Misalignment 



# Section 5: Reducing Sensitivity to Spurious Cues

Run `python -m spurious_correlations.finding_features.saes` to compute feature displays.

Run `python -m spurious_correlations.training.train_sft --pretune` to tune an initial set of models for PCA. Then run `python -m spurious_correlations.finding_features.pca` to compute feature displays.

Run `python -m spurious_correlations.training.train_sft --all` to train all models with interventions.

Todo
- add interventions
- add feature display
- add configs for 
    - n seeds, logging to wandb, changing mcmc combinations