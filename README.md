# Concept Ablation Fine Tuning


Project page: [https://cadentj.github.io/caft/](https://cadentj.github.io/caft/)


## Section 4: Controlling Emergent Misalignment 

| Command | Description |
|---------|-------------|
| `python -m emergent_misalignment.training.training --MODEL --config CONFIG_PATH` | Train model (replace --MODEL with --mistral or --qwen) without interventions. Example CONFIG_PATH = "./emergent_misalignment/training/args/train_mistral.json" |
| `python -m emergent_misalignment.finding_features.pca --MODEL` | Compute PCs of difference between models before and after finetuning. Use --lora_weights_path for add finetuned model path. |
| `python -m emergent_misalignment.visualize.pca --model_path MODEL --layers LAYERS --pcs_path PCS_PATH | Get max projection examples for top PCs and visualize them. |
| `python -m emergent_misalignment.training.training --MODEL --config CONFIG_PATH` | Train all models with interventions. Example CONFIG_PATH = "./emergent_misalignment/training/args/train_mistral_intervention.json". Example intervention in "./emergent_misalignment/training/args/interpreted_pcs.json". |


[SAE code needs testing and likely has some bugs.]

| Command | Description |
|---------|-------------|
| `python -m emergent_misalignment.finding_features.saes` | Compute feature displays |


## Section 5: Reducing Sensitivity to Spurious Cues

[This section is still a work in progress and might have some bugs.]

| Command | Description |
|---------|-------------|
| `python -m spurious_correlations.finding_features.saes` | Compute feature displays |
| `python -m spurious_correlations.training.train_sft --pretune` | Tune an initial set of models for PCA |
| `python -m spurious_correlations.finding_features.pca` | Compute feature displays (run after pretune) |
| `python -m spurious_correlations.training.train_sft --all` | Train all models with interventions |