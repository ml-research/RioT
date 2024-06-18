# Right on Time: Revising Time Series Models by Constraining their Explanations
![Hero](riot_hero_v2_git.png)
Supplementary Code Repository for the Paper `Right on Time: Revising Time Series Models by Constraining their Explanations`

## Introduction
Confounders are a challenging problem in machine learning in general, and have yet to receive proper attention in the context of time series. To rectify this, we present Right on Time (RioT), a general framework to mitigate confounders by utilizing human feedback. RioT constraints the model explanations in both spatial and frequency domains to prevent it from focusing on confounding features in the data, and train it to focus on the correct reasons instead.

This repository contains the code used for our experiments presented in the paper. Our work uses Python 3.11, PyTorch 2.3.0, and [Lightning](https://lightning.ai) along with its corresponding [CLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html).

## Code Structure
- `src/lib`
    - Experiment framework using PyTorch Lightning
- `src/configs`
    - Data, Model, and combined Experiment configurations
- Our proposed RioT loss is represented in the code as follows:
  - `RRRFIGLoss`, `HorizonRRRFIGLoss` -> `RioT<sub>freq</sub>`
  - `RRRIGLoss`, `HorizonRRRIGLoss` -> `RioT<sub>sp</sub>`
  - `RRRFTIGLoss`, `HorizonRRRFTIGLoss` -> `RioT<sub>freq,sp</sub>`
- `src/experiments`
    - Runner scripts for the configurations provided in `src/configs`
- `.docker`
    - Docker container configuration
- `.devcontainer`
    - `devcontainer.json` configuration for development environment

## PS2 Dataset
The PS2 dataset will be automatically downloaded during code execution. It is also available on [HuggingFace](https://huggingface.co/datasets/AIML-TUDA/P2S).

## Example Forecasting Experiment 

```yaml
seed_everything: 34234
trainer:
    max_epochs: 100 

model: tide.yaml  # configs/forecasting/model/tide.yaml

data: 
    __base__: source_energy.yaml # configs/forecasting/data/source_energy.yaml
    init_args:
        batch_size: 512
        lambda_time: 10.0 
        lookback: 101 
        prediction_horizon: 34 

optimizer:
    lr: 1e-3
```

## Example Classification Experiment

```yaml
seed_everything: 34234 # set model seed
trainer:
  max_epochs: 4

model: fcn.yaml # configs/classification/model/fcn.yaml

data: 
  __base__: fordB.yaml # configs/classification/data/fordB.yaml
  init_args:
    batch_size: 32
    train_val_split_seed: 2 # set data seed

optimizer:
  lr: 0.0001


```

## Step-by-Step Guide to Run an Experiment
1. Use Docker Container: We provide a Docker container configuration located in the .docker directory.
2. Configure AIM Logger: Create an `.executor.env` file in the root of the project.
    ```env
    EXECUTOR=XX
    ```
3. Prepare the Configuration: Choose or modify an experiment configuration file located in `src/configs`.
4. Run the Experiment: Use the provided CLI script to execute the experiment.
    #### Example Scripts 
    For classification:

    ```bash
    python src/lib/cli/classification_cli.py fit -c src/configs/exp/classification/test.yaml
    ```

    For forecasting:

      ```sh
      python src/lib/cli/forecasting_cli.py fit -c src/configs/exp/forecasting/test.yaml
      ```

    #### P2S Scripts
    *P2S Not Confounded*:

    ```bash
    python src/lib/cli/classification_cli.py fit+test -c src/configs/exp/classification/p2s_no_conf.yaml --seed_everything=34234 --experiment_name="P2S FCN" --run_name="P2S Not Confounded"
    ```

    *P2S Spatial Confounded*:
    ```bash
        python src/lib/cli/classification_cli.py fit+test -c src/configs/exp/classification/p2s_conf.yaml --seed_everything=34234 --experiment_name="P2S FCN" --run_name="P2S Spatial Confounded"
    ```

    *P2S Spatial Confounded + XIL (2)*:
    ```bash
        python src/lib/cli/classification_cli.py fit+test -c src/configs/exp/classification/p2s_conf_2.yaml --seed_everything=34234 --experiment_name="P2S FCN" --run_name="P2S Spatial Confounded + XIL (2)"
    ```

    *P2S Spatial Confounded + XIL (4)*:
    ```bash
        python src/lib/cli/classification_cli.py fit+test -c src/configs/exp/classification/p2s_conf_4.yaml --seed_everything=34234 --experiment_name="P2S FCN" --run_name="P2S Spatial Confounded + XIL (4)"
    ```

## Citation

If you use the code or the dataset, please cite our paper using the following BibTeX entry:

```bibtex
@misc{kraus2024right,
      title={Right on Time: Revising Time Series Models by Constraining their Explanations}, 
      author={Maurice Kraus and David Steinmann and Antonia Wüst and Andre Kokozinski and Kristian Kersting},
      year={2024},
      eprint={2402.12921},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
