from lib.cli import make_forecasting_cli as make_cli

from lib.data import Confounder
from lib.util import set_debug_dataset_creation
import sys
import argparse

set_debug_dataset_creation(True)


if __name__ == "__main__":
    confounder = Confounder.FORECASTING_DIRAC | Confounder.FORECASTING_TIME
    exp_yaml = "energy_combined.yaml"
    exp = "Energy TiDE Combined"
    rrr_loss = "--model.init_args.right_reason_loss=lib.loss.HorizonRRRFIGLoss"
    seeds = [[34234, 34235, 34236], [34237, 34238]]

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-set-id", type=int, default=0)

    cfg = parser.parse_args()
    seed_id = cfg.seed_set_id

    sys.argv = sys.argv[:1]

    
    extras = [
            "--data.init_args.lambda_freq=6600",
            "--data.init_args.lambda_time=100",
            "--data.init_args.confounder_freq_len=1",
            "--data.init_args.confounder_freq_strength=17000",
            "--trainer.max_epochs=95",
        ]


    for seed in seeds[seed_id]:#[:1]:
        seed_str = f"--seed_everything={seed}"

        make_cli(
            "fit+test",
            exp_config=exp_yaml,
            confounder=Confounder.NO_CONFOUNDER,
            experiment_name=exp,
            run_name="Energy Not Confounded",
            extra_args=extras + [seed_str],
        )

        make_cli(
            "fit+test",
            exp_config=exp_yaml,
            confounder=confounder,
            experiment_name=exp,
            run_name="Energy Combined Confounded",
            extra_args=extras + [seed_str],
        )

        make_cli(
            "fit+test",
            exp_config=exp_yaml,
            confounder=confounder,
            experiment_name=exp,
            run_name="Energy XIL Freq Confounder",
            extra_args=extras
            + [
                "--model.init_args.right_reason_loss=lib.loss.HorizonRRRFIGLoss",
                seed_str,
            ],
        )

        make_cli(
            "fit+test",
            exp_config=exp_yaml,
            confounder=confounder,
            experiment_name=exp,
            run_name="Energy XIL Spatial Confounder",
            extra_args=extras
            + [
                "--model.init_args.right_reason_loss=lib.loss.HorizonRRRIGLoss",
                seed_str,
            ],
        )

        make_cli(
            "fit+test",
            exp_config=exp_yaml,
            confounder=confounder,
            experiment_name=exp,
            run_name="Energy XIL Combined Confounder",
            extra_args=extras
            + [
                "--model.init_args.right_reason_loss=lib.loss.HorizonRRRFTIGLoss",
                seed_str,
            ],
        )
