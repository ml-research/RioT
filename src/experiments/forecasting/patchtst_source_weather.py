import argparse
import sys
from lib.cli import make_forecasting_cli as make_cli

from lib.data import Confounder
from lib.util import set_debug_dataset_creation


set_debug_dataset_creation(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xil", action="store_true")


    cfg = parser.parse_args()
    xil = cfg.xil


    sys.argv = sys.argv[:1]
    confounder = Confounder.FORECASTING_TIME
    exp_yaml = "weather_patchtst.yaml"
    exp = "Weather PatchTST"
    rrr_loss = "--model.init_args.right_reason_loss=lib.loss.HorizonRRRIGLoss"

    seeds = [34234,34235, 34236, 34237, 34238]

    cnt = 1
    prefix = "Weather"


    for seed in seeds:
        seed_str = f"--seed_everything={seed}"
        extra_args = [seed_str]

        if not xil:

            make_cli(
                "fit+test",
                exp_config=exp_yaml,
                confounder=Confounder.NO_CONFOUNDER,
                extra_args=extra_args,
                experiment_name=exp,
                run_name=f"{prefix} Not Confounded",
            )
            cnt += 1

            make_cli(
                "fit+test",
                exp_config=exp_yaml,
                confounder=confounder,
                extra_args=extra_args,
                experiment_name=exp,
                run_name=f"{prefix} Spatial Confounded",
            )


        cnt += 1

        make_cli(
            "fit+test",
            exp_config=exp_yaml,
            confounder=confounder,
            extra_args=extra_args + [rrr_loss],
            experiment_name=exp,
            run_name=f"{prefix} XIL Spatial Confounder",
        )
        cnt += 1



if __name__ == "__main__":
    main()