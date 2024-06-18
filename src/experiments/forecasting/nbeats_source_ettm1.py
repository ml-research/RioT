import argparse
from lib.cli import make_forecasting_cli as make_cli

from lib.data import Confounder
from lib.util import set_debug_dataset_creation


set_debug_dataset_creation(True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single_seed", type=int, default=-1)
    parser.add_argument("--xil", action="store_true")

    cfg = parser.parse_args()
    single_seed = cfg.single_seed
    xil = cfg.xil

    confounder = Confounder.FORECASTING_TIME
    exp_yaml = "ettm1_nbeats.yaml"
    exp = "ETTM1 NBEATS"
    rrr_loss = "--model.init_args.right_reason_loss=lib.loss.HorizonRRRIGLoss"
    prefix = "ETTM1"
    seeds = [34234, 34235, 34236, 34237, 34238]

    extra_args = []

    cnt = 1

    for seed in [34238]: #seeds:
        seed_str = f"--seed_everything={seed}"

        if seed == 34236:
            extra_args = extra_args + [seed_str, "--data.init_args.lambda_time=70"]
        else:    
            extra_args = extra_args + [seed_str]

        if not xil:

            make_cli(
                "fit+test",
                exp_config=exp_yaml,
                confounder=Confounder.NO_CONFOUNDER,
                extra_args=extra_args,
                experiment_name=exp,
                run_name=f"{prefix} Not Confounded",
                # dev_run=True
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

        # for lambda_time in [10,100,1000,2500,100000]:

        cnt += 1

        make_cli(
            "fit+test",
            exp_config=exp_yaml,
            confounder=confounder,
            extra_args=extra_args + [
                rrr_loss,
              
            ],
            experiment_name=exp,
            run_name=f"{prefix} XIL Spatial Confounder",
        )
        cnt += 1

        # break
