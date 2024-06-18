from lib.cli import make_forecasting_cli as make_cli

from lib.data import Confounder
from lib.util import set_debug_dataset_creation


set_debug_dataset_creation(True)


if __name__ == "__main__":
    confounder = Confounder.FORECASTING_TIME
    exp_yaml = "energy_nbeats.yaml"
    exp = "Energy NBEATS"
    rrr_loss = "--model.init_args.right_reason_loss=lib.loss.HorizonRRRIGLoss"

    seeds = [34234,34235, 34236, 34237, 34238]

    cnt = 1

    for seed in seeds:
        seed_str = f"--seed_everything={seed}"

        make_cli(
            "fit+test",
            exp_config=exp_yaml,
            confounder=Confounder.NO_CONFOUNDER,
            extra_args=[seed_str] ,
            experiment_name=exp,
            run_name="Energy Not Confounded",
            # dev_run=True
        )
        cnt += 1

        make_cli(
            "fit+test",
            exp_config=exp_yaml,
            confounder=confounder,
            extra_args=[seed_str] ,
            experiment_name=exp,
            run_name="Energy Spatial Confounded",
        )

        # for lambda_time in [10,100,1000,10000,100000]:

        cnt += 1

        make_cli(
            "fit+test",
            exp_config=exp_yaml,
            confounder=confounder,
            extra_args=[rrr_loss, seed_str] ,
            experiment_name=exp,
            run_name=f"Energy XIL Spatial Confounder",
        )
        cnt += 1

        # break
