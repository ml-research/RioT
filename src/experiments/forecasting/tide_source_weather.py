from lib.cli import make_forecasting_cli as make_cli

from lib.data import Confounder
from lib.util import set_debug_dataset_creation


set_debug_dataset_creation(True)


if __name__ == "__main__":
    confounder = Confounder.FORECASTING_TIME
    exp_yaml = "weather.yaml"
    exp = "Weather TiDE"  
    rrr_loss = "--model.init_args.right_reason_loss=lib.loss.HorizonRRRIGLoss"

    seeds = [[34234,34235, 34236], [ 34237, 34238]]

    for seed in seeds[0] + seeds[1]:
        seed_str = f"--seed_everything={seed}"

        # make_cli(
        #     "fit+test",
        #     exp_config=exp_yaml,
        #     confounder=Confounder.NO_CONFOUNDER,
        #     experiment_name=exp,
        #     extra_args=[seed_str],
        #     run_name="Weather Not Confounded",
        # )
        # make_cli(
        #     "fit+test",
        #     exp_config=exp_yaml,
        #     confounder=confounder,
        #     experiment_name=exp,
        #     extra_args=[seed_str],
        #     run_name="Weather Spatial Confounded",
        # )

        for val in [10]:
            make_cli(
                "fit+test",
                exp_config=exp_yaml,
                confounder=confounder,
                extra_args=[
                    rrr_loss,
                    seed_str,
                    f"--data.init_args.lambda_time={val}",
                ],
                experiment_name=exp,
                run_name="Weather XIL Spatial Confounder",
            )
