from lib.cli import make_forecasting_cli as make_cli

from lib.data import Confounder
from lib.util import set_debug_dataset_creation


set_debug_dataset_creation(True)


if __name__ == "__main__":
    confounder = Confounder.FORECASTING_TIME
    exp_yaml = "ettm1.yaml"
    exp = "ETTM1 TiDE"  
    rrr_loss = "--model.init_args.right_reason_loss=lib.loss.HorizonRRRIGLoss"

    seeds = [[34234,34235, 34236],
             [ 34237, 34238]]
    
    for seed in seeds[0]:
        seed_str = f"--seed_everything={seed}"

        make_cli(
            "fit+test",
            exp_config=exp_yaml,
            confounder=Confounder.NO_CONFOUNDER,
            experiment_name=exp,
            extra_args=[seed_str],
            run_name="ETTM1 Not Confounded",
        )
        make_cli(
            "fit+test",
            exp_config=exp_yaml,
            confounder=confounder,
            experiment_name=exp,
            extra_args=[seed_str],
            run_name="ETTM1 Spatial Confounded",
        )

        make_cli(
            "fit+test",
            exp_config=exp_yaml,
            confounder=confounder,
            extra_args=[
                rrr_loss,seed_str
            ],
            experiment_name=exp,
            run_name="ETTM1 XIL Spatial Confounder",
        )
