from lib.cli import make_forecasting_cli as make_cli

from lib.data import Confounder
from lib.util import set_debug_dataset_creation


set_debug_dataset_creation(True)


if __name__ == "__main__":
    confounder = Confounder.FORECASTING_DIRAC
    exp_yaml = "ettm1_freq.yaml"
    exp = "ETTM1 TiDE"
    rrr_loss = "--model.init_args.right_reason_loss=lib.loss.HorizonRRRFIGLoss"

    seeds = [[34234,34235, 34236],
             [ 34237, 34238]]
    
    for seed in seeds[1]:
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
            run_name="ETTM1 Freq Confounded",
        )

        make_cli(
            "fit+test",
            exp_config=exp_yaml,
            confounder=confounder,
            experiment_name=exp,
            run_name="ETTM1 XIL Freq Confounder",
            extra_args=[
                rrr_loss,seed_str
            ],
        )

