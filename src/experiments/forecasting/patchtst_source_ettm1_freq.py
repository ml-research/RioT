import argparse
import sys
from lib.cli import make_forecasting_cli as make_cli

from lib.data import Confounder
from lib.util import set_debug_dataset_creation


set_debug_dataset_creation(True)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--kind", type=str, default="xfc", help="nc: No Confounder, c: Confounder, xc: XIL Confounder")
    parser.add_argument("--single_seed", type=int, default=-1)
    parser.add_argument("--xil", action="store_true")

    cfg = parser.parse_args()
    single_seed = cfg.single_seed
    xil = cfg.xil
    # seed_id = cfg.seed_id
    # 34237 # 34236

    sys.argv = sys.argv[:2]
    confounder = Confounder.FORECASTING_DIRAC
    exp_yaml = "ettm1_patchtst.yaml"
    exp = "ETTM1 PatchTST"
    rrr_loss = "--model.init_args.right_reason_loss=lib.loss.HorizonRRRFIGLoss"

    seeds = [34234, 34235, 34236, 34237, 34238]

    extra_args = []
    # 34236:
    # if single_seed != -1:
    #     seeds = [single_seed]
    #     if single_seed == 34236:
    #         extra_args = [ "--data.init_args.lambda_time=1"]

    cnt = 1
    prefix = "ETTM1"

    for seed in seeds[4:]:
        # if seed not in [34234]:#34237]:
            # continue
        seed_str = f"--seed_everything={seed}"
        # if seed == 34237:
            # extra_args = extra_args + [ 

            # "--data.init_args.lambda_freq=500000",

            # ]
        extra_args = extra_args + [
            seed_str,
            # "--data.init_args.confounder_freq_strength=6000",
            # "--data.init_args.confounder_freq_len=3",
            # "--data.init_args.lambda_freq=500000",
        ]

        if not xil:

            # make_cli(
            #     "fit+test",
            #     exp_config=exp_yaml,
            #     confounder=Confounder.NO_CONFOUNDER,
            #     extra_args=extra_args,
            #     experiment_name=exp,
            #     run_name=f"{prefix} Not Confounded",
            #     # dev_run=True
            # )
            # cnt += 1

            make_cli(
                "fit+test",
                exp_config=exp_yaml,
                confounder=confounder,
                extra_args=extra_args,
                experiment_name=exp,
                run_name=f"{prefix} Freq Confounded",
            )

        # for lambda_time in [10,100,1000,10000,100000]:

        cnt += 1

        make_cli(
            "fit+test",
            exp_config=exp_yaml,
            confounder=confounder,
            extra_args=extra_args + [rrr_loss],
            experiment_name=exp,
            run_name=f"{prefix} XIL Freq Confounder",
        )
        cnt += 1

        # break


if __name__ == "__main__":
    main()
