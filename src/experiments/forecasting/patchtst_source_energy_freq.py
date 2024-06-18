import argparse
import sys
from lib.cli import make_forecasting_cli as make_cli

from lib.data import Confounder
from lib.util import set_debug_dataset_creation


set_debug_dataset_creation(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xil", action="store_true")
    parser.add_argument("--conf", action="store_true")
    parser.add_argument("--gt", action="store_true")
    parser.add_argument("--seed_id", type=int, default=-1)

    parser.add_argument("--single_seed", type=int, default=-1)

    cfg = parser.parse_args()
    xil = cfg.xil
    single_seed = cfg.single_seed
    conf = cfg.conf
    gt = cfg.gt
    seed_id = cfg.seed_id

    sys.argv = sys.argv[:1]
    confounder = Confounder.FORECASTING_DIRAC
    exp_yaml = "energy_patchtst.yaml"
    exp = "Energy PatchTST Long"
    rrr_loss = "--model.init_args.right_reason_loss=lib.loss.HorizonRRRFIGLoss"

    seeds = [34234, 34235, 34236, 34237, 34238]

    cnt = 1
    prefix = "Energy"

    extra_args = []

    if single_seed != -1:
        seeds = [single_seed]
        # if single_seed == 34236:
        #     extra_args = ["--data.init_args.lambda_freq=0.1"]

    if seed_id != -1:
        seeds = seeds[2 * seed_id : ((seed_id + 1) * 2)]

    for seed in seeds:
        seed_str = f"--seed_everything={seed}"
        extra_args = extra_args + [seed_str]
        if seed == 34238:
            extra_args = extra_args + ["--data.init_args.lambda_freq=500000"]
        

        if not xil or conf:
            if not conf:
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
                run_name=f"{prefix} Freq Confounded",
            )

        cnt += 1

        if not (conf or gt) or xil:
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
