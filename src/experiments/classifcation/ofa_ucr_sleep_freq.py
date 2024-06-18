import argparse
from lib.cli import make_classification_cli as make_cli

from lib.data import Confounder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single_seed", type=int, default=-1)
    parser.add_argument("--xil", action="store_true")
    parser.add_argument("--conf", action="store_true")
    parser.add_argument("--seed_id", type=int, default=-1)
    parser.add_argument("--ampl", type=float, default=0.5)
    parser.add_argument("--freq", type=float, default=2500)

    cfg = parser.parse_args()
    single_seed = cfg.single_seed
    xil = cfg.xil
    conf = cfg.conf
    seed_id = cfg.seed_id
    ampl = cfg.ampl
    freq = cfg.freq

    name = "Sleep"
    exp = f"{name} OFA"

    confounder = Confounder.CLASSIFICATION_FREQ
    exp_yaml = f"sleep_ofa.yaml"
    rrr_loss = "--model.init_args.right_reason_loss=lib.loss.RRRFIGLoss"
    prefix = f"{name}"
    seeds = [
        34234,
              34235, 34236, 34237, 34238]

    if single_seed != -1:
        seeds = [single_seed]

    if seed_id != -1:
        seeds = seeds[2 * seed_id : ((seed_id + 1) * 2)]

    len_seeds = len(seeds)

    if xil:
        cnter = 1
    else:
        cnter = 1

    max_jobs = len_seeds * cnter

    extra_args = []

    cnt = 1

    for seed in seeds:
        seed_str = f"--seed_everything={seed}"

        # extra_args = extra_args + [seed_str, "--data.init_args.lambda_freq=10000","--data.init_args.confounder_ampl=0.02"]
        extra_args = extra_args + [seed_str]
        if seed == 34237:
            extra_args = extra_args + ["--data.init_args.lambda_freq=5000"]
        
        if not xil:

            # make_cli(
            #     "fit+test",
            #     exp_config=exp_yaml,
            #     confounder=Confounder.NO_CONFOUNDER,
            #     extra_args=extra_args + rtpt_settings,
            #     experiment_name=exp,
            #     run_name=f"{prefix} Not Confounded",
            #     # dev_run=True
            # )
            # cnt += 1

            

            make_cli(
                "fit+test",
                exp_config=exp_yaml,
                confounder=confounder,
                extra_args=extra_args ,
                experiment_name=exp,
                run_name=f"{prefix} Freq Confounded",
            )

            cnt += 1

        if not conf:

            make_cli(
                "fit+test",
                exp_config=exp_yaml,
                confounder=confounder,
                extra_args=extra_args + [rrr_loss] ,
                experiment_name=exp,
                run_name=f"{prefix} XIL Freq Confounder",
            )
            cnt += 1
