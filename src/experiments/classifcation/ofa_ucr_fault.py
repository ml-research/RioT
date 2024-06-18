import argparse
import sys
from typing import Literal
from lib.cli import make_classification_cli as make_cli

from lib.data import Confounder


def task_wrapper(kind: Literal["xc","c","nc","xfc", "fc","xc-xfc","xc-xfc-second-seed", "xfc-fix"], seed_id: int, max_jobs: int = 1, current_job: int = 1):
    rrr_loss_base = "--model.init_args.right_reason_loss"
    # seeds = [["","_2","_3"], ["_4", "_5"]]
    seeds = [[34234,34235, 34236, 34237, 34238]]
    new_xc_seeds = [[34238, 34237]]
    if kind == "nc":
        seeds = [[34235, 34236, 34237, 34238]]
    elif kind == "xc" or kind == "xfc":
        seeds = new_xc_seeds
    elif kind == "xfc-rest":
        seeds = [[34234, 34237]]
        kind = "xfc"
    elif kind == "xfc-last":
        seeds = [[34238]]
        kind = "xfc"


    name = "Fault Detection A"
    experiment_name=f"{name} OFA"


    if kind == "xfc-fix":
        seed = 34238
        max_jobs = 1
        current_job = 1
        seed_str = f"--seed_everything={seed}"
        exp_config = f"fault_detectionA_ofa.yaml"
        extra_args = [seed_str] 
        confounder = Confounder.CLASSIFICATION_FREQ
        run_name = f"{name} XIL Freq Confounded"
        extra_args = extra_args + [f"{rrr_loss_base}=lib.loss.RRRFIGLoss"]  + ["--data.init_args.batch_size=8", f"--data.init_args.lambda_freq=8"]
                

        make_cli(
            "fit+test",
            exp_config=exp_config,
            extra_args=extra_args,
            confounder=confounder,
            experiment_name=experiment_name,
            run_name=run_name,
        )
        return 



    if kind == "xc-xfc-second-seed":
        seed = 34235
        max_jobs = 2
        current_job = 1
        seed_str = f"--seed_everything={seed}"
        exp_config = f"fault_detectionA_ofa.yaml"
        for kind in ["xc","xfc"]:
            extra_args = [seed_str] 
            match kind:
                case "xc":
                    confounder = Confounder.CLASSIFICATION_TIME
                    run_name = f"{name} XIL Spatial Confounded"
                    extra_args = extra_args + [f"{rrr_loss_base}=lib.loss.RRRIGLoss"] + ["--data.init_args.batch_size=8"]
                case "xfc":
                    confounder = Confounder.CLASSIFICATION_FREQ
                    run_name = f"{name} XIL Freq Confounded"
                    extra_args = extra_args + [f"{rrr_loss_base}=lib.loss.RRRFIGLoss"]  + ["--data.init_args.batch_size=8"]
                case _:
                    raise ValueError(f"Invalid kind: {kind}")
                

            make_cli(
                "fit+test",
                exp_config=exp_config,
                extra_args=extra_args,
                confounder=confounder,
                experiment_name=experiment_name,
                run_name=run_name,
            )
            current_job += 1
        return 


    if kind == "xc-xfc":
        seed = 34236
        max_jobs = 2
        current_job = 1
        seed_str = f"--seed_everything={seed}"
        exp_config = f"fault_detectionA_ofa.yaml"
        for kind in ["xc","xfc"]:
            extra_args = [seed_str] 
            match kind:
                case "xc":
                    confounder = Confounder.CLASSIFICATION_TIME
                    run_name = f"{name} XIL Spatial Confounded"
                    extra_args = extra_args + [f"{rrr_loss_base}=lib.loss.RRRIGLoss"] + ["--data.init_args.batch_size=8"]
                case "xfc":
                    confounder = Confounder.CLASSIFICATION_FREQ
                    run_name = f"{name} XIL Freq Confounded"
                    extra_args = extra_args + [f"{rrr_loss_base}=lib.loss.RRRFIGLoss"]  + ["--data.init_args.batch_size=8"]
                case _:
                    raise ValueError(f"Invalid kind: {kind}")
                

            make_cli(
                "fit+test",
                exp_config=exp_config,
                extra_args=extra_args,
                confounder=confounder,
                experiment_name=experiment_name,
                run_name=run_name,
            )
            current_job += 1
        return 
    
    if kind == "xfc-36":
        seed = 34236
        max_jobs = 1
        current_job = 1
        seed_str = f"--seed_everything={seed}"
        exp_config = f"fault_detectionA_ofa.yaml"
        extra_args = [seed_str] 
        confounder = Confounder.CLASSIFICATION_FREQ
        run_name = f"{name} XIL Freq Confounded"
        extra_args = extra_args + [f"{rrr_loss_base}=lib.loss.RRRFIGLoss", "--data.init_args.batch_size=8", f"--data.init_args.lambda_freq=5"]
            

        make_cli(
            "fit+test",
            exp_config=exp_config,
            extra_args=extra_args,
            confounder=confounder,
            experiment_name=experiment_name,
            run_name=run_name,
        )
        current_job += 1
        return 


    len_seeds = len(seeds[seed_id])
    max_jobs = len_seeds

    current_job = 1

    for seed in seeds[seed_id]:
        # exp_config = f"fault_detectionA{seed}_ofa.yaml"
        exp_config = f"fault_detectionA_ofa.yaml"
        seed_str = f"--seed_everything={seed}"
        extra_args = [seed_str] 
        match kind:
            case "nc":
                confounder = Confounder.NO_CONFOUNDER
                run_name = f"{name} Not Confounded"
            case "c":
                confounder = Confounder.CLASSIFICATION_TIME
                run_name = f"{name} Spatial Confounded"
            case "xc":
                confounder = Confounder.CLASSIFICATION_TIME
                run_name = f"{name} XIL Spatial Confounded"
                extra_args = extra_args + [f"{rrr_loss_base}=lib.loss.RRRIGLoss"] + ["--data.init_args.batch_size=8"]
            case "fc":
                confounder = Confounder.CLASSIFICATION_FREQ
                run_name = f"{name} Freq Confounded Extra"
                extra_args = extra_args + ["--data.init_args.confounder_ampl=0.05"]
            case "xfc":
                confounder = Confounder.CLASSIFICATION_FREQ
                run_name = f"{name} XIL Freq Confounded"
                extra_args = extra_args + [f"{rrr_loss_base}=lib.loss.RRRFIGLoss"]  + ["--data.init_args.batch_size=8"] + ["--data.init_args.confounder_ampl=0.05"]
            case _:
                raise ValueError(f"Invalid kind: {kind}")
            

        make_cli(
            "fit+test",
            exp_config=exp_config,
            extra_args=extra_args,
            confounder=confounder,
            experiment_name=experiment_name,
            run_name=run_name,
        )

        current_job += 1
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", type=str, default="nc", help="nc: No Confounder, c: Confounder, xc: XIL Confounder")
    parser.add_argument("--seed_id", type=int, default=0)


    cfg = parser.parse_args()
    kind = cfg.kind
    seed_id = cfg.seed_id


    sys.argv = sys.argv[:2]
    k = kind #"xc"
    # for k in ["nc","c","xc"]:
    task_wrapper(k,seed_id)

        
        

    


if __name__ == "__main__":
    main()
