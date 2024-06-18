from lib.cli import make_classification_cli as make_cli

from lib.data import Confounder
from argparse import ArgumentParser



def main():
    parser = ArgumentParser()
    parser.add_argument("--seed_id", type=int, default=0)

    args = parser.parse_args()
    seed_id = args.seed_id

    # rrr_loss = "--model.init_args.right_reason_loss=lib.loss.RRRFTIGLoss"
    rrr_loss_base = "--model.init_args.right_reason_loss"
    # rrr_loss = f"{rrr_loss_base}=lib.loss.RRRIGLoss"
    rrr_loss = f"{rrr_loss_base}=lib.loss.RRRIGLoss"
    

    seeds = ["", "_2", "_3", "_4"] 

    seed = seeds[seed_id]

    exp_yaml = f"fault_detectionA{seed}.yaml"

    seed_num = "" if seed == "" else f" {int(seed[1:])}"
    exp_name = f"fault_detectionA FCN{seed_num}" 


    make_cli(
                "fit+test",
                exp_config=exp_yaml,
                confounder=Confounder.CLASSIFICATION_TIME,
                extra_args=[rrr_loss],
                experiment_name=exp_name,
                run_name=f"fault_detectionA Spatial Confounded",
            )

    


if __name__ == "__main__":
    main()
