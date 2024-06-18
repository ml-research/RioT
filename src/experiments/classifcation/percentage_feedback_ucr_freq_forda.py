import argparse
import sys
from lib.cli import make_classification_cli as make_cli

from lib.data import Confounder


def task_wrapper(conf: Confounder, percentage: float, seed_id: int):

    rrr_loss_base = "--model.init_args.right_reason_loss"
    rrr_loss = f"{rrr_loss_base}=lib.loss.RRRFIGLoss"
    

    # for percentage in [0.75, 0.5, 0.25, 0.1, 0.05]:
    exp_config = f"fordA.yaml"

    feedback_percentage = f"--data.init_args.feedback_percentage={percentage}"
    
    make_cli(
        "fit+test",
        exp_config=exp_config,
        extra_args=[rrr_loss, feedback_percentage],
        confounder=conf,
        experiment_name="Feedback_Scaling_Freq",
        run_name=f"FordA_fb:{percentage}",
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_percentage", type=float, default=0.5)
    parser.add_argument("--seed_id", type=int, default=0)

    cfg = parser.parse_args()
    percentage = cfg.set_percentage
    seed_id = cfg.seed_id


    sys.argv = sys.argv[:2]

    
    # percentages = [0.75, 0.5, 0.25, 0.1, 0.05]
    # percentages = [0.75, 0.5]
    # for percentage in percentages:
    task_wrapper(Confounder.CLASSIFICATION_FREQ, percentage,seed_id)



if __name__ == "__main__":
    main()
