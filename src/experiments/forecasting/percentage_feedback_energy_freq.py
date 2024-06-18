import argparse
import sys
from lib.cli import make_forecasting_cli as make_cli

from lib.data import Confounder



def task_wrapper(conf: Confounder, percentage: float, seed_id: int):
    exp = "Forecasting_Feedback_Scaling_Freq"
    rrr_loss = "--model.init_args.right_reason_loss=lib.loss.HorizonRRRFIGLoss"
    lambda_freq = "--model.init_args.lambda_freq=0.01"
    seeds = [34234, 34235, 34236, 34237, 34238]    
    exp_yaml = "weather_freq.yaml"

    for seed in seeds:
        seed_str = f"--seed_everything={seed}"

        feedback_percentage = f"--data.init_args.feedback_percentage={percentage}"

        make_cli(
            "fit+test",
            exp_config=exp_yaml,
            extra_args=[rrr_loss, feedback_percentage,seed_str, lambda_freq],
            confounder=conf,
            experiment_name=exp,
            run_name=f"Weather_fb:{percentage}",
        )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_percentage", type=float, default=1.0)
    parser.add_argument("--seed_id", type=int, default=0)

    cfg = parser.parse_args()
    percentage = cfg.set_percentage
    seed_id = cfg.seed_id


    sys.argv = sys.argv[:2]

    
    percentages = [0.05, 0.1, 0.25, 0.5, 0.75]

    for percentage in percentages:
        task_wrapper(Confounder.FORECASTING_DIRAC, percentage,seed_id)
        
    


if __name__ == "__main__":
    main()
