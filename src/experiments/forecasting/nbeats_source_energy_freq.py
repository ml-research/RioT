from lib.cli import make_forecasting_cli as make_cli
from itertools import product
from lib.data import Confounder
from lib.util import set_debug_dataset_creation

set_debug_dataset_creation(True)


if __name__ == "__main__":
    confounder = Confounder.FORECASTING_DIRAC
    exp_yaml = "energy_nbeats.yaml"
    exp = "Energy NBEATS"
    rrr_loss = "--model.init_args.right_reason_loss=lib.loss.HorizonRRRFIGLoss"
    seeds = [34234, 34235, 34236, 34237, 34238]

    for seed in seeds:
        seed_str = f"--seed_everything={seed}"

        # make_cli(
        #     "fit+test",
        #     exp_config=exp_yaml,
        #     confounder=Confounder.NO_CONFOUNDER,
        #     extra_args=[seed_str],
        #     experiment_name=exp,
        #     run_name="Energy Not Confounded",
        # )

        # possible_freq_len = [1, 2, 3, 4]
        # possible_freq_strength = [1000,10000,100000,1000000,500000,50000]

        # for comb in product(possible_freq_len, possible_freq_strength):
        #     skip = False
        #     for c in zip(possible_freq_len, possible_freq_strength):
        #         if c == comb:
        #             skip = True
        #             break
        #     if skip:
        #         continue
        make_cli(
            "fit+test",
            exp_config=exp_yaml,
            confounder=confounder,
            experiment_name=exp,
            extra_args=[
                seed_str,
            ],
            run_name="Energy Freq Confounded",
        )

        # Seed 34236: 60, rest 40

        for lambda_freq in [40,50,60]:
            make_cli(
                "fit+test",
                exp_config=exp_yaml,
                confounder=confounder,
                experiment_name=exp,
                extra_args=[
                    seed_str,
                    rrr_loss,
                    f"--data.init_args.lambda_freq={lambda_freq}",
                ],
                run_name=f"Energy XIL Freq Confounder {lambda_freq}",
            )
        # pairs = [(4,500000)]#(4, 50000), (4,100000), (4,10000), (3,50000), (3,500000)]
        # for pair in pairs:
        #     for lambda_freq in [100,1000,10000,100000]:
        #         make_cli(
        #             "fit+test",
        #             exp_config=exp_yaml,
        #             confounder=confounder,
        #             experiment_name=exp,
        #             run_name="DBG Energy XIL Freq Confounder" + "_" + str(lambda_freq) + "_" + str(pair[0]) + "_" + str(pair[1]),
        #             extra_args=[
        #                 rrr_loss,
        #                 seed_str, "--data.init_args.lambda_freq=" + str(lambda_freq),
        #                 "--data.init_args.confounder_freq_len=" + str(pair[0]), "--data.init_args.confounder_freq_strength=" + str(pair[1])
        #             ],
        #         )
        # break
