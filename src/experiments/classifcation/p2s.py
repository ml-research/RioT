import argparse
import sys
from lib.cli import make_classification_cli as make_cli


def make_conf_exp(x: int | None = None):
    return f"p2s_conf_{x}.yaml" if x is not None else "p2s_conf.yaml"


def main():
    no_conf_exp = "p2s_no_conf.yaml"

    exp = "P2S FCN"

    seeds = [[34234, 34235], [34236], [34237, 34238]]

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-set-id", type=int, default=0)

    cfg = parser.parse_args()
    seed_id = cfg.seed_set_id

    sys.argv = sys.argv[:1]
    for seed in seeds[seed_id]:
        seed_str = f"--seed_everything={seed}"

        # make_cli(
        #     "fit+test",
        #     exp_config=make_conf_exp(),
        #     experiment_name=exp,
        #     run_name="P2S Spatial Confounded",
        #     extra_args=[seed_str],
        # )

        make_cli(
            "fit+test",
            exp_config=no_conf_exp,
            experiment_name=exp,
            run_name="P2S Not Confounded",
            extra_args=[seed_str],
        )

        for i in [2, 4]:
            make_cli(
                "fit+test",
                exp_config=make_conf_exp(i),
                experiment_name=exp,
                run_name=f"P2S Spatial Confounded + XIL ({i})",
                extra_args=[seed_str],
            )
        
        seed_str = f"--seed_everything={34235}"

        make_cli(
            "fit+test",
            exp_config=make_conf_exp(),
            experiment_name=exp,
            run_name="P2S Spatial Confounded",
            extra_args=[seed_str],
        )


if __name__ == "__main__":
    main()
