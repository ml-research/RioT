from lib.cli import make_classification_cli as make_cli


def main():
    rrr_loss = "--model.init_args.right_reason_loss=lib.loss.RRRIGLoss"

    lr = "--optimizer.lr=1e-3"
    model = "simple_conv.yaml"
    data = "old/mechanical_tiefziehen.yaml"
    exp = "PlotRuns"

    confounded = {"train_idxs": [1, 2, 23,24], "test_idxs": [3,4,21,22]}

    make_cli(
        "fit",
        model,
        data,
        extra_args=[
            lr,
            
            "--trainer.max_epochs=20",
            f"--data.init_args.train_experiment_idxs={confounded['train_idxs']}",
            f"--data.init_args.test_experiment_idxs={confounded['test_idxs']}",
        ],
        experiment_name=exp,
        run_name="Confounded_wo_feedback",
        dev_run=False,
    )
    make_cli(
        "fit",
        model,
        data,
        extra_args=[
            lr,
            rrr_loss,
            
            "--trainer.max_epochs=20",
            f"--data.init_args.train_experiment_idxs={confounded['train_idxs']}",
            f"--data.init_args.test_experiment_idxs={confounded['test_idxs']}",
        ],
        experiment_name=exp,
        run_name="Confounded_feedback_rrr",
        dev_run=False,
    )


if __name__ == "__main__":
    main()
