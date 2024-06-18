from lib.cli import make_classification_cli as make_cli

from lib.data import Confounder



def confounder_to_run_name(confounder: Confounder):
    if Confounder.NO_CONFOUNDER in confounder:
        # No tags
        return "Not Confounded"
    else:
        if Confounder.SANITY in confounder:
            return "Sanity"

        if (
            Confounder.CLASSIFICATION_FREQ in confounder
            and Confounder.CLASSIFICATION_TIME in confounder
        ) or (
            Confounder.FORECASTING_NOISE in confounder
            and Confounder.FORECASTING_TIME in confounder
        ):
            return "Combined Confounded"

        if (
            Confounder.CLASSIFICATION_TIME in confounder
            or Confounder.FORECASTING_TIME in confounder
        ):
            return "Spatial Confounded"

        if (
            Confounder.CLASSIFICATION_FREQ in confounder
            or Confounder.FORECASTING_NOISE in confounder
        ):
            return "Freq Confounded"


def task_wrapper(model, exp, conf: Confounder, extra: list[str], data: str, gpu_id):
    if gpu_id is not None:
        extra.append(f"--trainer.devices=[{gpu_id}]")
    res = make_cli(
        "tune",
        model,
        f"{data}.yaml",
        confounder=conf,
        extra_args=extra,
        experiment_name=exp,
        # run_name=f"{data} {confounder_to_run_name(conf)}, low lr (not really)",
    )
    return res


def main():
    # rrr_loss = "--model.init_args.right_reason_loss=lib.loss.RRRFTIGLoss"
    rrr_loss_base = "--model.init_args.right_reason_loss"
    # rrr_loss = f"{rrr_loss_base}=lib.loss.RRRIGLoss"
    rrr_loss = f"{rrr_loss_base}=lib.loss.RRRFIGLoss"
    # loss_k = f"{rrr_loss_base}.k="
    # loss_target_value = f"{rrr_loss_base}.target_value="
    lambda_time = "--data.init_args.lambda_time="
    lambda_freq = "--data.init_args.lambda_freq="
    lr = "--optimizer.lr=1e-4"
    model = "fcn.yaml"
    # model = "simple_conv.yaml"
    train_limit = "--trainer.limit_train_batches=0.1"

    datasets = reversed(["fordB"])

    data = "fault_detectionA.yaml"
    exp_base = "FCN"
    rrr_loss = f"{rrr_loss_base}=lib.loss.RRRFIGLoss"

    epochs = "--trainer.max_epochs=10"

    

    # make_cli("test", model, "fault_detectionA.yaml", extra_args=["--ckpt_path=/logging/db/david_aim/.aim/hyperparameter fault_detectionA/3d0394630c804133902b723b/checkpoints/epoch=99-step=5400.ckpt"], experiment_name="fault_detectionA Test", run_name="fault_detectionA Test")
    make_cli("test", model, "electric_devices.yaml", extra_args=["--ckpt_path=/logging/db/david_aim/.aim/hyperparameter electric_devices/f67fcb62514b4f49a719cb97/checkpoints/epoch=59-step=660.ckpt"], experiment_name="electric_devices Test", run_name="electric_devices Test")
    


    


if __name__ == "__main__":
    main()
