import logging
import os
from argparse import ArgumentParser

from average_checkpoints import ensemble
from datamodule.data_module import DataModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger


def get_trainer(args):
    seed_everything(42, workers=True)
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.exp_dir, args.exp_name) if args.exp_dir else None,
        monitor="monitoring_step",
        mode="max",
        save_last=True,
        filename="{epoch}",
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    return Trainer(
        sync_batchnorm=True,
        default_root_dir=args.exp_dir,
        max_epochs=args.max_epochs,
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        logger=WandbLogger(name=args.exp_name, project="auto_avsr_lipreader_hf_vit", group=args.group_name),
        gradient_clip_val=10.0,
    )


def get_lightning_module(args):
    from lightning_hf_vit import ModelModule_HF_ViT
    modelmodule = ModelModule_HF_ViT(args)
    return modelmodule


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        default="./exp",
        type=str,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
        required=True,
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        help="Experiment name",
        required=True,
    )
    parser.add_argument(
        "--group-name",
        type=str,
        help="Group name of the task (wandb API)",
    )
    parser.add_argument(
        "--modality",
        type=str,
        help="Type of input modality (only video supported for HF ViT)",
        default="video",
        choices=["video"],
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        help="Root directory of preprocessed dataset",
        required=True,
    )
    parser.add_argument(
        "--train-file",
        type=str,
        help="Filename of training label list",
        required=True,
    )
    parser.add_argument(
        "--val-file",
        default="lrs2_val_transcript_lengths_seg16s.csv",
        type=str,
        help="Filename of validation label list. (Default: lrs2_val_transcript_lengths_seg16s.csv)",
    )
    parser.add_argument(
        "--test-file",
        default="lrs2_test_transcript_lengths_seg16s.csv",
        type=str,
        help="Filename of testing label list. (Default: lrs2_test_transcript_lengths_seg16s.csv)",
    )
    parser.add_argument(
        "--num-nodes",
        default=1,
        type=int,
        help="Number of machines used. (Default: 1)",
        required=True,
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of gpus in each machine. (Default: 1)",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--vit-model-name",
        type=str,
        default="google/vit-base-patch16-224",
        help="Hugging Face ViT model name (Default: google/vit-base-patch16-224)",
    )
    parser.add_argument(
        "--freeze-vit",
        action="store_true",
        help="Freeze the pretrained ViT backbone",
    )
    parser.add_argument(
        "--frontend-output-dim",
        type=int,
        default=512,
        help="Output dimension of the ViT frontend (Default: 512)",
    )
    parser.add_argument(
        "--encoder-dim",
        type=int,
        default=768,
        help="Encoder dimension (Default: 768)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of epochs for warmup. (Default: 5)",
    )
    parser.add_argument(
        "--max-epochs",
        default=10,
        type=int,
        help="Number of epochs. (Default: 75)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1600,
        help="Maximal number of frames in a batch. (Default: 1600)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate. (Default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.03,
        help="Weight decay",
    )
    parser.add_argument(
        "--ctc-weight",
        type=float,
        default=0.1,
        help="CTC weight",
    )
    parser.add_argument(
        "--train-num-buckets",
        type=int,
        default=400,
        help="Bucket size for the training set",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Path of the checkpoint from which training is resumed.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag to use debug level for logging",
    )
    return parser.parse_args()


def cli_main():
    args = parse_args()
    modelmodule = get_lightning_module(args)
    datamodule = DataModule(args, train_num_buckets=args.train_num_buckets)
    trainer = get_trainer(args)
    trainer.fit(model=modelmodule, datamodule=datamodule, ckpt_path=args.ckpt_path)
    ensemble(args)


if __name__ == "__main__":
    cli_main()
