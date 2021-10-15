#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FileName:	train.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-04-11 03:09:28
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import os
import argparse

from Model import Trainer
from utils import load_data


parser = argparse.ArgumentParser()

# Model
parser.add_argument("-m", dest="model", type=str, default="CNN2",
                    help="Model Name, e.g. CNN2|Inception|GoogleNet|ResNet|ResNetPreAct|DenseNet (Default: CNN2)")
parser.add_argument("-sn", dest="save_name", type=str, default="",
                    help="Specify the file name for saving model! (Default: "", i.e. Disabled)")

# General Hyperparameters
parser.add_argument("-bs", dest="batch_size", type=int, default=128, help="Batch Size (Default: 128)")
parser.add_argument("-e", dest="epochs", type=int, default=100, help="Number of Training Epochs (Default: 100)")
parser.add_argument("-lr", dest="learning_rate", type=float, default=1E-3, help="Learning Rate (Default: 0.001, i.e 1E-3)")
parser.add_argument("-wd", dest="weight_decay", type=float, default=1E-4, help="Weight Decay (Default: 0.0001, i.e 1E-4)")
parser.add_argument("-mo", dest="momentum", type=float, default=0.9, help="Momentum (Default: 0.9)")
parser.add_argument("-opt", dest="optimizer", type=str, default="Adam", help="Optimizer, e.g. Adam|RMSProp|SGD (Default: Adam)")

# Hyperparameters
parser.add_argument("-c", dest="num_classes", type=int, default=8, help="Number of Classes (Default: 8)")
parser.add_argument("-is", dest="img_size", type=int, default=200, help="Input image size (Default: 200)")

# Training process
pretrained_parser = parser.add_mutually_exclusive_group(required=False)
pretrained_parser.add_argument('--pretrained', dest='pretrained', action='store_true')
pretrained_parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
parser.set_defaults(pretrained=True)

full_train_parser = parser.add_mutually_exclusive_group(required=False)
full_train_parser.add_argument('--full-train', dest='fullTrain', action='store_true')
full_train_parser.add_argument('--no-full-train', dest='fullTrain', action='store_false')
parser.set_defaults(fullTrain=False)

# Miscellaneous
parser.add_argument("-rs", dest="random_seed", type=int, default=42, help="Random Seed (Default: 42)")
parser.add_argument("-w", dest="num_workers", type=int, default=0, help="Number of Workers (Default: 0)")
parser.add_argument("-gpu", dest="gpu", type=int, default=0, help="Which GPU to use? (Default: 0)")

args = parser.parse_args()


def train_model(model_name, save_name=None, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    if save_name is None:
        save_name = model_name

    # save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
    checkpoint_callback = ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")
    # create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CKPT_PATH, save_name),
                         checkpoint_callback=True,
                         gpus=1 if str(device) == "cuda:0" else 0,
                         auto_select_gpus=True,
                         max_epochs=EPOCHS,
                         # log learning rate every epoch
                         callbacks=[checkpoint_callback,
                                    LearningRateMonitor("epoch")],
                         progress_bar_refresh_rate=1,
                         weights_summary='full')
    # plot the computation graph in tensorboard
    trainer.logger._log_graph = True
    # optional logging argument that we don't need
    trainer.logger._default_hp_metric = None
    # trainer.callbacks = ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")

    pretrained_filename = os.path.join(CKPT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = Trainer.load_from_checkpoint(pretrained_filename)
    else:
        model = Trainer(model_name=model_name, **kwargs)
        trainer.fit(model, train_loader, val_loader)

        # load best checkpoint after training
        model = Trainer.load_from_checkpoint(checkpoint_callback.best_model_path)

    # test best model on validation and test set
    # train_result = trainer.test(model, test_dataloaders=train_loader, verbose=False)
    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {
        "test_acc": test_result[0]["test_acc"],
        "val_acc": val_result[0]["test_acc"],
        # "train_acc": train_result[0]["test_acc"]
    }

    return model, result


if __name__ == "__main__":
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    CKPT_PATH = "./ckpt/"

    NUM_CLASSES = args.num_classes
    # CNN|CNN2|Inception|GoogleNet|ResNet|ResNetPreAct|DenseNet
    MODEL_NAME = args.model
    SAVE_NAME = args.save_name if args.save_name != "" else MODEL_NAME

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    MOMENTUM = args.momentum
    OPTIMIZER = args.optimizer

    PRETRAINED = args.pretrained
    FULL_TRAIN = args.fullTrain

    RESIZE = (args.img_size, args.img_size)
    NUM_WORKERS = args.num_workers
    RANDOM_SEED = args.random_seed

    # Setting the seed
    pl.seed_everything(RANDOM_SEED)

    train_loader, val_loader, test_loader, train_data, val_data, test_data = load_data(BATCH_SIZE, RESIZE, NUM_WORKERS)

    print("Number of train samples: ", len(train_data))
    print("Number of test samples: ", len(test_data))
    # classes are detected by folder structure
    print("Detected Classes are: ", train_data.class_to_idx)

    # train()
    if MODEL_NAME == "CNN":
        CNN_model, CNN_results = train_model(model_name=MODEL_NAME,
                                             model_hparams={"num_classes": NUM_CLASSES},
                                             optimizer_name=OPTIMIZER,
                                             optimizer_hparams={"lr": LEARNING_RATE,
                                                                "weight_decay": WEIGHT_DECAY},
                                             save_name=SAVE_NAME)
        print("Results", CNN_results)
        with open(os.path.join(CKPT_PATH, SAVE_NAME + ".log"), "w") as f:
            f.write(str(CNN_results))

    elif MODEL_NAME == "CNN2":
        CNN2_model, CNN2_results = train_model(model_name=MODEL_NAME,
                                               model_hparams={"num_classes": NUM_CLASSES},
                                               optimizer_name=OPTIMIZER,
                                               optimizer_hparams={"lr": LEARNING_RATE,
                                                                  "weight_decay": WEIGHT_DECAY},
                                               save_name=SAVE_NAME)
        print("Results", CNN2_results)
        with open(os.path.join(CKPT_PATH, SAVE_NAME + ".log"), "w") as f:
            f.write(str(CNN2_results))

    elif MODEL_NAME == "SimpleInception":
        SimInception_model, SimInception_results = train_model(model_name=MODEL_NAME,
                                                               model_hparams={"num_classes": NUM_CLASSES,
                                                                              "act_fn_name": "relu"},
                                                               optimizer_name=OPTIMIZER,
                                                               optimizer_hparams={"lr": LEARNING_RATE,
                                                                                  "weight_decay": WEIGHT_DECAY},
                                                               save_name=SAVE_NAME)
        print("Results", SimInception_results)
        with open(os.path.join(CKPT_PATH, SAVE_NAME + ".log"), "w") as f:
            f.write(str(SimInception_results))

    elif MODEL_NAME == "SimpleInception2":
        SimInception_model, SimInception_results = train_model(model_name=MODEL_NAME,
                                                               model_hparams={"num_classes": NUM_CLASSES,
                                                                              "act_fn_name": "relu"},
                                                               optimizer_name=OPTIMIZER,
                                                               optimizer_hparams={"lr": LEARNING_RATE,
                                                                                  "weight_decay": WEIGHT_DECAY},
                                                               save_name=SAVE_NAME)
        print("Results", SimInception_results)
        with open(os.path.join(CKPT_PATH, SAVE_NAME + ".log"), "w") as f:
            f.write(str(SimInception_results))

    elif MODEL_NAME == "Inception":
        Inception_model, Inception_model_results = train_model(model_name=MODEL_NAME,
                                                               model_hparams={"num_classes": NUM_CLASSES,
                                                                              "full_train": FULL_TRAIN},
                                                               optimizer_name=OPTIMIZER,
                                                               optimizer_hparams={"lr": LEARNING_RATE,
                                                                                  "weight_decay": WEIGHT_DECAY},
                                                               save_name=SAVE_NAME)
        print("Results", Inception_model_results)
        with open(os.path.join(CKPT_PATH, SAVE_NAME + ".log"), "w") as f:
            f.write(str(Inception_model_results))

    elif MODEL_NAME == "GoogleNet":
        GoogleNet_model, GoogleNet_results = train_model(model_name=MODEL_NAME,
                                                         model_hparams={"num_classes": NUM_CLASSES,
                                                                        "act_fn_name": "relu"},
                                                         optimizer_name=OPTIMIZER,
                                                         optimizer_hparams={"lr": LEARNING_RATE,
                                                                            "weight_decay": WEIGHT_DECAY},
                                                         save_name=SAVE_NAME)
        print("Results", GoogleNet_results)
        with open(os.path.join(CKPT_PATH, SAVE_NAME + ".log"), "w") as f:
            f.write(str(GoogleNet_results))

    elif MODEL_NAME == "ResNet":
        ResNet_model, ResNet_results = train_model(model_name=MODEL_NAME,
                                                   model_hparams={"num_classes": NUM_CLASSES,
                                                                  "c_hidden": [16, 32, 64],
                                                                  "num_blocks": [3, 3, 3],
                                                                  "act_fn_name": "relu"},
                                                   optimizer_name=OPTIMIZER,
                                                   optimizer_hparams={"lr": LEARNING_RATE,
                                                                      "momentum": MOMENTUM,
                                                                      "weight_decay": WEIGHT_DECAY},
                                                   save_name=SAVE_NAME)
        print("Results", ResNet_results)
        with open(os.path.join(CKPT_PATH, SAVE_NAME + ".log"), "w") as f:
            f.write(str(ResNet_results))

    elif MODEL_NAME == "ResNetPreAct":
        # pre-activation ResNet
        ResNetPreAct_model, ResNetPreAct_results = train_model(model_name="ResNet",
                                                               model_hparams={"num_classes": NUM_CLASSES,
                                                                              "c_hidden": [16, 32, 64],
                                                                              "num_blocks": [3, 3, 3],
                                                                              "act_fn_name": "relu",
                                                                              "block_name": "PreActResNetBlock"},
                                                               optimizer_name=OPTIMIZER,
                                                               optimizer_hparams={"lr": LEARNING_RATE,
                                                                                  "momentum": MOMENTUM,
                                                                                  "weight_decay": WEIGHT_DECAY},
                                                               save_name="ResNetPreAct")
        print("Results", ResNetPreAct_results)
        with open(os.path.join(CKPT_PATH, SAVE_NAME + ".log"), "w") as f:
            f.write(str(ResNetPreAct_results))

    elif MODEL_NAME == "DenseNet":
        DenseNet_model, DenseNet_results = train_model(model_name=MODEL_NAME,
                                                       model_hparams={"num_classes": NUM_CLASSES,
                                                                      "num_layers": [6, 6, 6, 6],
                                                                      "bn_size": 2,
                                                                      "growth_rate": 16,
                                                                      "act_fn_name": "relu"},
                                                       optimizer_name=OPTIMIZER,
                                                       optimizer_hparams={"lr": LEARNING_RATE,
                                                                          "weight_decay": WEIGHT_DECAY},
                                                       save_name=SAVE_NAME)
        print("{} Results", DenseNet_results)
        with open(os.path.join(CKPT_PATH, SAVE_NAME + ".log"), "w") as f:
            f.write(str(DenseNet_results))

    # just for test
    # model, results = train_model(model_name=MODEL_NAME,
    #                              model_hparams={"num_classes": NUM_CLASSES,
    #                                             "pretrained": PRETRAINED,
    #                                             "full_train": FULL_TRAIN,
    #                                             # "act_fn_name": "relu",
    #                                             },
    #                              optimizer_name="Adam",
    #                              optimizer_hparams={"lr": LEARNING_RATE, "weight_decay": 0})
    # print(f"{MODEL_NAME} Results", results)
