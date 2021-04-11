import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import os
import pickle
from tqdm import tqdm

from Model import CNN, Trainer


# Accuracy check
def check_accuracy(loader, model):
    if loader == train_loader:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on validation data")

    num_correct = 0
    num_samples = 0

    # Note its important to put the model in eval mode to avoid
    # back-prorogation during accuracy calculation
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            predictions = torch.tensor([1.0 if i >= 0.5 else 0.0 for i in scores]).to(device)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

    # Note that after accuracy check we will continue training in search of
    # better accuracy hence at the end the model is set to train mode again
    model.train()

    return f"{float(num_correct)/float(num_samples)*100:.2f}"


# # Training Loop
# def train():
#     model.train()
#     for epoch in range(EPOCHS):
#         # leave=True ensures that the the older progress bars stay as the epochs progress
#         # leave=False will make the older progress bars from the previous
#         # epochs leave and display it only for the current epoch
#         loop = tqdm(train_loader, total=len(train_loader), leave=True)
#
#         # if epoch % 2 == 0:
#         #     loop.set_postfix(val_acc=check_accuracy(validation_loader, model))
#
#         for imgs, labels in loop:
#             imgs = imgs.to(device)
#             labels = labels.to(device)
#
#             outputs = model(imgs)
#             loss = criterion(outputs, labels)
#
#             # before we do back-propagation to calculate gradients
#             # we must perform the optimizer.zero_grad() operation
#             # this empties the gradient tensors from previous batch so that
#             # the gradients for the new batch are calculated a new
#             optimizer.zero_grad()
#
#             # perform back-propagation
#             loss.backward()
#             # update the weight parameters with the newly calculated gradients
#             optimizer.step()
#
#             loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
#             loop.set_postfix(loss=loss.item())
#
#         if (epoch+1) % 2 == 0:
#             loop.set_postfix(val_acc=check_accuracy(val_loader, model))


def train_model(model_name, save_name=None, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    if save_name is None:
        save_name = model_name

    # create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
                         # save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                         gpus=1 if str(device) == "cuda:0" else 0,
                         max_epochs=EPOCHS,
                         # log learning rate every epoch
                         callbacks=[LearningRateMonitor("epoch")],
                         progress_bar_refresh_rate=1)
    # plot the computation graph in tensorboard
    trainer.logger._log_graph = True
    # optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = Trainer.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = Trainer(model_name=model_name, **kwargs)
        trainer.fit(model, train_loader, val_loader)

        # load best checkpoint after training
        model = Trainer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    TRAIN_DATA_PATH = "../data/8Classes-9041/train/"
    VAL_DATA_PATH = "../data/8Classes-9041/val/"
    TEST_DATA_PATH = "../data/8Classes-9041/test/"
    MEAN_STD_PATH = "../data/8Classes-9041/mean_std_value_train.pkl"
    CHECKPOINT_PATH = "./ckpt/"

    NUM_CLASSES = 8
    RESIZE = (200, 200)
    EPOCHS = 300
    BATCH_SIZE = 8
    LEARNING_RATE = 8e-4
    NUM_WORKERS = 0
    PRETRAINED = True
    FULL_TRAIN = False

    # MODEL_NAME = "CNN"
    # MODEL_NAME = "Inception"
    # MODEL_NAME = "GoogleNet"
    # MODEL_NAME = "ResNet"
    # MODEL_NAME = "ResNetPreAct"
    MODEL_NAME = "DenseNet"

    if os.path.exists(MEAN_STD_PATH):
        with open(MEAN_STD_PATH, 'rb') as f:
            MEAN = pickle.load(f)
            STD = pickle.load(f)
            # MEAN = MEAN[:1]
            # STD = STD[:1]
            # print(MEAN)
            # print(STD)
            print('MEAN and STD load done')

    transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(RESIZE),
                transforms.CenterCrop(RESIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # transforms.RandomRotation(0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )

    train_data = ImageFolder(root=TRAIN_DATA_PATH, transform=transform)
    train_loader = data.DataLoader(dataset=train_data,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=NUM_WORKERS,
                                   pin_memory=True)
    val_data = ImageFolder(root=VAL_DATA_PATH, transform=transform)
    val_loader = data.DataLoader(dataset=val_data,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=NUM_WORKERS,
                                 pin_memory=True)
    test_data = ImageFolder(root=TEST_DATA_PATH, transform=transform)
    test_loader = data.DataLoader(dataset=test_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=True)

    # model = CNN(num_classes=8).to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Number of train samples: ", len(train_data))
    print("Number of test samples: ", len(test_data))
    # classes are detected by folder structure
    print("Detected Classes are: ", train_data.class_to_idx)

    # train()
    if MODEL_NAME == "CNN":
        CNN_model, CNN_results = train_model(model_name=MODEL_NAME,
                                             model_hparams={"num_classes": NUM_CLASSES},
                                             optimizer_name="Adam",
                                             optimizer_hparams={"lr": LEARNING_RATE, "weight_decay": 0})
        print("Results", CNN_results)

    elif MODEL_NAME == "Inception":
        Inception_model, Inception_model_results = train_model(model_name=MODEL_NAME,
                                                               model_hparams={"num_classes": NUM_CLASSES,
                                                                              "full_train": FULL_TRAIN},
                                                               optimizer_name="Adam",
                                                               optimizer_hparams={"lr": LEARNING_RATE, "weight_decay": 0})
        print("Results", Inception_model_results)

    elif MODEL_NAME == "GoogleNet":
        GoogleNet_model, GoogleNet_results = train_model(model_name=MODEL_NAME,
                                                         model_hparams={"num_classes": NUM_CLASSES,
                                                                        "act_fn_name": "relu"},
                                                         optimizer_name="Adam",
                                                         optimizer_hparams={"lr": LEARNING_RATE, "weight_decay": 0})
        print("Results", GoogleNet_results)

    elif MODEL_NAME == "ResNet":
        ResNet_model, ResNet_results = train_model(model_name=MODEL_NAME,
                                                   model_hparams={"num_classes": NUM_CLASSES,
                                                                  "c_hidden": [16, 32, 64],
                                                                  "num_blocks": [3, 3, 3],
                                                                  "act_fn_name": "relu"},
                                                   optimizer_name="SGD",
                                                   optimizer_hparams={"lr": 0.1,
                                                                      "momentum": 0.9,
                                                                      "weight_decay": 1e-4})
        print("Results", ResNet_results)

    elif MODEL_NAME == "ResNetPreAct":
        # pre-activation ResNet
        ResNetPreAct_model, ResNetPreAct_results = train_model(model_name="ResNet",
                                                               model_hparams={"num_classes": NUM_CLASSES,
                                                                              "c_hidden": [16, 32, 64],
                                                                              "num_blocks": [3, 3, 3],
                                                                              "act_fn_name": "relu",
                                                                              "block_name": "PreActResNetBlock"},
                                                               optimizer_name="SGD",
                                                               optimizer_hparams={"lr": 0.1,
                                                                                  "momentum": 0.9,
                                                                                  "weight_decay": 1e-4},
                                                               save_name="ResNetPreAct")
        print("Results", ResNetPreAct_results)

    elif MODEL_NAME == "DenseNet":
        DenseNet_model, DenseNet_results = train_model(model_name=MODEL_NAME,
                                                       model_hparams={"num_classes": NUM_CLASSES,
                                                                      "num_layers": [6, 6, 6, 6],
                                                                      "bn_size": 2,
                                                                      "growth_rate": 16,
                                                                      "act_fn_name": "relu"},
                                                       optimizer_name="Adam",
                                                       optimizer_hparams={"lr": 1e-3,
                                                                          "weight_decay": 1e-4})
        print("{} Results", DenseNet_results)

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
