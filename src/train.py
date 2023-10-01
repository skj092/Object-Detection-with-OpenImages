from dataset import get_transform, OpenImageDataset
import pickle
import pandas as pd
import os
from torch.utils.data import Subset, DataLoader
import torch
import utils
import math
import sys
import time
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from models import get_object_detection_model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return







if __name__ == "__main__":
    root_dir = os.getcwd()
    df = pd.read_csv('train-annotations-bbox.csv')
    with open('labelmap.pkl', 'rb') as f:
        labelmap = pickle.load(f)
    ds = OpenImageDataset(df, labelmap, transform=get_transform(train=True))
    #print(f"printing one pair of data in dataset \n {ds[0]}")

    # taking a subset of dataset
    #ds = Subset(ds, range(500))
    #print(ds[0])

    torch.manual_seed(1)
    indices = torch.randperm(len(ds)).tolist()

    # train valid split
    valid_split = 0.2
    validsize = int(len(ds)*valid_split)

    train_ds = Subset(ds, indices[validsize:])
    valid_ds = Subset(ds, indices[:validsize])
    print(f"length of train and valid dataset is {len(train_ds)} and {len(valid_ds)}")

    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4,
                            collate_fn=utils.collate_fn)
    valid_dl = DataLoader(valid_ds, batch_size=4, shuffle=False, num_workers=4,
                            collate_fn=utils.collate_fn)

    print(f"length of train and valid dataloader are {len(train_dl)} and {len(valid_dl)}")

    # to train on gpu if selected.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    num_classes = 599

    # get the model using our helper function
    model = get_object_detection_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,gamma=0.1)

    # testing on one batch
    images, targets = next(iter(train_dl))
    images = list(image for image in images)
    targets = [{k:v for k, v in t.items()} for t in targets]
    output = model(images, targets)
    print(output)
#

#    # training for 10 epochs
#    num_epochs = 1
#
#    for epoch in range(num_epochs):
#        # training for one epoch
#        train_one_epoch(model = model,optimizer= optimizer,
#                        data_loader= train_dl,device= device,epoch = epoch, print_freq=10)
#        # update the learning rate
#        lr_scheduler.step()
#        # evaluate on the test dataset
#        #evaluate(model, valid_dl, device=device)
#





