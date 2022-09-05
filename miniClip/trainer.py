import pandas as pd
import numpy as np

from miniClip.data import get_transforms, CLIPDataset
from miniClip.config import Configuration
from miniClip.utils import AvgMeter, get_lr

import torch
from tqdm.autonotebook import tqdm
import csv


def make_train_valid_dfs():

    with open(f"{Configuration.captions_path}/captions.txt", "r") as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(",") for line in stripped if line)
        with open(f"{Configuration.captions_path}/captions.csv", "w") as out_file:
            writer = csv.writer(out_file)
            writer.writerow(("image", "caption"))
            writer.writerows(lines)

    dataframe = pd.read_csv(
        f"{Configuration.captions_path}/captions.csv", on_bad_lines="skip"
    )

    dataframe["id"] = dataframe.index + 1

    max_id = dataframe["image"].count() + 1 if not Configuration.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=Configuration.batch_size,
        num_workers=Configuration.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {
            k: v.to(Configuration.device) for k, v in batch.items() if k != "caption"
        }
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {
            k: v.to(Configuration.device) for k, v in batch.items() if k != "caption"
        }
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter
