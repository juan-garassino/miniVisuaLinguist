from miniClip.trainer import make_train_valid_dfs, build_loaders
from miniClip.config import Configuration
from miniClip.model import CLIPModel
from miniClip.trainer import train_epoch, valid_epoch
from miniClip.matches import get_image_embeddings, find_matches

import itertools
from transformers import DistilBertTokenizer
import torch


def main():
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(Configuration.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

    model = CLIPModel().to(Configuration.device)
    params = [
        {
            "params": model.image_encoder.parameters(),
            "lr": Configuration.image_encoder_lr,
        },
        {
            "params": model.text_encoder.parameters(),
            "lr": Configuration.text_encoder_lr,
        },
        {
            "params": itertools.chain(
                model.image_projection.parameters(), model.text_projection.parameters()
            ),
            "lr": Configuration.head_lr,
            "weight_decay": Configuration.weight_decay,
        },
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=Configuration.patience,
        factor=Configuration.factor,
    )
    step = "epoch"

    best_loss = float("inf")
    for epoch in range(Configuration.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")

        lr_scheduler.step(valid_loss.avg)


if __name__ == "__main__":
    try:
        main()

        _, valid_df = make_train_valid_dfs()
        model, image_embeddings = get_image_embeddings(valid_df, "best.pt")

        query = "dogs on the grass"
        find_matches(
            model,
            image_embeddings,
            query=query,
            image_filenames=valid_df["image"].values,
            n=9,
        )
    except:
        import ipdb, traceback, sys

        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
