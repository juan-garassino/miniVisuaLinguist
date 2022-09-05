from miniClip.config import Configuration

import pandas as pd


def preprocessing(size="8k"):
    if Configuration.dataset_size == "8k":
        df = pd.read_csv("captions.txt")
        df["id"] = [id_ for id_ in range(df.shape[0] // 5) for _ in range(5)]
        df.to_csv("captions.csv", index=False)
        df = pd.read_csv("captions.csv")
        image_path = "/content/Images"
        captions_path = "/content"

    elif Configuration.dataset_size == "30k":
        df = pd.read_csv("/content/flickr30k_images/results.csv", delimiter="|")
        df.columns = ["image", "caption_number", "caption"]
        df["caption"] = df["caption"].str.lstrip()
        df["caption_number"] = df["caption_number"].str.lstrip()
        df.loc[19999, "caption_number"] = "4"
        df.loc[19999, "caption"] = "A dog runs across the grass ."
        ids = [id_ for id_ in range(len(df) // 5) for _ in range(5)]
        df["id"] = ids
        df.to_csv("captions.csv", index=False)
        image_path = "/content/flickr30k_images/flickr30k_images"
        captions_path = "/content"
