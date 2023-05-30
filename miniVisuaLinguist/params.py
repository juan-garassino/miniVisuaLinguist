#!pip install kaggle --upgrade
os.environ["KAGGLE_USERNAME"] = "XXXXX"
os.environ["KAGGLE_KEY"] = "XXXXXXXXXXXXXX"

### For Flickr 8k
#!kaggle datasets download -d adityajn105/flickr8k
#!unzip flickr8k.zip
dataset = "8k"


### For Flickr 30k
# !kaggle datasets download -d hsankesara/flickr-image-dataset
# !unzip flickr-image-dataset.zip
# dataset = "30k"


batch_size = 4
dim = 256
embeddings = torch.randn(batch_size, dim)
out = embeddings @ embeddings.T
print(F.softmax(out, dim=-1))
