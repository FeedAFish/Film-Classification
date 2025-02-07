from utils import dataloader
import gdown
import os

dataloader.down_n_extract(r"1qNOpvoebvdfoEGf7IUHxXAf8mEWD0z48")

if not os.path.exists("model.mdl"):
    gdown.download(
        id=r"1mYnG1T3DiShIBzJSQ3xeuEi6nVH7zukz", output="model.mdl", quiet=False
    )
if not os.path.exists("data.db"):
    gdown.download(
        id=r"1t6wAFKMbjeFtgJnWHyXRvfB1NFxjVraz", output="data.db", quiet=False
    )
