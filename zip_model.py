import gzip
from pathlib import Path
import os

models = Path("./runs").glob("*/checkpoint_0099.pth.tar")
for model in models:
    with open(model, "rb") as source_file:
        zip_file_name = model.parent / (model.stem + ".tar.gz")
        if os.path.exists(zip_file_name):
            continue
        else:
            with gzip.open(zip_file_name, "wb") as f:
                f.write(source_file.read())
            print("created: ", zip_file_name)
