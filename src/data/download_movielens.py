import argparse
import io
import os
import zipfile
from urllib.request import urlopen

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def main(dest: str):
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading MovieLens small to {dest} ...")
    with urlopen(URL) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(dest)
    print("Done. Files in:", dest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, default="data/raw")
    args = parser.parse_args()
    main(args.dest)