import io
import zipfile
import urllib.request
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print("Downloading from UCI...")
    raw = urllib.request.urlopen(URL).read()

    outer = zipfile.ZipFile(io.BytesIO(raw))
    inner = zipfile.ZipFile(io.BytesIO(outer.read("bank.zip")))
    inner.extract("bank-full.csv", DATA_DIR)
    inner.extract("bank-names.txt", DATA_DIR)

    print(f"Done. Files saved to {DATA_DIR}")


if __name__ == "__main__":
    main()