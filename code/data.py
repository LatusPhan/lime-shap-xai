"""Oxford-IIIT Pet: images, 37 breed labels, and a foreground mask per image.

The transforms follow the reference LIME PyTorch tutorial exactly. Two separate
pipelines, and the split matters:

    pil_transform         resize and crop only -> the array LIME perturbs
    preprocess_transform  ToTensor and normalise -> what the model consumes

LIME must see the plain picture, not a normalised tensor, because it hides
regions by writing colours into the array. The masks come from the shipped
trimaps, where 1 is foreground, 2 is background and 3 is boundary.
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

IMAGES_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
ANNOTATIONS_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"

NUM_CLASSES = 37
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def pil_transform() -> transforms.Compose:
    """Geometry only. Output is a PIL image at 224x224."""
    return transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224)])


def preprocess_transform() -> transforms.Compose:
    """Tensor conversion and ImageNet normalisation. Applied after LIME."""
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def train_transform() -> transforms.Compose:
    """Fine-tuning only, with light augmentation."""
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def download(root: Path) -> None:
    """Fetch and extract both archives if they are not already present."""
    root.mkdir(parents=True, exist_ok=True)
    for url, folder in ((IMAGES_URL, "images"), (ANNOTATIONS_URL, "annotations")):
        if (root / folder).is_dir():
            continue
        archive = root / Path(url).name
        if not archive.is_file():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, archive)
        print(f"extracting {archive.name}")
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")


def read_split(root: Path, split: str) -> list[tuple[str, int]]:
    """Official split files. Class ids are 1-based in the file, 0-based here."""
    listing = root / "annotations" / ("trainval.txt" if split == "train" else "test.txt")
    if not listing.is_file():
        raise FileNotFoundError(f"{listing} missing. Run: python main.py download")

    entries = []
    for line in listing.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            parts = line.split()
            entries.append((parts[0], int(parts[1]) - 1))
    return entries


class PetImages:
    """Indexable access to the split, returning what each consumer needs."""

    def __init__(self, root: str | Path, split: str = "test") -> None:
        self.root = Path(root)
        self.entries = read_split(self.root, split)
        self.pil = pil_transform()

    def __len__(self) -> int:
        return len(self.entries)

    def image(self, index: int) -> Image.Image:
        """The cropped PIL image, before any normalisation."""
        name, _ = self.entries[index]
        return self.pil(Image.open(self.root / "images" / f"{name}.jpg").convert("RGB"))

    def array(self, index: int) -> np.ndarray:
        """(224, 224, 3) uint8. This is what goes into explain_instance."""
        return np.array(self.image(index))

    def mask(self, index: int) -> np.ndarray:
        """(224, 224) bool, True on the animal. Boundary counts as foreground."""
        name, _ = self.entries[index]
        trimap = Image.open(self.root / "annotations" / "trimaps" / f"{name}.png")
        return np.isin(np.array(self.pil(trimap)), (1, 3))

    def label(self, index: int) -> int:
        return self.entries[index][1]

    def name(self, index: int) -> str:
        return self.entries[index][0]


class PetDataset(PetImages):
    """The same split as a torch Dataset, for fine-tuning."""

    def __init__(self, root: str | Path, split: str = "train", augment: bool = True) -> None:
        super().__init__(root, split)
        self.transform = train_transform() if augment else transforms.Compose(
            [pil_transform(), preprocess_transform()]
        )
        self.augment = augment

    def __getitem__(self, index: int):
        name, label = self.entries[index]
        image = Image.open(self.root / "images" / f"{name}.jpg").convert("RGB")
        return self.transform(image), label
