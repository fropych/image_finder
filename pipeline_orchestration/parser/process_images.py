from pathlib import Path

import imagehash
import numba
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from PIL import Image


# TODO ERROR WITH B = NONE
@numba.njit(fastmath=True)
def get_duplicates(a, b, splitter=3):
    """
    Parameters
    ----------
    a : array_like
        First hash.
    b : array_like
        Second hash.
    out : ndarray
        Id of duplicates.
    """

    if b.ndim == 1:
        b = b.reshape(-1, 64)

    size_a = len(a)
    size_b = len(b)
    img_diff = np.zeros((size_b, size_a), dtype=np.int32)
    for i in range(size_b):
        for j in range(i, size_a):
            img_diff[i][j] = (np.count_nonzero(a[j] != b[i]) < splitter) * j * (j != i)
    return np.unique(img_diff)


def main(cfg: DictConfig):
    image_df = pd.read_csv(Path(cfg.paths.csv_dir.raw) / "raw_images.csv")

    def get_hashes(path):
        imgs_hash = []
        for img_path in path:
            img = Image.open(img_path)
            imgs_hash.append(imagehash.average_hash(img).hash)

        imgs_hash = np.array(imgs_hash)
        hashes = imgs_hash.reshape(-1, 64)
        return hashes

    template_paths = image_df[image_df.isTemplate].filename.apply(
        lambda x: Path(cfg.paths.images_dir.raw) / x
    )
    template_hashes = get_hashes(template_paths)
    duplicates_index = get_duplicates(template_hashes, template_hashes)[1:]
    names_to_drop = image_df.iloc[
        template_paths.iloc[duplicates_index].index
    ].name.to_list()
    image_df = image_df.query(f"not name in {names_to_drop}")
    image_df.to_csv(
        Path(cfg.paths.csv_dir.processed) / "processed_images.csv",
        encoding="utf-8",
        index=False,
    )


if __name__ == "__main__":
    main()
