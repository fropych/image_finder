from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(
        self, dataframe, transform, get_x, get_y=None, test=False, random_seed=42
    ):
        super().__init__()
        self.dataframe = dataframe.copy()
        self.test = test
        self.get_x = get_x
        self.get_y = get_y
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = self.transform(Image.open(row[self.get_x]).convert("RGB"))
        if self.test:
            return (image,)
        return (
            image,
            row["label"],
        )
