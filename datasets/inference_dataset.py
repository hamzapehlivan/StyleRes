import os
from PIL import Image
from .process_image import ImageProcessor
from torch.utils.data import Dataset

class InferenceDataset(Dataset):
    
    def __init__(self,
                 root_dir,
                 resolution=256,
                 aligner_path=None
                 ):
        """Initializes the dataset.

        Args:
            root_dir: Root directory containing the dataset.
            resolution: The resolution of the returned image.
            aligner_path: Landmark detection model path
        """

        self.root_dir = root_dir
        self.resolution = resolution
        self.image_paths = sorted(os.listdir(self.root_dir))
        self.num_samples = len(self.image_paths)
        self.processor = ImageProcessor(aligner_path)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = dict()
        image_path = self.image_paths[idx]
        image = Image.open(os.path.join(self.root_dir, image_path))
        image = self.processor.align_face(image)
        image = self.processor.preprocess_image(image)
        data.update({'image': image})
        data.update({'name': image_path})
        return data

