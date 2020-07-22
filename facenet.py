from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image
import argparse
import numpy as np
import pandas as pd
import cv2
import os
import sys
from tqdm import tqdm

workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


# Our database class
class VGGData(Dataset):
    def __init__(self, root_path):
        super().__init__()

        self.files = [os.path.join(root_path, filename)
                      for filename in sorted(os.listdir(root_path), key=lambda x: int(x[:-4]))]
        self.count = 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return os.path.basename(self.files[index]), Image.open(self.files[index])


def collate_fn(x):
    return x[0]


def transform_img(image):
    """image is PIL object"""
    img = image.resize((image_size, image_size))
    img = np.float32(img).reshape(3, image_size, image_size)
    tensor = torch.from_numpy(img)
    tensor = (tensor - 127.5) / 128.0
    return tensor


def prepare_dataset(images_path):
    global dataset
    global loader
    global embeddings
    dataset = VGGData(images_path)
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    # Find faces in each image from database
    aligned = []
    for name, x in tqdm(loader):
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            # print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)
        else:
            print(f'Face not found on {name} image')
            tensor = transform_img(x)
            aligned.append(tensor)

    # Get embeddings via network from database
    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()
    assert len(embeddings) == len(dataset)


def predict_once(image_path):
    """gets image path and
    returns tuple of (nearest image from database, distance)
    """
    img = Image.open(image_path)
    img_aligned = mtcnn(img, return_prob=False)
    if img_aligned is None:
        img_aligned = transform_img(img)
    img_aligned = [img_aligned]

    img_aligned = torch.stack(img_aligned).to(device)
    img_embeddings = resnet(img_aligned).detach().cpu()

    dists = [(img_embeddings[0] - e).norm().item() for e in embeddings]

    index = np.argmin(dists)
    return dataset[index][0], dists[index]


def predict(path1, path2):
    print(f'Preparing database from {path1}')
    prepare_dataset(path1)

    images = [os.path.join(path2, name)
              for name in sorted(os.listdir(path2), key=lambda x: int(x[:-4]))]
    # np.random.shuffle(images)
    # images = images[:10000]
    
    
    results = []
    print(f'Processing images from {path2}')
    for image_path in tqdm(images):
        results.append([int(predict_once(image_path)[0][:-4]),
                        int(os.path.basename(image_path)[:-4])])

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='FaceNet'
    )
    parser.add_argument(
        '--first_dir',
        default=None,
        type=str
    )
    parser.add_argument(
        '--second_dir',
        default=None,
        type=str
    )
    parser.add_argument(
        '--save_dir',
        default=None,
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    image_size = 160
    # Face detector
    mtcnn = MTCNN(
        image_size=image_size, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    # Embeddings
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Predictions
    results = predict(args.first_dir, args.second_dir)

    # Write results
    if args.save_dir:
        with open(os.path.join(args.save_dir, 'results.csv'), 'w') as f:
            for res in results:
                print(res[0], res[1], file=f)
