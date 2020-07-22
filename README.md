# re-identification
Based on [facenet_pytorch](https://github.com/timesler/facenet-pytorch).
Implemetation of [FaceNet](https://arxiv.org/pdf/1503.03832.pdf) pretrained on [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/).

Here is `facenet.py` script. 

For example:
```
python facenet.py --first_dir <your database dir> --second_dir <images to identify> \
                  --save_dir <path to save results>
```
Predictions will be saved into `save_dir/results.csv` (`person image` line by line)

Accuracy on 10k random test images from [VGGFace2](https://drive.google.com/drive/folders/10lBbk9vo3KOuH9P8PdwQMT2e6X2X7idO) is `0.83` (see `Demo.ipynb`).
