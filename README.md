# re-identification
Based on [facenet_pytorch](https://github.com/timesler/facenet-pytorch).
Implemetation of [FaceNet](https://arxiv.org/pdf/1503.03832.pdf) pretrained on [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/).

Реализован скрипт `facenet.py`. 

Пример вызова:
```
python facenet.py --first_dir <your database dir> --second_dir <images to identify> --save_dir <path to save results>
```
Предсказания модели сохраняются в `save_dir/results.csv` (содержит построчно через пробел `person` - `image`)

Точность на 10k случайных тестовых изображениях из датасета [VGGFace2](https://drive.google.com/drive/folders/10lBbk9vo3KOuH9P8PdwQMT2e6X2X7idO) - `0.83` (см. `Demo.ipynb`).
