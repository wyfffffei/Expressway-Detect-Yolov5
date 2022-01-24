
# YOLOV5_Cheat_Sheet
> @ wyfffffei



## Load YOLOV5 with Pytorch
### Simple Example
```python
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Image
img = 'https://ultralytics.com/images/zidane.jpg'

# Inference
results = model(img)

results.pandas().xyxy[0]  # pandas results
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
```

### Detailed Example
```python
import cv2
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Images
for f in 'zidane.jpg', 'bus.jpg':
    torch.hub.download_url_to_file('https://ultralytics.com/images/' + f, f)  # download 2 images
img1 = Image.open('zidane.jpg')  # PIL image
img2 = cv2.imread('bus.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)
imgs = [img1, img2]  # batch of images

# Inference
results = model(imgs, size=640)  # includes NMS

# Results
results.print()  
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
```

### Inference Settings
```python
model.conf = 0.25  # NMS confidence threshold
      iou = 0.45  # NMS IoU threshold
      agnostic = False  # NMS class-agnostic
      multi_label = False  # NMS multiple labels per box
      classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
      max_det = 1000  # maximum number of detections per image
      amp = False  # Automatic Mixed Precision (AMP) inference

results = model(imgs, size=320)  # custom inference size
```

### Device
```python
model.cpu()  # CPU
model.cuda()  # GPU
model.to(device)  # i.e. device=torch.device(0)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu')  # load on CPU
```

### ScreenShot
```python
import torch
from PIL import ImageGrab

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Image
img = ImageGrab.grab()  # take a screenshot

# Inference
results = model(img)
```

### Training
```python
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)  # load pretrained
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False, pretrained=False)  # load scratch
```

### Custom Models
```python
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best.pt')  # local model
model = torch.hub.load('path/to/yolov5', 'custom', path='path/to/best.pt', source='local')  # local repo
```

### Format Exporting
><https://github.com/ultralytics/yolov5/issues/251>






## Visualize
### Tensorboard
```python
# https://pytorch.org/docs/stable/tensorboard.html
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
model = torchvision.models.resnet50(False)
# Have ResNet model take in grayscale rather than RGB
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
images, labels = next(iter(trainloader))

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()
```

### Wandb
```terminal
pip install wandb
# https://wandb.ai
```

### Plots from Yolov5
```python
from utils.plots import plot_results
plot_results('path/to/results.csv')  # plot 'results.csv' as 'results.png'
```





## Weights & Bias (wandb)

><https://github.com/ultralytics/yolov5/issues/1289>





## Tips for Best Training Results

><https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results>
Q: How to produce the best mAP and training results with YOLOV5 ?


### Dataset
-   **Images per class.**  ≥ 1500 images per class recommended
-   **Instances per class.**  ≥ 10000 instances (labeled objects) per class recommended
-   **Image variety.**  Must be representative of deployed environment. For real-world use cases we recommend images from different times of day, different seasons, different weather, different lighting, different angles, different sources (scraped online, collected locally, different cameras) etc.
-   **Label consistency.**  All instances of all classes in all images must be labelled. Partial labelling will not work.
-   **Label accuracy.**  Labels must closely enclose each object. No space should exist between an object and it's bounding box. No objects should be missing a label.
-   **Background images.**  Background images are images with no objects that are added to a dataset to reduce False Positives (FP). We recommend about 0-10% background images to help reduce FPs (COCO has 1000 background images for reference, 1% of the total). No labels are required for background images.

### Model Selection
Larger models like YOLOv5x and  [YOLOv5x6](https://github.com/ultralytics/yolov5/releases/tag/v5.0)  will produce better results in nearly all cases, but have more parameters, require more CUDA memory to train, and are slower to run. For  **mobile**  deployments we recommend YOLOv5s/m, for  **cloud**  deployments we recommend YOLOv5l/x. See our README  [table](https://github.com/ultralytics/yolov5#pretrained-checkpoints)  for a full comparison of all models.

![YOLOv5 Models](https://github.com/ultralytics/yolov5/releases/download/v1.0/model_comparison.png)

### Training Settings
Before modifying anything, **first train with default settings to establish a performance baseline**. A full list of train.py settings can be found in the [train.py](https://github.com/ultralytics/yolov5/blob/master/train.py) argparser.

-   **Epochs.**  Start with 300 epochs. If this overfits early then you can reduce epochs. If overfitting does not occur after 300 epochs, train longer, i.e. 600, 1200 etc epochs.
-   **Image size.**  COCO trains at native resolution of  `--img 640`, though due to the high amount of small objects in the dataset it can benefit from training at higher resolutions such as  `--img 1280`. If there are many small objects then custom datasets will benefit from training at native or higher resolution. Best inference results are obtained at the same  `--img`  as the training was run at, i.e. if you train at  `--img 1280`  you should also test and detect at  `--img 1280`.
-   **Batch size.**  Use the largest  `--batch-size`  that your hardware allows for. Small batch sizes produce poor batchnorm statistics and should be avoided.
-   **Hyperparameters.**  Default hyperparameters are in  [hyp.scratch.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyp.scratch.yaml). We recommend you train with default hyperparameters first before thinking of modifying any. In general, increasing augmentation hyperparameters will reduce and delay overfitting, allowing for longer trainings and higher final mAP. Reduction in loss component gain hyperparameters like  `hyp['obj']`  will help reduce overfitting in those specific loss components. For an automated method of optimizing these hyperparameters, see our  [Hyperparameter Evolution Tutorial](https://github.com/ultralytics/yolov5/issues/607).

