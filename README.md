## Stripped Down version of HaMeR

I needed a version that can detect handshapes properly, as I used [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/guide) but it didn't suit my needs. So I searched something else and found [HaMeR](https://github.com/geopavlakos/hamer.git) ... I don't need multiple person detection nor the fancy handshape rendering. Only thing i need are the MANO joints. So I decided to strip down the whole thing and remove the unneeded stuff. For Hand detection I still use Mediapipe and afterwards let HaMeR detect the more precise keypoints. My initial tests seemed very promising so I try to make a smaller package out of it. Thanks also to the amazing docker image provided by https://github.com/chaitanya1chawla/hamer.git which allowed me to test the framework and decide what is possible to remove. I also removed the dependencies to the detectron2 framework as well VitPose stuff (as I just need hand inference).

You can use this module after installation, but be aware you need the checkpoint somewhere... Therefore you should edit `CACHE_DIR_HAMER = "./hamer_ckpt/_DATA"` in `hamer/configs/__init__.py` to the location where the ckpt file is. I prepared a repository with the structure for the _DATA folder ([here](https://github.com/Schmetzler/hamer_ckpt.git)). It contains the configuration and a link to the checkpoint file in the README, but you can also find it [here](https://drive.google.com/drive/folders/1hfLQhse5DP460Q-j0d-vG_obCVIsc9Bt?usp=sharing). ~~It uses git-lfs because the checkpoint file is really large and I also had to split it into 2 files so you have to unpack it after cloning the repo (see README.md of the repo).~~ You should also download the mano files from https://mano.is.tue.mpg.de/ and put it into `CACHE_DIR_HAMER/data/`. See README of [hamer_ckpt](https://github.com/Schmetzler/hamer_ckpt.git).

I also included the [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) and [pose_landmarker_lite.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task) from Mediapipe.

To use it as a module:
```python
# to change the CACHE_DIR_HAMER variable import configs first
import hamer.configs as hcfg
hcfg.CACHE_DIR_HAMER = "./hamer_ckpt/_DATA"

from hamer.hamer_module import HAMER
import cv2

# possible modes are "VIDEO" and "IMAGE" and device can be "cpu" or depending on your torch installation "cuda" or something else
hamer = HAMER(mode="IMAGE", device="cpu")
img = cv2.imread("img.png")

# keypoints and bboxes are dictionaries with keys "Right" and "Left" if present
# the image should be in OpenCVs BGR color format
keypoints, bboxes = hamer.process_with_mediapipe(img)

# you can also use the other outputs of the hamer algorithm, they are stored in the class object
res = hamer.result
# when processing with mediapipe the mediapipe keypoints are stored in hamer.result["mp_kpts"]
# look at the code of hamer/models/hamer.py --> forward_step to check the possible dictionary keys

```

### Some notes about usage

* If you want to use another hand bbox detector you can do so and just call `process` with the calculated bboxes in the format `{"Right": bbox_xyxy, "Left": bbox_xyxy}`
* I added both the hand_landmarker and the pose_landmarker for bounding box detection. If the hand_landmarker cannot find a hand it falls back to the pose_landmarker. You can adjust this and use only one of the tasks
* I also added a bounding box only mode... maybe useful for testing purposes

```python
from hamer.hamer_module import HAMER, hand_path, pose_path
# to use only hand_landmarker
hamer = HAMER(task_paths={"hand": hand_path})
# to use only pose_landmarker
hamer = HAMER(task_paths={"pose": pose_path})
# to use the bounding box only mode
_, bboxes = hamer.process_with_mediapipe(img, bbox_only=True)
```

### Export as ONNX model

To make inference easier I implemented the possibility to export it as an ONNX model. Just set a filepath to `export_onnx` in the HAMER constructor and it will export to an onnx model when process is run the first time with actual data. Be aware this is only for the hamer part not the mediapipe bounding box stuff.

```python
from hamer.hamer_module import HAMER
hamer = HAMER(export_onnx="<FILEPATH>")
hamer.process_with_mediapipe(img) # this will export to onnx model
```

#### Further Notes

This will create many files for the model, so I recommend using an empty folder containing the model. Afterwards one can use [onnx_shrink_ray](https://github.com/usefulsensors/onnx_shrink_ray.git) to shrink the model to a usable size (and also only one file) (I also recommend using the float32 model instead of 8 bit model as the shrinking does the job well).

For inference with the onnx model I think I will create a new project to remove some clutter. (Especially the whole config, models, components, heads, backbones whatsoever... which is not really needed for inference (if it is saved in a single graph)).

### Issues

I had an issue importing the `hamer_module` under special unknown circumstances. The error said something about unsafe globals. I found a solution which allowed me to import this module:

```python
import torch
import yacs.config
torch.serialization.add_safe_globals([yacs.config.CfgNode])
# import hamer.configs and hamer_module as usual
```

## Funding

The work was conducted as part of the research project **KI-StudiUm** at the [**Westsächsische Hochschule Zwickau**](https://www.whz.de/english/), which was funded by the [**Federal Ministry of Research, Technology and Space**](https://www.bmftr.bund.de/EN/Home/home_node.html) as part of the federal-state initiative "KI in der Hochschulbildung" under the funding code `16DHBKI063`.

<picture>
     <source srcset="assets/bmftr-en-dark.svg" media="(prefers-color-scheme: dark)">
     <img src="assets/bmftr-en-light.svg" height="75px">
</picture>
<picture>
     <source srcset="assets/whz-en-dark.svg" media="(prefers-color-scheme: dark)">
     <img src="assets/whz-en-light.svg" height="75px">
</picture>

## Citing
If you find this code useful please cite the original research paper:

```bibtex
@inproceedings{pavlakos2024reconstructing,
    title={Reconstructing Hands in 3{D} with Transformers},
    author={Pavlakos, Georgios and Shan, Dandan and Radosavovic, Ilija and Kanazawa, Angjoo and Fouhey, David and Malik, Jitendra},
    booktitle={CVPR},
    year={2024}
}
```
