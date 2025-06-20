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

Now the original README:

# HaMeR: Hand Mesh Recovery
Code repository for the paper:
**Reconstructing Hands in 3D with Transformers**

[Georgios Pavlakos](https://geopavlakos.github.io/), [Dandan Shan](https://ddshan.github.io/), [Ilija Radosavovic](https://people.eecs.berkeley.edu/~ilija/), [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/), [David Fouhey](https://cs.nyu.edu/~fouhey/), [Jitendra Malik](http://people.eecs.berkeley.edu/~malik/)

[![arXiv](https://img.shields.io/badge/arXiv-2312.05251-00ff00.svg)](https://arxiv.org/pdf/2312.05251.pdf)  [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://geopavlakos.github.io/hamer/)     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rQbQzegFWGVOm1n1d-S6koOWDo7F2ucu?usp=sharing)  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/geopavlakos/HaMeR)

![teaser](assets/teaser.jpg)

## News

- [2024/06] HaMeR received the 2nd place award in the Ego-Pose Hands task of the Ego-Exo4D Challenge! Please check the [validation report](https://www.cs.utexas.edu/~pavlakos/hamer/resources/egoexo4d_challenge.pdf).
- [2024/05] We have released the evaluation pipeline!
- [2024/05] We have released the HInt dataset annotations! Please check [here](https://github.com/ddshan/hint).
- [2023/12] Original release!

## Installation
First you need to clone the repo:
```
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer
```

We recommend creating a virtual environment for HaMeR. You can use venv:
```bash
python3.10 -m venv .hamer
source .hamer/bin/activate
```

or alternatively conda:
```bash
conda create --name hamer python=3.10
conda activate hamer
```

Then, you can install the rest of the dependencies. This is for CUDA 11.7, but you can adapt accordingly:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -e .[all]
pip install -v -e third-party/ViTPose
```

You also need to download the trained models:
```bash
bash fetch_demo_data.sh
```

Besides these files, you also need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de) and register to get access to the downloads section.  We only require the right hand model. You need to put `MANO_RIGHT.pkl` under the `_DATA/data/mano` folder.

### Docker Compose

If you wish to use HaMeR with Docker, you can use the following command:

```
docker compose -f ./docker/docker-compose.yml up -d
```

After the image is built successfully, enter the container and run the steps as above:

```
docker compose -f ./docker/docker-compose.yml exec hamer-dev /bin/bash
```

Continue with the installation steps:

```bash
bash fetch_demo_data.sh
```

## Demo
```bash
python demo.py \
    --img_folder example_data --out_folder demo_out \
    --batch_size=48 --side_view --save_mesh --full_frame
```

## HInt Dataset
We have released the annotations for the HInt dataset. Please follow the instructions [here](https://github.com/ddshan/hint)

## Training
First, download the training data to `./hamer_training_data/` by running:
```
bash fetch_training_data.sh
```

Then you can start training using the following command:
```
python train.py exp_name=hamer data=mix_all experiment=hamer_vit_transformer trainer=gpu launcher=local
```
Checkpoints and logs will be saved to `./logs/`.

## Evaluation
Download the [evaluation metadata](https://www.dropbox.com/scl/fi/7ip2vnnu355e2kqbyn1bc/hamer_evaluation_data.tar.gz?rlkey=nb4x10uc8mj2qlfq934t5mdlh) to `./hamer_evaluation_data/`. Additionally, download the FreiHAND, HO-3D, and HInt dataset images and update the corresponding paths in  `hamer/configs/datasets_eval.yaml`.

Run evaluation on multiple datasets as follows, results are stored in `results/eval_regression.csv`. 
```bash
python eval.py --dataset 'FREIHAND-VAL,HO3D-VAL,NEWDAYS-TEST-ALL,NEWDAYS-TEST-VIS,NEWDAYS-TEST-OCC,EPICK-TEST-ALL,EPICK-TEST-VIS,EPICK-TEST-OCC,EGO4D-TEST-ALL,EGO4D-TEST-VIS,EGO4D-TEST-OCC'
```

Results for HInt are stored in `results/eval_regression.csv`. For [FreiHAND](https://github.com/lmb-freiburg/freihand) and [HO-3D](https://codalab.lisn.upsaclay.fr/competitions/4318) you get as output a `.json` file that can be used for evaluation using their corresponding evaluation processes.

## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [4DHumans](https://github.com/shubham-goel/4D-Humans)
- [SLAHMR](https://github.com/vye16/slahmr)
- [ProHMR](https://github.com/nkolot/ProHMR)
- [SPIN](https://github.com/nkolot/SPIN)
- [SMPLify-X](https://github.com/vchoutas/smplify-x)
- [HMR](https://github.com/akanazawa/hmr)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [Detectron2](https://github.com/facebookresearch/detectron2)

Additionally, we thank [StabilityAI](https://stability.ai/) for a generous compute grant that enabled this work.

## Citing
If you find this code useful for your research, please consider citing the following paper:

```bibtex
@inproceedings{pavlakos2024reconstructing,
    title={Reconstructing Hands in 3{D} with Transformers},
    author={Pavlakos, Georgios and Shan, Dandan and Radosavovic, Ilija and Kanazawa, Angjoo and Fouhey, David and Malik, Jitendra},
    booktitle={CVPR},
    year={2024}
}
```
