from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.utils.hand_detector import HandDetector
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD

from importlib.resources import files, as_file
from contextlib import ExitStack
import atexit
file_manager = ExitStack()
atexit.register(file_manager.close)
ref_hand = files('hamer') / 'hand_landmarker.task'
ref_pose = files('hamer') / 'pose_landmarker_lite.task'
hand_path = file_manager.enter_context(as_file(ref_hand)).as_posix()
pose_path = file_manager.enter_context(as_file(ref_pose)).as_posix()

class HAMER():
    def __init__(self, mode="IMAGE", device="cpu", chkpt_path=DEFAULT_CHECKPOINT, task_paths={"hand": hand_path, "pose": pose_path}):
        self.device = torch.device(device)
        self.hamer, self.model_cfg = load_hamer(chkpt_path, map_location=self.device)
        self.hamer.eval()
        
        self.hand_det = HandDetector(taskfiles=task_paths, mode=mode)
        self.result = None
        self.__frameNr = 0
        
    def process(self, cv_img, rescale_factor=2.0, bbox_only=False):
        bboxes = []
        is_right = []
        
        _bboxes = self.hand_det.detect_bboxes(cv_img)
        
        if bbox_only:
            return {}, _bboxes

        # Use hands based on hand keypoint detections
        for name, bbox in _bboxes.items():
            if len(bbox) > 0:
                bboxes.append(bbox)
                is_right.append(1 if name == "Right" else 0)
            
        if len(bboxes) == 0:
            return {}, {}
        
        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(self.model_cfg, cv_img, boxes, right, rescale_factor=rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        
        result = {}
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.hamer(batch)
                
            for i, right in enumerate(batch["right"].cpu().detach()):
                kp3d = np.array(out["pred_keypoints_3d"].cpu().detach()[i])
                if int(right) == 1:
                    result["Right"] = kp3d
                else:
                    result["Left"] = kp3d
            self.result = out
        return result, _bboxes
