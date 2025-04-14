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
    def __init__(self, mode="IMAGE", device="cpu", chkpt_path=DEFAULT_CHECKPOINT, task_paths={"hand": hand_path, "pose": pose_path}, lazy=False):
        """
        Init the HAMER module. If lazy is True the hamer checkpoint isn't loaded until first use.
        This way you can utilize this class in a mediapipe only mode without loading the hamer checkpoint.
        (Beware in mediapipe only mode you must call process_with_mediapipe)
        """
        self.device = torch.device(device)
        self.hamer = None
        self.chkpt_path = chkpt_path
        self._mp_only = False
        if not Path(self.chkpt_path).exists():
            print(f"Checkpoint {chkpt_path} does not exist. Fallback to Mediapipe only mode. Running process will result in an error.")
            self._mp_only = True

        if not lazy and not self._mp_only:
            self.hamer, self.model_cfg = load_hamer(self.chkpt_path, map_location=self.device)
            self.hamer.eval()

        self.hand_det = HandDetector(taskfiles=task_paths, mode=mode)
        self.result = {}
        
    def process_with_mediapipe(self, cv_img, rescale_factor=2.0, bbox_only=False):      
        bboxes = self.hand_det.detect_bboxes(cv_img)
        
        if "mp_kpts" in self.hand_det.result:
            self.result["mp_kpts"] = self.hand_det.result["mp_kpts"]
        else:
            self.result = {}

        if bbox_only or self._mp_only:
            return {}, bboxes

        return self.process(cv_img, bboxes, rescale_factor)

    def process(self, cv_img, bboxes, rescale_factor=2.0):
        """
        bboxes must be a dictionary and looks as follows:
        { "Right" : bbox_xyxy, "Left" : bbox_xyxy } in range (0,1)
        """
        _bboxes = []
        _is_right = []
        h, w = cv_img.shape[:2]

        # Use hands based on hand keypoint detections
        for name, bbox in bboxes.items():
            if len(bbox) > 0:
                # expand bbox to image space
                _bboxes.append([bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h])
                _is_right.append(1 if name == "Right" else 0)
            
        if len(_bboxes) == 0:
            self.result = {}
            return {}, {}

        if self.hamer is None:
            self.hamer, self.model_cfg = load_hamer(self.chkpt_path, map_location=self.device)
            self.hamer.eval()
        
        boxes = np.stack(_bboxes)
        right = np.stack(_is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(self.model_cfg, cv_img, boxes, right, rescale_factor=rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        
        result = {}
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.hamer(batch)

            del batch["img"]
            del batch["personid"]
            out["input"] = batch
                
            for i, right in enumerate(batch["right"].cpu().detach()):
                kp3d = np.array(out["pred_keypoints_3d"].cpu().detach()[i])
                if int(right) == 1:
                    result["Right"] = kp3d
                else:
                    result["Left"] = kp3d
            self.result.update(recursive_to(out, "cpu"))
        return result, bboxes
