import torch
from typing import Any, Dict, Mapping, Tuple
from safetensors.torch import load_model

from yacs.config import CfgNode

from ..utils.geometry import perspective_projection
from .backbones import create_backbone
from .heads import build_mano_head
from . import MANO


class HAMER(torch.nn.Module):

    def __init__(self, cfg: CfgNode):
        """
        Setup HAMER model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        self.cfg = cfg
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
            log.info(f'Loading backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
            self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu')['state_dict'])

        # Create MANO head
        self.mano_head = build_mano_head(cfg)

        # Instantiate MANO model
        mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}
        self.mano = MANO(**mano_cfg)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))

    def get_parameters(self):
        all_params = list(self.mano_head.parameters())
        all_params += list(self.backbone.parameters())
        return all_params

    def forward(self, x) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        # Use RGB image as input
        batch_size = x.shape[0]

        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio
        conditioning_feats = self.backbone(x[:,:,:,32:-32])

        pred_mano_params, pred_cam, _ = self.mano_head(conditioning_feats)

        # Store useful regression outputs to the output dict
        output = {}
        output['pred_cam'] = pred_cam
        pred_mano = dict(zip(
            [f"pred_mano_{k}" for k in pred_mano_params.keys()],
            [v.clone() for v in pred_mano_params.values()]
        ))
        output.update(pred_mano)

        # Compute camera translation
        device = pred_mano_params['hand_pose'].device
        dtype = pred_mano_params['hand_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_mano_params['global_orient'] = pred_mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['hand_pose'] = pred_mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['betas'] = pred_mano_params['betas'].reshape(batch_size, -1)
        mano_output = self.mano(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        return output

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, cfg, map_location=None, strict=None):
        """
        Load the checkpoint file and initialize the model with the weights.
        Can use the original *.ckpt file or a .safetensors file
        """
        model = cls(cfg)

        if checkpoint_path.endswith(".safetensors"):
            if map_location is None:
                device = "cpu"
            elif isinstance(map_location, torch.device):
                if not map_location.index is None:
                    device = map_location.index
                else:
                    device = map_location.type
            else:
                device = map_location
            load_model(model, checkpoint_path, strict=strict, device=device)
        else:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
            except Exception:
                print("Cannot load weights only... load again.")
                checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
                if "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
        
            if strict:
                # remove the discriminator if present
                to_rem = []
                for key in checkpoint:
                    if key.startswith("discriminator"):
                        to_rem.append(key)
                for key in to_rem:
                    del checkpoint[key]
            model.load_state_dict(checkpoint, strict=strict)
        return model.to(map_location)
