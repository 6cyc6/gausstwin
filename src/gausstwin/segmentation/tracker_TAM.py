import os
import cv2
import time
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt

from gausstwin.cfg.domain import RigidBody, Rope
from gausstwin.utils.vis_utils import show_track_masks_cv2
from gausstwin.cfg.tracker.gs_track_cfg import GSTrackUnifiedConfig
from gausstwin.utils.load_utils import get_rigid_body_json, get_objs_prompt_json, get_rope_json
from sam2.build_sam import build_sam2_camera_predictor
from efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor


class TAMTracker:
    def __init__(self, cfg: GSTrackUnifiedConfig, use_sam2: bool = False):
        self.cfg = cfg
        if use_sam2:
            self.tam_predictors = [build_sam2_camera_predictor(cfg.sam2_config, cfg.sam2_checkpoint) 
                                   for _ in range(cfg.n_cams)]
        else:
            self.tam_predictors = [build_efficienttam_camera_predictor(cfg.tam_config, cfg.tam_checkpoint, vos_optimized=True) 
                                   for _ in range(cfg.n_cams)]
            
        self.n_cams = cfg.n_cams
        self.frame_idx = 0
        self.obj_idx = 0
        self.obj_id_dict = {}  # maps object name to object id
        
        # logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.is_init = False
        self.is_load_first_frame = False


    def load_first_frames(self, frames):
        assert len(frames) == self.n_cams, "Number of frames must match number of cameras."
        self.init_frames = frames  # save the initial frames for tracking
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(self.n_cams):
                self.tam_predictors[i].load_first_frame(frames[i])
        self.cam_H = frames[0].shape[0]
        self.cam_W = frames[0].shape[1]
        self.masks = torch.ones((self.n_cams, self.cam_H, self.cam_W), dtype=torch.float32, device="cuda:0")  # (N, H, W)
        self.is_load_first_frame = True

    
    def add_new_prompts(self, obj, eval=False, vis=False):
        assert self.is_load_first_frame, "Tracker must load first frames before adding new prompts."
        if obj != "Rope":
            data = get_rigid_body_json(obj, self.cfg.exp_name)
            obj_data = RigidBody(**data)
        else:
            data = get_rope_json("Rope", self.cfg.exp_name)
            obj_data = Rope(**data)
        prompts = obj_data.prompts
        for cam_idx in range(self.n_cams):
            if eval:
                prompt = prompts[f"view_3"]
            else:
                prompt = prompts[f"view_{cam_idx}"]
            _, out_obj_ids, out_mask_logits = self.tam_predictors[cam_idx].add_new_prompt(
                frame_idx=0, 
                obj_id=self.obj_idx, 
                points=prompt.uvs, 
                labels=prompt.labels,
                bbox=prompt.bbox,
            )
            if cam_idx == 0:
                self.obj_id_dict[obj] = self.obj_idx # save the object id
            
            if vis:
                mask = (out_mask_logits > 0).squeeze().cpu().numpy().astype(np.uint8)
                window_name = f"Mask for View {cam_idx}"
                show_track_masks_cv2(self.init_frames[cam_idx], mask, borders=True, window_name=window_name)
        
        self.obj_idx += 1
        self.is_init = True
    
    
    def add_new_prompts_multi(self, eval=False, vis=False):
        assert self.is_load_first_frame, "Tracker must load first frames before adding new prompts."
        data = get_objs_prompt_json(self.cfg.exp_name, sim=False)
        prompts = data["objs"]
        for cam_idx in range(self.n_cams):
            if eval:
                prompt = prompts[f"view_3"]
            else:
                prompt = prompts[f"view_{cam_idx}"]
            _, out_obj_ids, out_mask_logits = self.tam_predictors[cam_idx].add_new_prompt(
                frame_idx=0,
                obj_id=self.obj_idx,
                points=prompt["points"],
                labels=prompt["labels"],
                bbox=prompt["bbox"],
            )
            if cam_idx == 0:
                self.obj_id_dict["objs"] = self.obj_idx # save the object id
            
            if vis:
                mask = (out_mask_logits > 0).squeeze().cpu().numpy().astype(np.uint8)
                window_name = f"Mask for View {cam_idx}"
                show_track_masks_cv2(self.init_frames[cam_idx], mask, borders=True, window_name=window_name)
        
        self.obj_idx += 1
        self.is_init = True
        
        
    def track(self, frames):
        assert self.is_init, "Tracker must be initialized with initial frames and prompts."
        # assert len(frames) == self.n_cams, "Number of frames must match number of cameras."
        
        # time_start = time.time()
        # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        #     masks = self.masks.clone()  # (N, H, W) # 1 for background, 0 for object # init: all ones
        #     for cam_idx in range(self.n_cams):
        #         # Track the current frame
        #         out_obj_ids, out_mask_logits = self.tam_predictors[cam_idx].track(frames[cam_idx])
        #         for i, obj_id in enumerate(out_obj_ids):
        #             obj_mask = (out_mask_logits[i] > 0.0).squeeze()  # (H, W)
        #             print(masks.shape)
        #             print(obj_mask.shape)
        #             masks[cam_idx][obj_mask] = 0.0  # Update the mask for the current camera
        
        masks = self.masks.clone()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            futures = []
            for cam_idx in range(self.n_cams):
                frames_i = frames[cam_idx]
                predictor = self.tam_predictors[cam_idx]
                futures.append(torch.jit.fork(predictor.track, frames_i))

            results = [torch.jit.wait(future) for future in futures]
            
        for cam_idx, (out_obj_ids, out_mask_logits) in enumerate(results):
            # Combine all object masks for this camera into a single boolean mask
            combined_mask = (out_mask_logits > 0.0).any(dim=0).squeeze()
            masks[cam_idx][combined_mask] = 0.0

        return masks
    