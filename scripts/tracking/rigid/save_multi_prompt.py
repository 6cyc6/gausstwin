import os
import json
import torch
import logging
import numpy as np

from PIL import Image

from gausstwin.camera.cam import Camera
from gausstwin.segmentation.prompt import PromptTool
from gausstwin.cfg.builder.builder_cfg import RigidBodyConfig
from gausstwin.utils.vis_utils import to_torch, show_track_masks_cv2
from gausstwin.utils.path_utils import safe_mkdir, get_save_dir, get_builder_cfg_path
from efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor


# load path and configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"
cfg_path = get_builder_cfg_path("rigid_body_builder.yaml")
cfg = RigidBodyConfig.from_yaml(cfg_path)
data_dir = os.path.join(cfg.data_dir, f"trajectory_{cfg.exp_name}")

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info(f"-------------------- Experiment: {cfg.exp_name} --------------------")
logging.info(f"-------------------- Objects: {cfg.obj_list} --------------------")

# Load the camera configuration
cam_cfg_list = []
for cam_idx in range(cfg.num_views):
    f_name = os.path.join(data_dir, f"camera_{cam_idx + 1}.json")
    cam_cfg_list.append(f_name)

cam = Camera(cam_cfg_list)

# predictor
tam_predictor = build_efficienttam_camera_predictor(cfg.tam_config, cfg.tam_checkpoint, device=device)

rgb_list = [] # [(H, W, 3), (H, W, 3), ...]
rgb_np_list = [] # [(H, W, 3), (H, W, 3), ...]
img_path_list = []
# load images
for cam_idx in range(cfg.num_views):
    if cam_idx == 0:
        fix_idx = cfg.fix_idx_1
    elif cam_idx == 1:
        fix_idx = cfg.fix_idx_2
    elif cam_idx == 2:
        fix_idx = cfg.fix_idx_3
    else:
        fix_idx = cfg.fix_idx_4
    # load rgb image
    img_path = os.path.join(data_dir, f"camera/static_{cam_idx + 1}/rgb/{fix_idx}.png")
    img_path_list.append(img_path)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"View file {img_path} does not exist.")
    img = Image.open(img_path)
    img = np.array(img.convert("RGB"))
    rgb_np_list.append(img)  # (H, W, 3)
    img_torch = torch.from_numpy(img).to(device)  # (H, W, 3)
    rgb_list.append(img_torch)  


rgbs = to_torch(rgb_list, device=device) # type: ignore (N, H, W, 3)

gt_rgb_list = [rgb / 255.0 for rgb in rgbs] # [tensor(H, W, 3), ...]
gt_rgbs = rgbs / 255.0 # (N, H, W, 3); ground truth rgb images # 0-255 -> 0-1

prompt_dict = {}
obj_mask_dict = {}
obj = "objs"
i = 0
for view_idx in range(cfg.num_views): # save prompt information for each view (for tracking)
    # set the prompt
    prompt_tool = PromptTool(image_path=img_path_list[view_idx], obj_name=obj)
    point_coords, point_labels, bbox = prompt_tool.run()
    
    # run segmentation
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        tam_predictor.load_first_frame(rgb_np_list[view_idx])  # set the first image
        _, out_obj_ids, out_mask_logits = tam_predictor.add_new_prompt(
            frame_idx=0, obj_id=i, bbox=bbox, points=point_coords, labels=point_labels
        ) # add prompt for the first view
        mask = (out_mask_logits > 0).squeeze().cpu().numpy().astype(np.uint8)
        window_name = f"Mask for View {view_idx} - {obj}"
        show_track_masks_cv2(rgb_np_list[view_idx], mask, point_coords=point_coords, input_labels=point_labels, box_coords=bbox, borders=True, window_name=window_name)
    
    # save prompt information
    prompt_data = {
        "points": [list(pt) for pt in point_coords],
        "labels": point_labels,  # 1 for positive, 0 for negative
        "bbox": bbox 
    }
    if view_idx == 0:
        prompt_dict[obj] = {}  # init the prompt dict for the object
        obj_mask_dict[obj] = []
    view_key = f"view_{view_idx}"
    prompt_dict[obj][view_key] = prompt_data
    # save mask
    obj_mask_dict[obj].append(torch.from_numpy(mask).to(device).bool())  # save the mask for the object
    
    i += 1

save_dir = get_save_dir()
safe_mkdir(f"{save_dir}/{cfg.exp_name}")
save_path = f"{save_dir}/{cfg.exp_name}/prompt_objs.json"

with open(save_path, "w") as f:
    json.dump(prompt_dict, f, indent=4)
    