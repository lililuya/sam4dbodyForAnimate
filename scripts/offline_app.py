import os
ROOT = os.path.dirname(os.path.dirname(__file__))

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(top_dir)
sys.path.append(os.path.join(top_dir, 'models', 'sam_3d_body'))
sys.path.append(os.path.join(top_dir, 'models', 'diffusion_vas'))

import uuid
from datetime import datetime

def gen_id():
    t = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    u = uuid.uuid4().hex[:8]
    return f"{t}_{u}"

import argparse
import time
import cv2
import glob
import random
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

from scripts.app_4d_pipeline import build_4d_context, run_4d_pipeline_from_context
from scripts.offline_tracking_compat import unpack_propagate_output
from utils import draw_point_marker, mask_painter, images_to_mp4, DAVIS_PALETTE, jpg_folder_to_mp4, is_super_long_or_wide, keep_largest_component, is_skinny_mask, bbox_from_mask, gpu_profile, resize_mask_with_unique_label
from scripts.offline_completion_indexing import build_completion_window_from_ious

from models.sam_3d_body.sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from models.sam_3d_body.notebook.utils import process_image_with_mask, save_mesh_results
from models.sam_3d_body.tools.vis_utils import visualize_sample_together, visualize_sample
from models.diffusion_vas.demo import init_amodal_segmentation_model, init_rgb_model, init_depth_model, load_and_transform_masks, load_and_transform_rgbs, rgb_to_depth

import torch
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")
if device.type == "cuda":
    # use bfloat16 for the entire notebook
    # torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 3 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


def build_sam3_from_config(cfg):
    """
    Construct and return your SAM-3 model from config.
    You replace this with your real init code.
    """
    from models.sam3.sam3.model_builder import build_sam3_video_model

    sam3_model = build_sam3_video_model(checkpoint_path=cfg.sam3['ckpt_path'])
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone

    return sam3_model, predictor


def read_frame_at(path: str, idx: int):
    """Read a specific frame (by index) from a video file."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def build_sam3_3d_body_config(cfg):
    mhr_path = cfg.sam_3d_body['mhr_path']
    fov_path = cfg.sam_3d_body['fov_path']
    detector_path = cfg.sam_3d_body['detector_path']
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        cfg.sam_3d_body['ckpt_path'], device=device, mhr_path=mhr_path
    )
    
    human_detector, human_segmentor, fov_estimator = None, None, None
    from models.sam_3d_body.tools.build_fov_estimator import FOVEstimator
    fov_estimator = FOVEstimator(name='moge2', device=device, path=fov_path)
    from models.sam_3d_body.tools.build_detector import HumanDetector
    human_detector = HumanDetector(name="vitdet", device=device, path=detector_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    return estimator


def build_diffusion_vas_config(cfg):
    model_path_mask = cfg.completion['model_path_mask']
    model_path_rgb = cfg.completion['model_path_rgb']
    depth_encoder = cfg.completion['depth_encoder']
    model_path_depth = cfg.completion['model_path_depth']
    max_occ_len = min(cfg.completion['max_occ_len'], cfg.sam_3d_body['batch_size'])

    generator = torch.manual_seed(23)

    pipeline_mask = init_amodal_segmentation_model(model_path_mask)
    pipeline_rgb = init_rgb_model(model_path_rgb)
    depth_model = init_depth_model(model_path_depth, depth_encoder)

    return pipeline_mask, pipeline_rgb, depth_model, max_occ_len, generator


class OfflineApp:
    def __init__(self, config_path: str = os.path.join(ROOT, "configs", "body4d.yaml")):
        """Initialize CONFIG, SAM3_MODEL, and global RUNTIME dict."""
        self.CONFIG = OmegaConf.load(config_path)
        self.sam3_model, self.predictor = build_sam3_from_config(self.CONFIG)
        self.sam3_3d_body_model = build_sam3_3d_body_config(self.CONFIG)

        if self.CONFIG.completion.get('enable', False):
            self.pipeline_mask, self.pipeline_rgb, self.depth_model, self.max_occ_len, self.generator = build_diffusion_vas_config(self.CONFIG)
        else:
            self.pipeline_mask, self.pipeline_rgb, self.depth_model, self.max_occ_len, self.generator = None, None, None, None, None
        
        self.RUNTIME = {}  # clear any old state
        self.OUTPUT_DIR = os.path.join(self.CONFIG.runtime['output_dir'], gen_id())
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.RUNTIME['batch_size'] = self.CONFIG.sam_3d_body.get('batch_size', 1)
        self.RUNTIME['detection_resolution'] = self.CONFIG.completion.get('detection_resolution', [256, 512])
        self.RUNTIME['completion_resolution'] = self.CONFIG.completion.get('completion_resolution', [512, 1024])
        self.RUNTIME['smpl_export'] = self.CONFIG.runtime.get('smpl_export', False)
        self.RUNTIME['bboxes'] = None

    def on_mask_generation(self, video_path: str=None, start_frame_idx: int = 0, max_frame_num_to_track: int = 1800):
        """
        Mask generation across the video.
        Currently runs SAM-3 propagation and renders a mask video.
        """
        print("[DEBUG] Mask Generation button clicked.")

        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for propagate_output in self.predictor.propagate_in_video(
            self.RUNTIME['inference_state'],
            start_frame_idx=0,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=False,
            propagate_preflight=True,
        ):
            frame_idx, obj_ids, low_res_masks, video_res_masks = unpack_propagate_output(propagate_output)
            video_segments[frame_idx] = {
                out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(self.RUNTIME['out_obj_ids'])
            } 

        # render the segmentation results every few frames
        vis_frame_stride = 1
        out_h = self.RUNTIME['inference_state']['video_height']
        out_w = self.RUNTIME['inference_state']['video_width']
        # img_to_video = []

        IMAGE_PATH = os.path.join(self.OUTPUT_DIR, 'images') # for sam3-3d-body
        MASKS_PATH = os.path.join(self.OUTPUT_DIR, 'masks')  # for sam3-3d-body
        os.makedirs(IMAGE_PATH, exist_ok=True)
        os.makedirs(MASKS_PATH, exist_ok=True)

        for out_frame_idx in range(0, len(video_segments), vis_frame_stride):
            img = self.RUNTIME['inference_state']['images'][out_frame_idx].detach().float().cpu()
            img = (img + 1) / 2
            img = img.clamp(0, 1)
            img = F.interpolate(
                img.unsqueeze(0),
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            img = img.permute(1, 2, 0)
            img = (img.numpy() * 255).astype("uint8")
            img_pil = Image.fromarray(img).convert('RGB')
            msk = np.zeros_like(img[:, :, 0])
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                mask = (out_mask[0] > 0).astype(np.uint8) * 255
                # img = mask_painter(img, mask, mask_color=4 + out_obj_id)
                msk[mask == 255] = out_obj_id
            # img_to_video.append(img)

            msk_pil = Image.fromarray(msk).convert('P')
            msk_pil.putpalette(DAVIS_PALETTE)
            img_pil.save(os.path.join(IMAGE_PATH, f"{out_frame_idx+start_frame_idx:08d}.jpg"))
            msk_pil.save(os.path.join(MASKS_PATH, f"{out_frame_idx+start_frame_idx:08d}.png"))

        out_video_path = os.path.join(self.OUTPUT_DIR, f"video_mask_{time.time():.0f}.mp4")
        # images_to_mp4(img_to_video, out_video_path, fps=self.RUNTIME['video_fps'])

        return out_video_path

    def on_4d_generation(self, video_path: str=None):
        context = build_4d_context(
            input_dir=self.OUTPUT_DIR,
            output_dir=self.OUTPUT_DIR,
            runtime=self.RUNTIME,
            sam3_3d_body_model=self.sam3_3d_body_model,
            pipeline_mask=self.pipeline_mask,
            pipeline_rgb=self.pipeline_rgb,
            depth_model=self.depth_model,
            predictor=self.predictor,
            generator=self.generator,
        )
        return run_4d_pipeline_from_context(context)


def inference(args):
    # init configs and cover with cmd options
    predictor = OfflineApp()
    if args.output_dir is not None:
        predictor.OUTPUT_DIR = args.output_dir
        os.makedirs(predictor.OUTPUT_DIR, exist_ok=True)

    # human detection on the frame where human FIRST appear
    if os.path.isfile(args.input_video) and args.input_video.endswith(".mp4"):
        input_type = "video"
        image = read_frame_at(args.input_video, 0)
        width, height = image.size
        for starting_frame_idx in range(10, 100):
            image = np.array(read_frame_at(args.input_video, starting_frame_idx))
            outputs = predictor.sam3_3d_body_model.process_one_image(image, bbox_thr=0.6,)
            if len(outputs) > 0:
                break
        
        inference_state = predictor.predictor.init_state(video_path=args.input_video)
        predictor.predictor.clear_all_points_in_video(inference_state)
        predictor.RUNTIME['inference_state'] = inference_state
        predictor.RUNTIME['out_obj_ids'] = []

        # 1. load bbox (first frame)
        for obj_id, output in enumerate(outputs):
            # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
            xmin, ymin, xmax, ymax = output['bbox']
            rel_box = [[xmin / width, ymin / height, xmax / width, ymax / height]]
            rel_box = np.array(rel_box, dtype=np.float32)
            _, predictor.RUNTIME['out_obj_ids'], low_res_masks, video_res_masks = predictor.predictor.add_new_points_or_box(
                inference_state=predictor.RUNTIME['inference_state'],
                frame_idx=starting_frame_idx,
                obj_id=obj_id+1,
                box=rel_box,
            )

    elif os.path.isdir(args.input_video):
        input_type = "images"
        image_list = glob.glob(os.path.join(args.input_video, '*.jpg'))
        image_list.sort()
        image = Image.open(image_list[0]).convert('RGB')
        width, height = image.size
        starting_frame_idx = 0
        for image_path in image_list:
            outputs = predictor.sam3_3d_body_model.process_one_image(image_path, bbox_thr=0.6,)
            if len(outputs) > 0:
                break
            starting_frame_idx += 1

        inference_state = predictor.predictor.init_state(video_path=image_list)
        predictor.predictor.clear_all_points_in_video(inference_state)
        predictor.RUNTIME['inference_state'] = inference_state
        predictor.RUNTIME['out_obj_ids'] = []

        # 1. load bbox (first frame)
        for obj_id, output in enumerate(outputs):
            # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
            xmin, ymin, xmax, ymax = output['bbox']
            rel_box = [[xmin / width, ymin / height, xmax / width, ymax / height]]
            rel_box = np.array(rel_box, dtype=np.float32)
            _, predictor.RUNTIME['out_obj_ids'], low_res_masks, video_res_masks = predictor.predictor.add_new_points_or_box(
                inference_state=predictor.RUNTIME['inference_state'],
                frame_idx=starting_frame_idx,
                obj_id=obj_id+1,
                box=rel_box,
            )

    # 2. tracking
    predictor.on_mask_generation(start_frame_idx=0)
    # 3. hmr upon masks
    with torch.autocast("cuda", enabled=False):
        predictor.on_4d_generation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline 4D Body Generation from Videos")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video (either *.mp4 or a directory containing image sequences)")
    args = parser.parse_args()

    input_path = args.input_video
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"--input_video does not exist: {input_path}")
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(".mp4"):
            raise ValueError(
                f"--input_video must be an .mp4 file or a directory, got file: {input_path}"
            )
    elif os.path.isdir(input_path):
        # Optional: check whether the directory contains images
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        images = [
            f for f in os.listdir(input_path)
            if f.lower().endswith(valid_ext)
        ]
        if len(images) == 0:
            raise ValueError(
                f"--input_video directory contains no image files: {input_path}"
            )
    else:
        raise ValueError(
            f"--input_video must be an .mp4 file or a directory: {input_path}"
        )

    inference(args)
