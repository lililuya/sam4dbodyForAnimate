import argparse
import glob
import os
import random
import sys
import time
from contextlib import nullcontext

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "models", "sam_3d_body"))
sys.path.append(os.path.join(ROOT, "models", "diffusion_vas"))

from scripts.mask_video_export import export_binary_mask_videos
from scripts.openpose_export import build_openpose_people, write_openpose_frame_json

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test envs
    torch = None


IMAGE_PATTERNS = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff", "*.webp"]


def _autocast_disabled():
    if torch is None:
        return nullcontext()
    return torch.autocast("cuda", enabled=False)


def process_image_with_mask(*args, **kwargs):
    from models.sam_3d_body.notebook.utils import process_image_with_mask as _process_image_with_mask

    return _process_image_with_mask(*args, **kwargs)


def save_mesh_results(*args, **kwargs):
    from models.sam_3d_body.notebook.utils import save_mesh_results as _save_mesh_results

    return _save_mesh_results(*args, **kwargs)


def visualize_sample(*args, **kwargs):
    from models.sam_3d_body.tools.vis_utils import visualize_sample as _visualize_sample

    return _visualize_sample(*args, **kwargs)


def visualize_sample_together(*args, **kwargs):
    from models.sam_3d_body.tools.vis_utils import visualize_sample_together as _visualize_sample_together

    return _visualize_sample_together(*args, **kwargs)


def load_and_transform_masks(*args, **kwargs):
    from models.diffusion_vas.demo import load_and_transform_masks as _load_and_transform_masks

    return _load_and_transform_masks(*args, **kwargs)


def load_and_transform_rgbs(*args, **kwargs):
    from models.diffusion_vas.demo import load_and_transform_rgbs as _load_and_transform_rgbs

    return _load_and_transform_rgbs(*args, **kwargs)


def rgb_to_depth(*args, **kwargs):
    from models.diffusion_vas.demo import rgb_to_depth as _rgb_to_depth

    return _rgb_to_depth(*args, **kwargs)


def jpg_folder_to_mp4(*args, **kwargs):
    from utils.image2video import jpg_folder_to_mp4 as _jpg_folder_to_mp4

    return _jpg_folder_to_mp4(*args, **kwargs)


def _load_runtime_utils():
    from utils import DAVIS_PALETTE
    from utils.mask_utils import (
        bbox_from_mask,
        is_skinny_mask,
        is_super_long_or_wide,
        keep_largest_component,
        resize_mask_with_unique_label,
    )

    return {
        "DAVIS_PALETTE": DAVIS_PALETTE,
        "bbox_from_mask": bbox_from_mask,
        "is_skinny_mask": is_skinny_mask,
        "is_super_long_or_wide": is_super_long_or_wide,
        "keep_largest_component": keep_largest_component,
        "resize_mask_with_unique_label": resize_mask_with_unique_label,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Offline 4D body generation with OpenPose JSON and binary mask video export"
    )
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument(
        "--input_video",
        type=str,
        required=True,
        help="Path to the input video (.mp4) or a directory containing frames",
    )
    parser.add_argument("--mask_video_fps", type=int, default=None)
    return parser


def load_base_offline_module():
    from scripts import offline_app as base_offline_app

    return base_offline_app


def resolve_mask_video_fps(input_video, override_fps=None, default_fps=25):
    if override_fps is not None:
        return int(override_fps)
    if os.path.isfile(input_video) and input_video.lower().endswith(".mp4"):
        capture = cv2.VideoCapture(input_video)
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        if fps and fps > 1:
            return int(round(fps))
    return int(default_fps)


def list_input_images(input_dir):
    image_paths = []
    for pattern in IMAGE_PATTERNS:
        image_paths.extend(glob.glob(os.path.join(input_dir, pattern)))
    return sorted(image_paths)


def validate_input_path(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"--input_video does not exist: {input_path}")
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(".mp4"):
            raise ValueError(f"--input_video must be an .mp4 file or a directory, got file: {input_path}")
        return
    if os.path.isdir(input_path):
        images = list_input_images(input_path)
        if not images:
            raise ValueError(f"--input_video directory contains no image files: {input_path}")
        return
    raise ValueError(f"--input_video must be an .mp4 file or a directory: {input_path}")


def run_export_pipeline(app, input_video, fps):
    app.on_mask_generation(start_frame_idx=0)
    export_binary_mask_videos(
        os.path.join(app.OUTPUT_DIR, "masks"),
        os.path.join(app.OUTPUT_DIR, "mask_videos"),
        app.RUNTIME["out_obj_ids"],
        fps=fps,
    )
    with _autocast_disabled():
        app.on_4d_generation(video_path=input_video)


def build_export_app_class(base_module):
    class OfflineAppExport(base_module.OfflineApp):
        def _write_openpose_frame(self, image_path, mask_output, id_current):
            frame_stem = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.join(self.OUTPUT_DIR, "openpose_json")
            people = []
            if mask_output and id_current:
                people = build_openpose_people(mask_output, id_current)
            write_openpose_frame_json(output_dir, frame_stem, people)

        def on_4d_generation(self, video_path=None):
            print("[DEBUG] 4D Generation button clicked.")

            image_path_root = os.path.join(self.OUTPUT_DIR, "images")
            masks_path_root = os.path.join(self.OUTPUT_DIR, "masks")
            images_list = sorted(
                [image for pattern in IMAGE_PATTERNS for image in glob.glob(os.path.join(image_path_root, pattern))]
            )
            masks_list = sorted(
                [image for pattern in IMAGE_PATTERNS for image in glob.glob(os.path.join(masks_path_root, pattern))]
            )

            os.makedirs(f"{self.OUTPUT_DIR}/rendered_frames", exist_ok=True)
            os.makedirs(os.path.join(self.OUTPUT_DIR, "openpose_json"), exist_ok=True)
            for obj_id in self.RUNTIME["out_obj_ids"]:
                os.makedirs(f"{self.OUTPUT_DIR}/mesh_4d_individual/{obj_id}", exist_ok=True)
                os.makedirs(f"{self.OUTPUT_DIR}/focal_4d_individual/{obj_id}", exist_ok=True)
                os.makedirs(f"{self.OUTPUT_DIR}/rendered_frames_individual/{obj_id}", exist_ok=True)

            batch_size = self.RUNTIME["batch_size"]
            n = len(images_list)

            pred_res = self.RUNTIME["detection_resolution"]
            pred_res_hi = self.RUNTIME["completion_resolution"]
            modal_pixels_list = []
            if self.pipeline_mask is not None:
                runtime_utils = _load_runtime_utils()
                davis_palette = runtime_utils["DAVIS_PALETTE"]
                bbox_from_mask = runtime_utils["bbox_from_mask"]
                is_skinny_mask = runtime_utils["is_skinny_mask"]
                is_super_long_or_wide = runtime_utils["is_super_long_or_wide"]
                keep_largest_component = runtime_utils["keep_largest_component"]
                resize_mask_with_unique_label = runtime_utils["resize_mask_with_unique_label"]
                for obj_id in self.RUNTIME["out_obj_ids"]:
                    modal_pixels, ori_shape = load_and_transform_masks(
                        self.OUTPUT_DIR + "/masks",
                        resolution=pred_res,
                        obj_id=obj_id,
                    )
                    modal_pixels_list.append(modal_pixels)
                rgb_pixels, _, _ = load_and_transform_rgbs(self.OUTPUT_DIR + "/images", resolution=pred_res)
                depth_pixels = rgb_to_depth(rgb_pixels, self.depth_model)

            mhr_shape_scale_dict = {}
            obj_ratio_dict = {}

            for i in tqdm(range(0, n, batch_size)):
                batch_images = images_list[i : i + batch_size]
                batch_masks = masks_list[i : i + batch_size]

                with Image.open(batch_masks[0]) as first_mask_image:
                    width, height = first_mask_image.size

                idx_dict = {}
                idx_path = {}
                occ_dict = {}
                if len(modal_pixels_list) > 0:
                    print("detect occlusions ...")
                    pred_amodal_masks_dict = {}
                    for modal_pixels, obj_id in zip(modal_pixels_list, self.RUNTIME["out_obj_ids"]):
                        pred_amodal_masks = self.pipeline_mask(
                            modal_pixels[:, i : i + batch_size, :, :, :],
                            depth_pixels[:, i : i + batch_size, :, :, :],
                            height=pred_res[0],
                            width=pred_res[1],
                            num_frames=modal_pixels[:, i : i + batch_size, :, :, :].shape[1],
                            decode_chunk_size=8,
                            motion_bucket_id=127,
                            fps=8,
                            noise_aug_strength=0.02,
                            min_guidance_scale=1.5,
                            max_guidance_scale=1.5,
                            generator=self.generator,
                        ).frames[0]

                        pred_amodal_masks_com = [
                            np.array(img.resize((pred_res_hi[1], pred_res_hi[0]))) for img in pred_amodal_masks
                        ]
                        pred_amodal_masks_com = np.array(pred_amodal_masks_com).astype("uint8")
                        pred_amodal_masks_com = (pred_amodal_masks_com.sum(axis=-1) > 600).astype("uint8")
                        pred_amodal_masks_com = [keep_largest_component(mask_current) for mask_current in pred_amodal_masks_com]

                        pred_amodal_masks = [np.array(img.resize((width, height))) for img in pred_amodal_masks]
                        pred_amodal_masks = np.array(pred_amodal_masks).astype("uint8")
                        pred_amodal_masks = (pred_amodal_masks.sum(axis=-1) > 600).astype("uint8")
                        pred_amodal_masks = [keep_largest_component(mask_current) for mask_current in pred_amodal_masks]

                        masks = [(np.array(Image.open(mask_path).convert("P")) == obj_id).astype("uint8") for mask_path in batch_masks]
                        ious = []
                        masks_margin_shrink = [mask_current.copy() for mask_current in masks]
                        mask_height, mask_width = masks_margin_shrink[0].shape
                        for batch_index, (modal_mask, amodal_mask) in enumerate(zip(masks, pred_amodal_masks)):
                            zero_mask_cp = np.zeros_like(masks_margin_shrink[batch_index])
                            zero_mask_cp[masks_margin_shrink[batch_index] == 1] = 255
                            mask_binary_cp = zero_mask_cp.astype(np.uint8)
                            mask_binary_cp[: int(mask_height * 0.05), :] = 0
                            mask_binary_cp[-int(mask_height * 0.05) :, :] = 0
                            mask_binary_cp[:, : int(mask_width * 0.05)] = 0
                            mask_binary_cp[:, -int(mask_width * 0.05) :] = 0
                            if mask_binary_cp.max() == 0:
                                ious.append(1.0)
                                continue

                            area_modal = (modal_mask > 0).sum()
                            area_amodal = (amodal_mask > 0).sum()
                            if area_modal == 0 and area_amodal == 0:
                                ious.append(1.0)
                            elif area_modal > area_amodal:
                                ious.append(1.0)
                            else:
                                inter = np.logical_and(modal_mask > 0, amodal_mask > 0).sum()
                                union = np.logical_or(modal_mask > 0, amodal_mask > 0).sum()
                                ious.append(inter / (union + 1e-6))

                            if i == 0 and batch_index == 0:
                                if ious[0] < 0.7:
                                    obj_ratio_dict[obj_id] = bbox_from_mask(amodal_mask)
                                else:
                                    obj_ratio_dict[obj_id] = bbox_from_mask(modal_mask)

                        for pred_index, pred_mask_com in enumerate(pred_amodal_masks_com):
                            if masks[pred_index].sum() > pred_amodal_masks[pred_index].sum():
                                ious[pred_index] = 1.0
                                pred_amodal_masks_com[pred_index] = resize_mask_with_unique_label(
                                    masks[pred_index],
                                    pred_res_hi[0],
                                    pred_res_hi[1],
                                    obj_id,
                                )
                            elif is_super_long_or_wide(pred_amodal_masks[pred_index], obj_id):
                                ious[pred_index] = 1.0
                                pred_amodal_masks_com[pred_index] = resize_mask_with_unique_label(
                                    masks[pred_index],
                                    pred_res_hi[0],
                                    pred_res_hi[1],
                                    obj_id,
                                )
                            elif is_skinny_mask(pred_amodal_masks[pred_index]):
                                ious[pred_index] = 1.0
                                pred_amodal_masks_com[pred_index] = resize_mask_with_unique_label(
                                    masks[pred_index],
                                    pred_res_hi[0],
                                    pred_res_hi[1],
                                    obj_id,
                                )

                        pred_amodal_masks_dict[obj_id] = pred_amodal_masks_com
                        occluded_indexes = [idx for idx, iou in enumerate(ious) if iou < 0.7]
                        occ_dict[obj_id] = [1 if iou > 0.7 else 0 for iou in ious]

                        if occluded_indexes:
                            start = max(0, occluded_indexes[0] - 2)
                            end = min(
                                modal_pixels[:, i : i + batch_size, :, :, :].shape[1] - 1,
                                occluded_indexes[-1] + 2,
                            )
                            idx_dict[obj_id] = (start, end)
                            completion_path = "".join(random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=4))
                            completion_image_path = f"{self.OUTPUT_DIR}/completion/{completion_path}/images"
                            completion_masks_path = f"{self.OUTPUT_DIR}/completion/{completion_path}/masks"
                            os.makedirs(completion_image_path, exist_ok=True)
                            os.makedirs(completion_masks_path, exist_ok=True)
                            idx_path[obj_id] = {"images": completion_image_path, "masks": completion_masks_path}
                            for idx_ in range(start, end):
                                mask_idx = pred_amodal_masks[idx_].copy()
                                mask_idx[mask_idx > 0] = obj_id
                                mask_idx = Image.fromarray(mask_idx).convert("P")
                                mask_idx.putpalette(davis_palette)
                                mask_idx.save(os.path.join(completion_masks_path, f"{idx_:08d}.png"))

                    for obj_id, (start, end) in idx_dict.items():
                        completion_image_path = idx_path[obj_id]["images"]
                        modal_pixels_current, ori_shape = load_and_transform_masks(
                            self.OUTPUT_DIR + "/masks",
                            resolution=pred_res_hi,
                            obj_id=obj_id,
                        )
                        rgb_pixels_current, _, _ = load_and_transform_rgbs(
                            self.OUTPUT_DIR + "/images",
                            resolution=pred_res_hi,
                        )
                        modal_pixels_current = modal_pixels_current[:, i : i + batch_size, :, :, :]
                        modal_pixels_current = modal_pixels_current[:, start:end]
                        pred_amodal_masks_current = pred_amodal_masks_dict[obj_id][start:end]
                        modal_mask_union = (modal_pixels_current[0, :, 0, :, :].cpu().numpy() > 0).astype("uint8")
                        pred_amodal_masks_current = np.logical_or(pred_amodal_masks_current, modal_mask_union).astype("uint8")
                        pred_amodal_masks_tensor = (
                            torch.from_numpy(np.where(pred_amodal_masks_current == 0, -1, 1))
                            .float()
                            .unsqueeze(0)
                            .unsqueeze(2)
                            .repeat(1, 1, 3, 1, 1)
                        )

                        rgb_pixels_current = rgb_pixels_current[:, i : i + batch_size, :, :, :][:, start:end]
                        modal_obj_mask = (modal_pixels_current > 0).float()
                        modal_background = 1 - modal_obj_mask
                        rgb_pixels_current = (rgb_pixels_current + 1) / 2
                        modal_rgb_pixels = rgb_pixels_current * modal_obj_mask + modal_background
                        modal_rgb_pixels = modal_rgb_pixels * 2 - 1

                        print("content completion by diffusion-vas ...")
                        pred_amodal_rgb = self.pipeline_rgb(
                            modal_rgb_pixels,
                            pred_amodal_masks_tensor,
                            height=pred_res_hi[0],
                            width=pred_res_hi[1],
                            num_frames=end - start,
                            decode_chunk_size=8,
                            motion_bucket_id=127,
                            fps=8,
                            noise_aug_strength=0.02,
                            min_guidance_scale=1.5,
                            max_guidance_scale=1.5,
                            generator=self.generator,
                        ).frames[0]

                        pred_amodal_rgb = [np.array(img) for img in pred_amodal_rgb]
                        pred_amodal_rgb = np.array(pred_amodal_rgb).astype("uint8")
                        pred_amodal_rgb_save = np.array(
                            [
                                cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)
                                for frame in pred_amodal_rgb
                            ]
                        )
                        frame_index = start
                        for image_rgb in pred_amodal_rgb_save:
                            cv2.imwrite(
                                os.path.join(completion_image_path, f"{frame_index:08d}.jpg"),
                                cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                            )
                            frame_index += 1
                else:
                    for obj_id in self.RUNTIME["out_obj_ids"]:
                        occ_dict[obj_id] = [1] * len(batch_masks)

                mask_outputs, id_batch, empty_frame_list = process_image_with_mask(
                    self.sam3_3d_body_model,
                    batch_images,
                    batch_masks,
                    idx_path,
                    idx_dict,
                    mhr_shape_scale_dict,
                    occ_dict,
                )

                num_empty_ids = 0
                for frame_id in range(len(batch_images)):
                    image_path = batch_images[frame_id]
                    if frame_id in empty_frame_list:
                        mask_output = None
                        id_current = None
                        num_empty_ids += 1
                    else:
                        mask_output = mask_outputs[frame_id - num_empty_ids]
                        id_current = id_batch[frame_id - num_empty_ids]
                    image_bgr = cv2.imread(image_path)
                    render_all = visualize_sample_together(
                        image_bgr,
                        mask_output,
                        self.sam3_3d_body_model.faces,
                        id_current,
                    )
                    cv2.imwrite(
                        f"{self.OUTPUT_DIR}/rendered_frames/{os.path.basename(image_path)[:-4]}.jpg",
                        render_all.astype(np.uint8),
                    )

                    rendered_individual = visualize_sample(
                        image_bgr,
                        mask_output,
                        self.sam3_3d_body_model.faces,
                        id_current,
                    )
                    for render_index, rendered_image in enumerate(rendered_individual):
                        cv2.imwrite(
                            f"{self.OUTPUT_DIR}/rendered_frames_individual/{render_index + 1}/{os.path.basename(image_path)[:-4]}_{render_index + 1}.jpg",
                            rendered_image.astype(np.uint8),
                        )
                    self._write_openpose_frame(image_path, mask_output, id_current)
                    save_mesh_results(
                        outputs=mask_output,
                        faces=self.sam3_3d_body_model.faces,
                        save_dir=f"{self.OUTPUT_DIR}/mesh_4d_individual",
                        focal_dir=f"{self.OUTPUT_DIR}/focal_4d_individual",
                        image_path=image_path,
                        id_current=id_current,
                    )

            out_4d_path = os.path.join(self.OUTPUT_DIR, f"4d_{time.time():.0f}.mp4")
            jpg_folder_to_mp4(f"{self.OUTPUT_DIR}/rendered_frames", out_4d_path)
            return out_4d_path

    return OfflineAppExport


def prepare_initial_tracking(predictor, base_module, input_video):
    if os.path.isfile(input_video) and input_video.lower().endswith(".mp4"):
        first_frame = base_module.read_frame_at(input_video, 0)
        if first_frame is None:
            raise RuntimeError(f"Unable to read the first frame from video: {input_video}")
        width, height = first_frame.size
        outputs = []
        starting_frame_idx = 0
        for frame_idx in range(0, 100):
            frame_pil = base_module.read_frame_at(input_video, frame_idx)
            if frame_pil is None:
                break
            outputs = predictor.sam3_3d_body_model.process_one_image(np.array(frame_pil), bbox_thr=0.6)
            if outputs:
                starting_frame_idx = frame_idx
                break
        if not outputs:
            raise RuntimeError(f"No humans detected in the initial video search window: {input_video}")

        inference_state = predictor.predictor.init_state(video_path=input_video)
        predictor.predictor.clear_all_points_in_video(inference_state)
        predictor.RUNTIME["inference_state"] = inference_state
        predictor.RUNTIME["out_obj_ids"] = []

        for obj_id, output in enumerate(outputs):
            xmin, ymin, xmax, ymax = output["bbox"]
            rel_box = np.array([[xmin / width, ymin / height, xmax / width, ymax / height]], dtype=np.float32)
            _, predictor.RUNTIME["out_obj_ids"], _, _ = predictor.predictor.add_new_points_or_box(
                inference_state=predictor.RUNTIME["inference_state"],
                frame_idx=starting_frame_idx,
                obj_id=obj_id + 1,
                box=rel_box,
            )
        return

    image_list = list_input_images(input_video)
    if not image_list:
        raise ValueError(f"No input images found in directory: {input_video}")

    image = Image.open(image_list[0]).convert("RGB")
    width, height = image.size
    outputs = []
    starting_frame_idx = 0
    for image_path in image_list:
        outputs = predictor.sam3_3d_body_model.process_one_image(image_path, bbox_thr=0.6)
        if outputs:
            break
        starting_frame_idx += 1
    if not outputs:
        raise RuntimeError(f"No humans detected in the input frame directory: {input_video}")

    inference_state = predictor.predictor.init_state(video_path=image_list)
    predictor.predictor.clear_all_points_in_video(inference_state)
    predictor.RUNTIME["inference_state"] = inference_state
    predictor.RUNTIME["out_obj_ids"] = []

    for obj_id, output in enumerate(outputs):
        xmin, ymin, xmax, ymax = output["bbox"]
        rel_box = np.array([[xmin / width, ymin / height, xmax / width, ymax / height]], dtype=np.float32)
        _, predictor.RUNTIME["out_obj_ids"], _, _ = predictor.predictor.add_new_points_or_box(
            inference_state=predictor.RUNTIME["inference_state"],
            frame_idx=starting_frame_idx,
            obj_id=obj_id + 1,
            box=rel_box,
        )


def inference(args):
    base_module = load_base_offline_module()
    app_class = build_export_app_class(base_module)
    predictor = app_class()
    if args.output_dir is not None:
        predictor.OUTPUT_DIR = args.output_dir
        os.makedirs(predictor.OUTPUT_DIR, exist_ok=True)

    prepare_initial_tracking(predictor, base_module, args.input_video)
    run_export_pipeline(
        predictor,
        args.input_video,
        fps=resolve_mask_video_fps(args.input_video, args.mask_video_fps),
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_input_path(args.input_video)
    inference(args)


if __name__ == "__main__":
    main()
