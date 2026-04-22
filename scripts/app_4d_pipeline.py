import glob
import os
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from scripts.offline_completion_indexing import build_completion_window_from_ious
from scripts.completion_safety import build_completion_slice, resolve_completion_batch_size, resolve_decode_chunk_size

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "models", "sam_3d_body"))
sys.path.append(os.path.join(ROOT, "models", "diffusion_vas"))

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional in lightweight test envs
    torch = None


IMAGE_PATTERNS = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff", "*.webp"]


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


def _list_input_images(input_dir):
    image_paths = []
    for pattern in IMAGE_PATTERNS:
        image_paths.extend(glob.glob(os.path.join(input_dir, pattern)))
    return sorted(image_paths)


def _resolution_key(resolution):
    return tuple(int(value) for value in resolution)


def _load_modal_pixels_cached(cache, masks_dir, resolution, obj_id):
    cache_key = ("mask", os.path.abspath(masks_dir), _resolution_key(resolution), int(obj_id))
    if cache_key not in cache:
        cache[cache_key] = load_and_transform_masks(
            masks_dir,
            resolution=resolution,
            obj_id=obj_id,
        )
    return cache[cache_key]


def _load_rgb_pixels_cached(cache, images_dir, resolution):
    cache_key = ("rgb", os.path.abspath(images_dir), _resolution_key(resolution))
    if cache_key not in cache:
        cache[cache_key] = load_and_transform_rgbs(images_dir, resolution=resolution)
    return cache[cache_key]


def _load_indexed_mask_cached(cache, mask_path):
    cache_key = os.path.abspath(mask_path)
    if cache_key not in cache:
        with Image.open(cache_key) as mask_image:
            cache[cache_key] = np.array(mask_image.convert("P"))
    return cache[cache_key]


def _load_bgr_image_cached(cache, image_path):
    cache_key = os.path.abspath(image_path)
    if cache_key not in cache:
        image_bgr = cv2.imread(cache_key)
        if image_bgr is None:
            raise FileNotFoundError(f"unable to read image: {image_path}")
        cache[cache_key] = image_bgr
    return cache[cache_key].copy()


def _format_cam_int_progress_postfix(stats):
    misses = int(stats.get("misses", 0))
    hits = int(stats.get("hits", 0))
    if misses <= 0 and hits <= 0:
        return None

    postfix = f"cam_int={misses}m/{hits}h"
    last_miss_frame = stats.get("last_miss_frame")
    if last_miss_frame:
        postfix = f"{postfix} miss={last_miss_frame}"
    return postfix


def build_4d_context(
    *,
    input_dir,
    output_dir,
    runtime,
    sam3_3d_body_model,
    pipeline_mask,
    pipeline_rgb,
    depth_model,
    predictor,
    generator,
    frame_writer=None,
):
    return SimpleNamespace(
        input_dir=input_dir,
        output_dir=output_dir,
        runtime=runtime,
        sam3_3d_body_model=sam3_3d_body_model,
        pipeline_mask=pipeline_mask,
        pipeline_rgb=pipeline_rgb,
        depth_model=depth_model,
        predictor=predictor,
        generator=generator,
        frame_writer=frame_writer,
    )


def run_4d_pipeline_from_context(context):
    print("[DEBUG] 4D Generation button clicked.")

    input_images_dir = os.path.join(context.input_dir, "images")
    input_masks_dir = os.path.join(context.input_dir, "masks")
    images_list = _list_input_images(input_images_dir)
    masks_list = _list_input_images(input_masks_dir)
    if not images_list:
        raise FileNotFoundError(f"missing exported images under {input_images_dir}")
    if not masks_list:
        raise FileNotFoundError(f"missing exported masks under {input_masks_dir}")

    output_dir = context.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_rendered_frames = bool(context.runtime.get("save_rendered_frames", True))
    save_rendered_frames_individual = bool(context.runtime.get("save_rendered_frames_individual", True))
    save_mesh = bool(context.runtime.get("save_mesh_4d_individual", True))
    save_focal = bool(context.runtime.get("save_focal_4d_individual", True))

    if save_rendered_frames:
        os.makedirs(os.path.join(output_dir, "rendered_frames"), exist_ok=True)
    for obj_id in context.runtime["out_obj_ids"]:
        if save_mesh:
            os.makedirs(os.path.join(output_dir, "mesh_4d_individual", str(obj_id)), exist_ok=True)
        if save_focal:
            os.makedirs(os.path.join(output_dir, "focal_4d_individual", str(obj_id)), exist_ok=True)
        if save_rendered_frames_individual:
            os.makedirs(os.path.join(output_dir, "rendered_frames_individual", str(obj_id)), exist_ok=True)

    hmr_batch_size = context.runtime["batch_size"]
    completion_batch_size = resolve_completion_batch_size(context.runtime.get("completion_batch_size", 1))
    batch_size = completion_batch_size if context.pipeline_mask is not None else hmr_batch_size
    n = len(images_list)
    runtime_cache = {
        "modal_pixels": {},
        "rgb_pixels": {},
        "mask_frames": {},
        "image_frames": {},
        "cam_int": {},
        "cam_int_stats": {"hits": 0, "misses": 0},
    }

    pred_res = context.runtime["detection_resolution"]
    pred_res_hi = context.runtime["completion_resolution"]
    modal_pixels_by_obj = {}
    if context.pipeline_mask is not None:
        runtime_utils = _load_runtime_utils()
        davis_palette = runtime_utils["DAVIS_PALETTE"]
        bbox_from_mask = runtime_utils["bbox_from_mask"]
        is_skinny_mask = runtime_utils["is_skinny_mask"]
        is_super_long_or_wide = runtime_utils["is_super_long_or_wide"]
        keep_largest_component = runtime_utils["keep_largest_component"]
        resize_mask_with_unique_label = runtime_utils["resize_mask_with_unique_label"]

        for obj_id in context.runtime["out_obj_ids"]:
            modal_pixels, _ = _load_modal_pixels_cached(
                runtime_cache["modal_pixels"],
                input_masks_dir,
                pred_res,
                obj_id,
            )
            modal_pixels_by_obj[obj_id] = modal_pixels
        rgb_pixels, _, _ = _load_rgb_pixels_cached(
            runtime_cache["rgb_pixels"],
            input_images_dir,
            pred_res,
        )
        depth_pixels = rgb_to_depth(rgb_pixels, context.depth_model)

    mhr_shape_scale_dict = {}
    obj_ratio_dict = {}

    progress = tqdm(range(0, n, batch_size))
    for i in progress:
        batch_images = images_list[i : i + batch_size]
        batch_masks = masks_list[i : i + batch_size]
        batch_mask_frames = [
            _load_indexed_mask_cached(runtime_cache["mask_frames"], mask_path)
            for mask_path in batch_masks
        ]
        height, width = batch_mask_frames[0].shape

        idx_dict = {}
        idx_path = {}
        occ_dict = {}
        completion_cache = {}
        persist_completion_artifacts = bool(context.runtime.get("save_completion_artifacts", False))
        if len(modal_pixels_by_obj) > 0:
            print("detect occlusions ...")
            pred_amodal_masks_dict = {}
            for obj_id in context.runtime["out_obj_ids"]:
                modal_pixels = modal_pixels_by_obj[obj_id]
                decode_chunk_size = resolve_decode_chunk_size(
                    context.runtime.get("completion_decode_chunk_size", 8),
                    num_frames=modal_pixels[:, i : i + batch_size, :, :, :].shape[1],
                )
                pred_amodal_masks = context.pipeline_mask(
                    modal_pixels[:, i : i + batch_size, :, :, :],
                    depth_pixels[:, i : i + batch_size, :, :, :],
                    height=pred_res[0],
                    width=pred_res[1],
                    num_frames=modal_pixels[:, i : i + batch_size, :, :, :].shape[1],
                    decode_chunk_size=decode_chunk_size,
                    motion_bucket_id=127,
                    fps=8,
                    noise_aug_strength=0.02,
                    min_guidance_scale=1.5,
                    max_guidance_scale=1.5,
                    generator=context.generator,
                ).frames[0]

                pred_amodal_masks_com = [
                    np.array(image.resize((pred_res_hi[1], pred_res_hi[0]))) for image in pred_amodal_masks
                ]
                pred_amodal_masks_com = np.array(pred_amodal_masks_com).astype("uint8")
                pred_amodal_masks_com = (pred_amodal_masks_com.sum(axis=-1) > 600).astype("uint8")
                pred_amodal_masks_com = [keep_largest_component(mask_current) for mask_current in pred_amodal_masks_com]

                pred_amodal_masks = [np.array(image.resize((width, height))) for image in pred_amodal_masks]
                pred_amodal_masks = np.array(pred_amodal_masks).astype("uint8")
                pred_amodal_masks = (pred_amodal_masks.sum(axis=-1) > 600).astype("uint8")
                pred_amodal_masks = [keep_largest_component(mask_current) for mask_current in pred_amodal_masks]

                masks = [(mask_frame == obj_id).astype("uint8") for mask_frame in batch_mask_frames]
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

                for pred_index, _ in enumerate(pred_amodal_masks_com):
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
                occ_dict[obj_id], completion_window = build_completion_window_from_ious(
                    ious,
                    padding=0,
                    iou_threshold=0.7,
                )

                if completion_window is not None:
                    start, end = build_completion_slice(
                        first_occ_idx=completion_window[0],
                        last_occ_idx=completion_window[1] - 1,
                        total_frames=modal_pixels[:, i : i + batch_size, :, :, :].shape[1],
                        pad_before=2,
                        pad_after=2,
                        max_occ_len=context.runtime.get("max_occ_len", 0),
                    )
                    if end <= start:
                        continue
                    idx_dict[obj_id] = (start, end)
                    completion_cache[obj_id] = {"images": {}, "masks": {}}
                    if persist_completion_artifacts:
                        batch_tag = f"batch_{i:08d}"
                        completion_image_path = os.path.join(output_dir, "completion", str(obj_id), batch_tag, "images")
                        completion_masks_path = os.path.join(output_dir, "completion", str(obj_id), batch_tag, "masks")
                        os.makedirs(completion_image_path, exist_ok=True)
                        os.makedirs(completion_masks_path, exist_ok=True)
                        idx_path[obj_id] = {"images": completion_image_path, "masks": completion_masks_path}
                    for frame_index in range(start, end):
                        mask_idx = pred_amodal_masks[frame_index].copy()
                        mask_idx[mask_idx > 0] = obj_id
                        completion_cache[obj_id]["masks"][frame_index] = mask_idx.astype(np.uint8)
                        if persist_completion_artifacts:
                            mask_image = Image.fromarray(mask_idx).convert("P")
                            mask_image.putpalette(davis_palette)
                            mask_image.save(os.path.join(completion_masks_path, f"{frame_index:08d}.png"))

            for obj_id, (start, end) in idx_dict.items():
                modal_pixels_current, ori_shape = _load_modal_pixels_cached(
                    runtime_cache["modal_pixels"],
                    input_masks_dir,
                    pred_res_hi,
                    obj_id,
                )
                rgb_pixels_current, _, _ = _load_rgb_pixels_cached(
                    runtime_cache["rgb_pixels"],
                    input_images_dir,
                    pred_res_hi,
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
                decode_chunk_size = resolve_decode_chunk_size(
                    context.runtime.get("completion_decode_chunk_size", 8),
                    num_frames=max(1, end - start),
                )
                pred_amodal_rgb = context.pipeline_rgb(
                    modal_rgb_pixels,
                    pred_amodal_masks_tensor,
                    height=pred_res_hi[0],
                    width=pred_res_hi[1],
                    num_frames=end - start,
                    decode_chunk_size=decode_chunk_size,
                    motion_bucket_id=127,
                    fps=8,
                    noise_aug_strength=0.02,
                    min_guidance_scale=1.5,
                    max_guidance_scale=1.5,
                    generator=context.generator,
                ).frames[0]

                pred_amodal_rgb = [np.array(image) for image in pred_amodal_rgb]
                pred_amodal_rgb = np.array(pred_amodal_rgb).astype("uint8")
                pred_amodal_rgb_save = np.array(
                    [
                        cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)
                        for frame in pred_amodal_rgb
                    ]
                )
                for frame_index, image_rgb in zip(range(start, end), pred_amodal_rgb_save):
                    completion_cache[obj_id]["images"][frame_index] = image_rgb.copy()
                    if persist_completion_artifacts:
                        completion_image_path = idx_path[obj_id]["images"]
                        cv2.imwrite(
                            os.path.join(completion_image_path, f"{frame_index:08d}.jpg"),
                            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                        )
        else:
            for obj_id in context.runtime["out_obj_ids"]:
                occ_dict[obj_id] = [1] * len(batch_masks)

        mask_outputs, id_batch, empty_frame_list = process_image_with_mask(
            context.sam3_3d_body_model,
            batch_images,
            batch_masks,
            idx_path,
            idx_dict,
            mhr_shape_scale_dict,
            occ_dict,
            completion_cache=completion_cache,
            mask_frames=batch_mask_frames,
            image_cache=runtime_cache["image_frames"],
            cam_int_cache=runtime_cache["cam_int"],
            cam_int_cache_stats=runtime_cache["cam_int_stats"],
        )
        progress_postfix = _format_cam_int_progress_postfix(runtime_cache["cam_int_stats"])
        if progress_postfix:
            progress.set_postfix_str(progress_postfix)

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
            image_bgr = None
            if save_rendered_frames or save_rendered_frames_individual:
                image_bgr = _load_bgr_image_cached(runtime_cache["image_frames"], image_path)

            if save_rendered_frames:
                render_all = visualize_sample_together(
                    image_bgr,
                    mask_output,
                    context.sam3_3d_body_model.faces,
                    id_current,
                )
                cv2.imwrite(
                    os.path.join(output_dir, "rendered_frames", f"{os.path.basename(image_path)[:-4]}.jpg"),
                    render_all.astype(np.uint8),
                )

            if save_rendered_frames_individual:
                rendered_individual = visualize_sample(
                    image_bgr,
                    mask_output,
                    context.sam3_3d_body_model.faces,
                    id_current,
                )
                for render_index, rendered_image in enumerate(rendered_individual):
                    track_id = int(id_current[render_index]) if id_current and render_index < len(id_current) else render_index + 1
                    cv2.imwrite(
                        os.path.join(
                            output_dir,
                            "rendered_frames_individual",
                            str(track_id),
                            f"{os.path.basename(image_path)[:-4]}_{track_id}.jpg",
                        ),
                        rendered_image.astype(np.uint8),
                    )

            frame_writer = getattr(context, "frame_writer", None)
            if callable(frame_writer):
                frame_writer(image_path, mask_output, id_current)

            if save_mesh or save_focal:
                save_mesh_results(
                    outputs=mask_output,
                    faces=context.sam3_3d_body_model.faces,
                    save_dir=os.path.join(output_dir, "mesh_4d_individual"),
                    focal_dir=os.path.join(output_dir, "focal_4d_individual"),
                    image_path=image_path,
                    id_current=id_current,
                    save_mesh=save_mesh,
                    save_focal=save_focal,
                )

    cam_int_hits = int(runtime_cache["cam_int_stats"].get("hits", 0))
    cam_int_misses = int(runtime_cache["cam_int_stats"].get("misses", 0))
    if cam_int_hits > 0 or cam_int_misses > 0:
        print(f"cam_int cache: {cam_int_misses} misses, {cam_int_hits} hits")

    if not save_rendered_frames:
        return None

    out_4d_path = os.path.join(output_dir, f"4d_{time.time():.0f}.mp4")
    jpg_folder_to_mp4(
        os.path.join(output_dir, "rendered_frames"),
        out_4d_path,
        fps=context.runtime.get("video_fps", 25),
    )
    return out_4d_path
