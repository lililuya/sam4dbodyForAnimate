# Copyright (c) Meta Platforms, Inc. and affiliates.
import cv2
import numpy as np

from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from utils.painter import color_list

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)


def _normalize_render_mode(render_mode) -> str:
    normalized = str(render_mode or "mesh").strip().lower()
    if normalized in {"mesh", "mesh_color", "color", "default"}:
        return "mesh"
    if normalized in {"normal", "norm", "normals"}:
        return "normal"
    raise ValueError(f"unsupported render_mode: {render_mode}")


def _coerce_scene_bg_color(scene_bg_color) -> tuple[float, float, float]:
    if isinstance(scene_bg_color, str):
        normalized = scene_bg_color.strip().lower()
        if normalized == "black":
            return (0.0, 0.0, 0.0)
        if normalized == "white":
            return (1.0, 1.0, 1.0)
        raise ValueError(f"unsupported scene_bg_color string: {scene_bg_color}")

    values = tuple(float(value) for value in scene_bg_color)
    if len(values) != 3:
        raise ValueError("scene_bg_color must contain exactly three values")
    if max(values) > 1.0:
        array = np.clip(np.asarray(values, dtype=np.float32) / 255.0, 0.0, 1.0)
        return float(array[0]), float(array[1]), float(array[2])
    return tuple(float(np.clip(value, 0.0, 1.0)) for value in values)


def _build_scene_canvas(img_cv2, scene_bg_color: tuple[float, float, float]) -> np.ndarray:
    background_value = int(round(max(scene_bg_color) * 255.0))
    return np.ones_like(img_cv2) * background_value


def _render_person(renderer, person_output, canvas, mesh_color, render_mode: str, scene_bg_color):
    if render_mode == "normal":
        return (
            renderer.render_normals(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                canvas.copy(),
                scene_bg_color=scene_bg_color,
            )
            * 255
        )

    return (
        renderer(
            person_output["pred_vertices"],
            person_output["pred_cam_t"],
            canvas.copy(),
            mesh_base_color=mesh_color,
            scene_bg_color=scene_bg_color,
        )
        * 255
    )


def visualize_sample(img_cv2, outputs, faces, id_current, render_mode="mesh", scene_bg_color=(1, 1, 1)):
    render_mode = _normalize_render_mode(render_mode)
    scene_bg_color = _coerce_scene_bg_color(scene_bg_color)
    img_mesh = _build_scene_canvas(img_cv2, scene_bg_color)

    if outputs is None:
        return img_mesh

    rend_img = []
    for pid, person_output in enumerate(outputs):
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
        rendered = _render_person(
            renderer,
            person_output,
            img_mesh,
            color_list[id_current[pid] + 4],
            render_mode,
            scene_bg_color,
        )
        rend_img.append(rendered)

    return rend_img


def visualize_sample_together(img_cv2, outputs, faces, id_current, render_mode="mesh", scene_bg_color=(1, 1, 1)):
    render_mode = _normalize_render_mode(render_mode)
    scene_bg_color = _coerce_scene_bg_color(scene_bg_color)
    img_mesh = _build_scene_canvas(img_cv2, scene_bg_color)

    if outputs is None:
        return img_mesh

    try:
        all_depths = np.stack([tmp["pred_cam_t"] for tmp in outputs], axis=0)[:, 2]
    except Exception:
        return img_mesh
    outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

    id_sorted = np.argsort(-all_depths)

    all_pred_vertices = []
    all_faces = []
    all_color = []
    for pid, person_output in enumerate(outputs_sorted):
        all_pred_vertices.append(person_output["pred_vertices"] + person_output["pred_cam_t"])
        all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
        all_color.append(color_list[id_current[id_sorted[pid]] + 4])
    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    fake_pred_cam_t = (np.max(all_pred_vertices[-2 * 18439 :], axis=0) + np.min(all_pred_vertices[-2 * 18439 :], axis=0)) / 2
    all_pred_vertices = all_pred_vertices - fake_pred_cam_t

    renderer = Renderer(focal_length=person_output["focal_length"], faces=all_faces)
    if render_mode == "normal":
        return (
            renderer.render_normals(
                all_pred_vertices,
                fake_pred_cam_t,
                img_mesh,
                scene_bg_color=scene_bg_color,
            )
            * 255
        )

    return (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            img_mesh,
            mesh_base_color=all_color,
            scene_bg_color=scene_bg_color,
        )
        * 255
    )
