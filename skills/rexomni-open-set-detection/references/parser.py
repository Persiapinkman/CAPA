#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Output parsing utilities for Rex Omni
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple


def parse_prediction(
    text: str, w: int, h: int, task_type: str = "detection"
) -> Dict[str, List]:
    """
    Parse model output text to extract category-wise predictions.

    Args:
        text: Model output text
        w: Image width
        h: Image height
        task_type: Type of task ("detection", "keypoint", etc.)

    Returns:
        Dictionary with category as key and list of predictions as value
    """
    if task_type == "keypoint":
        return parse_keypoint_prediction(text, w, h)
    else:
        return parse_standard_prediction(text, w, h)


def parse_standard_prediction(text: str, w: int, h: int) -> Dict[str, List]:
    """
    Parse standard prediction output for detection, pointing, etc.

    Input format example:
    "<|object_ref_start|>person<|object_ref_end|><|box_start|><0><35><980><987>, <646><0><999><940><|box_end|>"

    Returns:
    {
        'category1': [{"type": "box/point/polygon", "coords": [...]}],
        'category2': [{"type": "box/point/polygon", "coords": [...]}],
        ...
    }
    """
    result = {}

    # Remove the end marker if present
    text = text.split("<|im_end|>")[0]
    if not text.endswith("<|box_end|>"):
        text = text + "<|box_end|>"

    # Use regex to find all object references and coordinate pairs
    pattern = r"<\|object_ref_start\|>\s*([^<]+?)\s*<\|object_ref_end\|>\s*<\|box_start\|>(.*?)<\|box_end\|>"
    matches = re.findall(pattern, text)

    for category, coords_text in matches:
        category = category.strip()

        # Find all coordinate tokens in the format <{number}>
        coord_pattern = r"<(\d+)>"
        coord_matches = re.findall(coord_pattern, coords_text)

        annotations = []
        # Split by comma to handle multiple coordinates for the same phrase
        coord_strings = coords_text.split(",")

        for coord_str in coord_strings:
            coord_nums = re.findall(coord_pattern, coord_str.strip())

            if len(coord_nums) == 2:
                # Point: <{x}><{y}>
                try:
                    x_bin = int(coord_nums[0])
                    y_bin = int(coord_nums[1])

                    # Convert from bins [0, 999] to absolute coordinates
                    x = (x_bin / 999.0) * w
                    y = (y_bin / 999.0) * h

                    annotations.append({"type": "point", "coords": [x, y]})
                except (ValueError, IndexError) as e:
                    print(f"Error parsing point coordinates: {e}")
                    continue

            elif len(coord_nums) == 4:
                # Bounding box: <{x0}><{y0}><{x1}><{y1}>
                try:
                    x0_bin = int(coord_nums[0])
                    y0_bin = int(coord_nums[1])
                    x1_bin = int(coord_nums[2])
                    y1_bin = int(coord_nums[3])

                    # Convert from bins [0, 999] to absolute coordinates
                    x0 = (x0_bin / 999.0) * w
                    y0 = (y0_bin / 999.0) * h
                    x1 = (x1_bin / 999.0) * w
                    y1 = (y1_bin / 999.0) * h

                    annotations.append({"type": "box", "coords": [x0, y0, x1, y1]})
                except (ValueError, IndexError) as e:
                    print(f"Error parsing box coordinates: {e}")
                    continue

            elif len(coord_nums) > 4 and len(coord_nums) % 2 == 0:
                # Polygon: <{x0}><{y0}><{x1}><{y1}>...
                try:
                    polygon_coords = []
                    for i in range(0, len(coord_nums), 2):
                        x_bin = int(coord_nums[i])
                        y_bin = int(coord_nums[i + 1])

                        # Convert from bins [0, 999] to absolute coordinates
                        x = (x_bin / 999.0) * w
                        y = (y_bin / 999.0) * h

                        polygon_coords.append([x, y])

                    annotations.append({"type": "polygon", "coords": polygon_coords})
                except (ValueError, IndexError) as e:
                    print(f"Error parsing polygon coordinates: {e}")
                    continue

        if category not in result:
            result[category] = []
        result[category].extend(annotations)

    return result


def parse_keypoint_prediction(text: str, w: int, h: int) -> Dict[str, List]:
    """
    Parse keypoint task JSON output to extract bbox and keypoints.
    """
    json_pattern = r"```json\s*(.*?)\s*```"
    json_matches = re.findall(json_pattern, text, re.DOTALL)

    if not json_matches:
        try:
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx : end_idx + 1]
            else:
                return {}
        except Exception:
            return {}
    else:
        json_str = json_matches[0]

    try:
        keypoint_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing keypoint JSON: {e}")
        return {}

    result = {}
    coord_pattern = r"<(\d+)>"

    for instance_id, instance_data in keypoint_data.items():
        if "bbox" not in instance_data or "keypoints" not in instance_data:
            continue

        bbox = instance_data["bbox"]
        keypoints = instance_data["keypoints"]

        if isinstance(bbox, str) and bbox.strip():
            coord_matches = re.findall(coord_pattern, bbox)
            if len(coord_matches) == 4:
                try:
                    x0_bin, y0_bin, x1_bin, y1_bin = [int(m) for m in coord_matches]
                    x0 = (x0_bin / 999.0) * w
                    y0 = (y0_bin / 999.0) * h
                    x1 = (x1_bin / 999.0) * w
                    y1 = (y1_bin / 999.0) * h
                    converted_bbox = [x0, y0, x1, y1]
                except (ValueError, IndexError):
                    continue
            else:
                continue
        else:
            continue

        converted_keypoints = {}
        for kp_name, kp_coords in keypoints.items():
            if kp_coords == "unvisible" or kp_coords is None:
                converted_keypoints[kp_name] = "unvisible"
            elif isinstance(kp_coords, str) and kp_coords.strip():
                coord_matches = re.findall(coord_pattern, kp_coords)
                if len(coord_matches) == 2:
                    try:
                        x_bin, y_bin = [int(m) for m in coord_matches]
                        x = (x_bin / 999.0) * w
                        y = (y_bin / 999.0) * h
                        converted_keypoints[kp_name] = [x, y]
                    except (ValueError, IndexError):
                        converted_keypoints[kp_name] = "unvisible"
                else:
                    converted_keypoints[kp_name] = "unvisible"
            else:
                converted_keypoints[kp_name] = "unvisible"

        category = "keypoint_instance"
        if instance_id:
            category_match = re.match(r"^([a-zA-Z_]+)", instance_id)
            if category_match:
                category = category_match.group(1)

        if category not in result:
            result[category] = []
        result[category].append(
            {
                "type": "keypoint",
                "bbox": converted_bbox,
                "keypoints": converted_keypoints,
                "instance_id": instance_id,
            }
        )

    return result
