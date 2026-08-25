#!/usr/bin/env python3
"""
ArUco Marker Generator Utility
Generates printable ArUco markers for attaching to wrist rigs / teleoperation gloves.

Usage:
    python3 generate_marker.py --id 0 --dict DICT_4X4_50 --size 500 --output marker_0.png
"""

import argparse
import sys
import cv2
import numpy as np

# Dictionary map for OpenCV ArUco
ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11 if hasattr(cv2.aruco, "DICT_APRILTAG_36h11") else None,
}


def get_aruco_dictionary(dict_name: str):
    if dict_name not in ARUCO_DICT_MAP or ARUCO_DICT_MAP[dict_name] is None:
        raise ValueError(f"Unknown or unsupported ArUco dictionary: {dict_name}. Available: {list(ARUCO_DICT_MAP.keys())}")
    
    dict_id = ARUCO_DICT_MAP[dict_name]
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    else:
        return cv2.aruco.Dictionary_get(dict_id)


def generate_marker(
    marker_id: int = 0,
    dict_name: str = "DICT_4X4_50",
    pixel_size: int = 600,
    border_bits: int = 1,
    add_label: bool = True,
    output_path: str = "marker.png"
):
    dictionary = get_aruco_dictionary(dict_name)
    
    # Generate marker image (OpenCV 4.7+ uses generateImageMarker, older uses drawMarker)
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, pixel_size, borderBits=border_bits)
    else:
        marker_img = np.zeros((pixel_size, pixel_size), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, marker_id, pixel_size, marker_img, borderBits=border_bits)

    # Add white margin and text label for easy printing
    margin = 80 if add_label else 40
    canvas_h = pixel_size + margin * 2 + (60 if add_label else 0)
    canvas_w = pixel_size + margin * 2
    canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255

    canvas[margin:margin + pixel_size, margin:margin + pixel_size] = marker_img

    if add_label:
        label = f"ArUco {dict_name} | ID: {marker_id}"
        sub_label = f"Recommended Size: 50mm x 50mm (Cut along outer border)"
        cv2.putText(canvas, label, (margin, pixel_size + margin + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,), 2, cv2.LINE_AA)
        cv2.putText(canvas, sub_label, (margin, pixel_size + margin + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,), 1, cv2.LINE_AA)

        # Draw a thin light-gray cutting guideline around the marker margin
        cv2.rectangle(canvas, (margin - 10, margin - 10), (margin + pixel_size + 10, margin + pixel_size + 10), (200,), 1)

    cv2.imwrite(output_path, canvas)
    print(f"[OK] Generated ArUco marker: {output_path}")
    print(f"     Dictionary: {dict_name}, ID: {marker_id}, Resolution: {canvas_w}x{canvas_h}")


def main():
    parser = argparse.ArgumentParser(description="Generate Printable ArUco Marker for Wrist Tracking")
    parser.add_argument("--id", type=int, default=0, help="Marker ID (default: 0)")
    parser.add_argument("--dict", type=str, default="DICT_4X4_50", choices=list(ARUCO_DICT_MAP.keys()),
                        help="ArUco dictionary name (default: DICT_4X4_50)")
    parser.add_argument("--size", type=int, default=600, help="Marker image size in pixels (default: 600)")
    parser.add_argument("--border", type=int, default=1, help="Number of border bits (default: 1)")
    parser.add_argument("--no-label", action="store_true", help="Do not add text label and cutting guidelines")
    parser.add_argument("--output", type=str, default="marker_0.png", help="Output file path (default: marker_0.png)")

    args = parser.parse_args()
    generate_marker(
        marker_id=args.id,
        dict_name=args.dict,
        pixel_size=args.size,
        border_bits=args.border,
        add_label=not args.no_label,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
