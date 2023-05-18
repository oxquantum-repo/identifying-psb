import numpy as np
from typing import List


def convert_voltage_coord_to_px_coordinates(
    coord: List[float], d_file: dict
) -> np.ndarray:
    """
    Converts voltage coordinates to pixel coordinates.

    Parameters:
    coord (List[float]): Voltage coordinates to be converted.
    d_file (dict): Dictionary containing the relevant data for conversion (a measurement file that
    these experiments created).

    Returns:
    np.ndarray: Pixel coordinates after conversion.
    """
    bottom_x = d_file["chan0"]["vals"][1][0]
    res_x = np.abs(bottom_x - d_file["chan0"]["vals"][1][1])
    px_x = (coord[0] - bottom_x) / res_x
    bottom_y = d_file["chan0"]["vals"][0][0]
    res_y = np.abs(bottom_y - d_file["chan0"]["vals"][0][1])
    px_y = (coord[1] - bottom_y) / res_y

    return np.array([px_x, px_y], dtype=int)


def convert_vboxes_to_pxboxes(boxes: List[List[float]], d_file: dict) -> np.ndarray:
    """
    Converts boxes in voltage space to pixel space.

    Parameters:
    boxes (List[List[float]]): Boxes in voltage space to be converted.
    d_file (dict): Dictionary containing the relevant data for conversion (a measurement file that
    these experiments created).

    Returns:
    np.ndarray: Boxes in pixel space after conversion.
    """
    px_boxes = []
    for box in boxes:
        px_boxes.append([])
        for coord in box:
            coord_px = convert_voltage_coord_to_px_coordinates(coord, d_file)
            px_boxes[-1].append(coord_px)
    return np.array(px_boxes)


def convert_px_coord_to_voltage_coordinates(
    coord: List[int], d_file: dict
) -> List[float]:
    """
    Converts pixel coordinates to voltage coordinates.

    Parameters:
    coord (List[int]): Pixel coordinates to be converted.
    d_file (dict): Dictionary containing the relevant data for conversion.

    Returns:
    List[float]: Voltage coordinates after conversion.
    """
    voltage_x = d_file["chan0"]["vals"][1][coord[0]]
    voltage_y = d_file["chan0"]["vals"][0][coord[1]]
    return [voltage_x, voltage_y]


def convert_pxboxes_to_vbox(boxes: List[List[int]], d_file: dict) -> np.ndarray:
    """
    Converts boxes in pixel space to voltage space.

    Parameters:
    boxes (List[List[int]]): Boxes in pixel space to be converted.
    d_file (dict): Dictionary containing the relevant data for conversion.

    Returns:
    np.ndarray: Boxes in voltage space after conversion.
    """
    v_boxes = []
    for box in boxes:
        v_boxes.append([])
        for coord in box:
            coord_v = convert_px_coord_to_voltage_coordinates(coord, d_file)
            v_boxes[-1].append(coord_v)
    return np.array(v_boxes)
