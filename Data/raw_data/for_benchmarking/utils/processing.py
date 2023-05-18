import pickle
import math
import toml
import pickle
import os
import numpy as np
from igor import binarywave
import matplotlib.pyplot as plt
from skimage.feature import blob_log

import imgaug.augmenters as iaa


from typing import Dict, Tuple, Any, Union, Optional, List


def load(
    data_set_name: str, root: str = "data", verbose: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load the data from a specified dataset. The dataset is organized in sets and conditions.
    Depending on the configuration, the data can be loaded from various formats such as pickle and igorstyle files.

    Parameters:
    data_set_name (str): The name of the dataset to load.
    root (str, optional): The root directory where the dataset is located. Defaults to "data".
    verbose (bool, optional): Whether or not to print out detailed loading information. Defaults to False.

    Returns:
    Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing the loaded dataset as a nested dictionary and the configuration settings.

    Raises:
    NotImplementedError: If the data style specified in the configuration is unknown.
    """
    print("loading", data_set_name)

    data_set = {}

    path_to_data = os.path.join(root, data_set_name)
    path_to_config = os.path.join(path_to_data, "config.toml")
    path_to_saved_labels = os.path.join(path_to_data, "saved_blobs_and_labels.pkl")

    with open(path_to_saved_labels, "rb") as f:
        blobs_and_labels = pickle.load(f)

    configs = toml.load(path_to_config)
    structure = configs["structure"]
    if verbose:
        print("structure: ", structure)

    # getting raw data and labels
    for set_name in structure:
        data_set[set_name] = {}
        for condition in structure[set_name]:
            filename = structure[set_name][condition]
            data_set[set_name][condition] = {}
            path = os.path.join(path_to_data, set_name, filename)
            if verbose:
                print("loading:", path)

            if configs["metainfo"]["data_mode"] == "igorstyle":
                data_set[set_name][condition]["raw"] = binarywave.load(path + ".ibw")
                data_set[set_name][condition]["img"] = data_set[set_name][condition][
                    "raw"
                ]["wave"]["wData"]

                # fudged because of weird measurements
                if (
                    data_set_name == "202010_Basel_FinFET"
                    and set_name == "set2"
                    and condition == "neg_bias_blocked"
                ):
                    data_set[set_name][condition]["img"] = data_set[set_name][
                        condition
                    ]["img"][12:80, 10:]
                    data_set[set_name][condition]["raw"]["wave"]["wData"] = data_set[
                        set_name
                    ][condition]["img"]

            elif configs["metainfo"]["data_mode"] == "pickle_jonas":
                with open(path + ".pkl", "rb") as f:
                    data_set[set_name][condition]["raw"] = pickle.load(f)
                data_set[set_name][condition]["img"] = data_set[set_name][condition][
                    "raw"
                ]["chan0"]["data"]

            else:
                print("data style", configs["metainfo"]["data_mode"], "unkown, abort")
                raise NotImplementedError

            if data_set_name == "202012_Basel_FinFET":
                if condition == "pos_bias_blocked" or condition == "pos_bias_unblocked":
                    path_to_saved_boxes = os.path.join(
                        path_to_data, "boxes_positive_bias_voltage_space.npy"
                    )
                else:
                    path_to_saved_boxes = os.path.join(
                        path_to_data, "boxes_negative_bias_voltage_space.npy"
                    )
                v_boxes = np.load(path_to_saved_boxes)
                boxes = convert_vboxes_to_pxboxes(
                    v_boxes, data_set[set_name][condition]["raw"]
                )
                data_set[set_name][condition]["boxes"] = boxes

            else:
                data_set[set_name][condition]["blobs"] = blobs_and_labels[set_name][
                    condition
                ]["blobs"]
            data_set[set_name][condition]["bias_triangle"] = blobs_and_labels[set_name][
                condition
            ]["bias_triangle"]
            data_set[set_name][condition]["pauli_spin_blockade"] = blobs_and_labels[
                set_name
            ][condition]["pauli_spin_blockade"]

            if configs["metainfo"]["data_mode"] == "igorstyle":
                data_set[set_name][condition]["stepsizes"] = data_set[set_name][
                    condition
                ]["raw"]["wave"]["wave_header"]["sfA"][:2]
                if np.any(data_set[set_name][condition]["stepsizes"] < 0):
                    if verbose:
                        print("correcting negative stepsizes in:", set_name, condition)
                    data_set[set_name][condition]["stepsizes"] = np.abs(
                        data_set[set_name][condition]["stepsizes"]
                    )

            elif configs["metainfo"]["data_mode"][:3] == "dat":
                data_set[set_name][condition]["stepsizes"] = get_stepsizes_from_dat(
                    raw, configs, condition
                )
            elif configs["metainfo"]["data_mode"] == "pickle_jonas":
                vals = data_set[set_name][condition]["raw"]["chan0"]["vals"]

                data_set[set_name][condition]["stepsizes"] = [
                    vals[0][1] - vals[0][0],
                    vals[1][1] - vals[1][0],
                ]
            else:
                print("step size not recognised, use default of 1 mV instead")
                data_set[set_name][condition]["stepsizes"] = [1, 1]

            # print(set_name, condition, 'which_direction_triangles')
            data_set[set_name][condition]["which_direction_triangles"] = configs[
                "metainfo"
            ]["which_direction_triangles"][condition]

            # print(configs['metainfo'].keys())
            if configs["metainfo"]["current_correction"]["bias_reversed"]:
                data_set[set_name][condition]["bias_reversed"] = configs["metainfo"][
                    "current_correction"
                ][set_name]
            else:
                data_set[set_name][condition]["bias_reversed"] = False

    if verbose:
        print("loading", data_set_name, "done")
    return data_set, configs


def get_img_and_stepsize(
    raw: np.ndarray, configs: Dict[str, Union[str, Dict]], condition: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the image and step size data from raw input based on specified configuration and condition.

    Args:
        raw (np.ndarray): Raw data.
        configs (Dict[str, Union[str, Dict]]): Configuration parameters.
        condition (str): Specific condition for processing the raw data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the processed image and step sizes.
    """
    data_column = 3
    ax2_column = 1

    img = raw[:, data_column]
    ax2 = raw[:, ax2_column]
    length_of_axis = 0
    old_val = ax2[0]
    for ind, val in enumerate(ax2):
        if val != old_val:
            length_of_axis = ind
            break
    img = img.reshape(-1, length_of_axis)

    return img, stepsizes


def get_img_from_dat(
    raw: np.ndarray, configs: Dict[str, Union[str, Dict]], condition: str
) -> np.ndarray:
    """
    Returns the image data from raw input based on specified configuration and condition.

    Args:
        raw (np.ndarray): Raw data.
        configs (Dict[str, Union[str, Dict]]): Configuration parameters.
        condition (str): Specific condition for processing the raw data.

    Returns:
        np.ndarray: Processed image data.
    """
    img, _ = get_img_and_stepsize(raw, configs, condition)
    return img


def get_stepsizes_from_dat(
    raw: np.ndarray, configs: Dict[str, Union[str, Dict]], condition: str
) -> np.ndarray:
    """
    Returns the step sizes from raw input based on specified configuration and condition.

    Args:
        raw (np.ndarray): Raw data.
        configs (Dict[str, Union[str, Dict]]): Configuration parameters.
        condition (str): Specific condition for processing the raw data.

    Returns:
        np.ndarray: Step sizes data.
    """
    _, stepsizes = get_img_and_stepsize(raw, configs, condition)
    return stepsizes


def normalise(
    img: np.ndarray,
    maximum: Optional[float] = None,
    minimum: Optional[float] = None,
    mode: Optional[str] = None,
) -> np.ndarray:
    """
    Normalises the image based on specified maximum, minimum and mode parameters.

    Args:
        img (np.ndarray): Input image.
        maximum (Optional[float], optional): Maximum value for normalisation. Defaults to None.
        minimum (Optional[float], optional): Minimum value for normalisation. Defaults to None.
        mode (Optional[str], optional): Normalisation mode. Can be 'squared' or 'log'. Defaults to None.

    Returns:
        np.ndarray: Normalised image.
    """
    img = np.asarray(img)
    if maximum == None:
        maximum = np.max(img)
    if minimum == None:
        minimum = np.min(img)
    if mode == None:
        return (img - minimum) / (maximum - minimum)
    elif mode == "squared":
        img_new = (img - minimum) / (maximum - minimum)
        return 1 - (img_new - 1) ** 2
    elif mode == "log":
        img_new = (img - minimum) / (maximum - minimum)
        return np.log(1000 * img_new + 1) / math.log(1001)
    else:
        print("Unkown mode: ", mode)


def normalise_two_images(
    img1: np.ndarray,
    img2: np.ndarray,
    maximum: Optional[float] = None,
    minimum: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalises two images based on specified maximum and minimum parameters.
    Images might be of different shape, then use this.

    Args:
        img1 (np.ndarray): First input image.
        img2 (np.ndarray): Second input image.
        maximum (Optional[float], optional): Maximum value for normalisation. Defaults to None.
        minimum (Optional[float], optional): Minimum value for normalisation. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the normalised images.
    """
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    if maximum == None:
        maximum = np.max([np.max(img1), np.max(img2)])
    if minimum == None:
        minimum = np.min([np.min(img1), np.min(img2)])
    return (img1 - minimum) / (maximum - minimum), (img2 - minimum) / (
        maximum - minimum
    )


def cutout_rectangle(
    img: np.ndarray, blob: np.ndarray, sidelength: np.ndarray, initial_ratio: float
) -> np.ndarray:
    """
    Cuts out a rectangle from the input image based on specified blob, side length and initial ratio parameters.

    Args:
        img (np.ndarray): Input image.
        blob (np.ndarray): Blob data used to define the rectangle.
        sidelength (np.ndarray): Side length of the rectangle.
        initial_ratio (float): Initial ratio parameter.

    Returns:
        np.ndarray: Image with the rectangle cut out.
    """
    bottom = np.array(blob - sidelength / 2, dtype=int)
    top = np.array(blob + sidelength / 2, dtype=int)

    bottom = np.maximum(0, bottom)
    top = np.maximum(0, top)

    img_save = img.copy()
    img = img[bottom[0] : top[0], bottom[1] : top[1]]

    return img


def cutout_and_attach(
    data_set: Dict,
    configs: Dict[str, Union[str, Dict]],
    target_margin: Union[int, float],
    verbose: bool = False,
    visualise: bool = False,
) -> Dict:
    """
    Cuts out rectangles from the images in the dataset and attaches them based on the specified configuration and target margin.

    Args:
        data_set (Dict): Dataset containing image data.
        configs (Dict[str, Union[str, Dict]]): Configuration parameters.
        target_margin (Union[int, float]): Target margin for cutting out rectangles.
        verbose (bool, optional): If True, print verbose messages during the process. Defaults to False.
        visualise (bool, optional): If True, visualise the images during the process. Defaults to False.

    Returns:
        Dict: Dataset with the cutouts attached.
    """
    print("cutting")
    for set_name in data_set:
        if verbose:
            print("set_name", set_name)
        if configs["metainfo"]["bias_triangles_all_same_size"]:
            size_bias_tri = configs["metainfo"]["size_of_bias_triangles"]
            target_margin = np.array(target_margin)
        else:
            size_bias_tri = configs["metainfo"]["size_of_bias_triangles"][set_name]
        size_bias_tri = np.array(size_bias_tri)

        if configs["metainfo"]["device_name_all_same"]:
            device_name = configs["metainfo"]["device_name"]
        else:
            device_name = configs["metainfo"]["device_name"][set_name]

        target_size = size_bias_tri * (1 + target_margin / 100)  # this is in mV
        if verbose:
            print("target size", target_size)
            print("size_bias_tri", size_bias_tri)

        for condition in data_set[set_name]:
            if verbose:
                print("condition", condition)
                print("imgshape", data_set[set_name][condition]["img"].shape)
            data_set[set_name][condition]["device_name"] = device_name

            stepsize = np.array(data_set[set_name][condition]["stepsizes"])
            if verbose:
                print(set_name, condition, ", stepsize", stepsize)
            sidelength_px = np.array((target_size / stepsize), dtype=int)
            if verbose:
                print("sidelength:", sidelength_px)

            raw_image = data_set[set_name][condition]["img"]

            data_set[set_name][condition]["cutouts"] = []

            if "blobs" in data_set[set_name][condition].keys():
                blobs = data_set[set_name][condition]["blobs"][:, :2]

                initial_ratio = sidelength_px[0] / sidelength_px[1]
                if verbose:
                    print("sidelength_px", sidelength_px)
                for ind, blob in enumerate(blobs):
                    if visualise:
                        plt.imshow(raw_image)
                        plt.show()
                        print("blob", blob)
                        plt.imshow(raw_image)
                        plt.scatter(
                            blob[1].flatten(),
                            blob[0].flatten(),
                            s=100,
                            marker="x",
                            color="red",
                        )
                        plt.show()

                    # if blob<
                    cutouts = cutout_rectangle(
                        raw_image,
                        blob,
                        sidelength_px.copy(),
                        initial_ratio=initial_ratio,
                    )
                    if verbose:
                        print("cutoutsshape", cutouts.shape)

                    if visualise:
                        plt.imshow(cutouts)
                        plt.show()

                    data_set[set_name][condition]["cutouts"].append(cutouts)
            else:
                boxes = data_set[set_name][condition]["boxes"]

                initial_ratio = sidelength_px[0] / sidelength_px[1]
                if verbose:
                    print("sidelength_px", sidelength_px)
                for ind, box in enumerate(boxes):
                    cutouts = cutout_box(raw_image, box)
                    if verbose:
                        print("cutoutsshape", cutouts.shape)

                    if visualise:
                        plt.imshow(cutouts)
                        plt.show()

                    data_set[set_name][condition]["cutouts"].append(cutouts)

    return data_set


def export_same_bias(
    data_set: Dict,
    ds_name: str,
    configs: Dict[str, Union[str, Dict]],
    output_size: Union[int, Tuple[int, int]],
    verbose: bool = False,
) -> Tuple[List, List, List, List]:
    """
    Exports images of same bias from the dataset according to the provided configurations.

    Args:
        data_set (Dict): Dataset containing image data.
        ds_name (str): Dataset name.
        configs (Dict[str, Union[str, Dict]]): Configuration parameters.
        output_size (Union[int, Tuple[int, int]]): Output size for the image. If it's a single integer, it will be treated as a square size.
        verbose (bool, optional): If True, print verbose messages during the process. Defaults to False.

    Returns:
        Tuple[List, List, List, List]: Lists of images, labels, names and device names.
    """
    imgs = []
    names = []
    labels = []
    device_name = []

    for set_name in data_set:
        if verbose:
            print("extracting", set_name)

        for condition in ["pos_bias_blocked", "neg_bias_blocked"]:
            try:
                for i, tri in enumerate(data_set[set_name][condition]["bias_triangle"]):
                    if tri:  # checks if this blob is a triangle or not
                        blocked = data_set[set_name][condition]["cutouts"][i]
                        if output_size:
                            blocked = iaa.Resize(output_size).augment_image(blocked)

                        cond2 = (
                            "pos_bias_unblocked"
                            if condition == "pos_bias_blocked"
                            else "neg_bias_unblocked"
                        )
                        unblocked = data_set[set_name][cond2]["cutouts"][i]
                        if output_size:
                            unblocked = iaa.Resize(output_size).augment_image(unblocked)

                        if data_set[set_name][condition]["bias_reversed"]:
                            blocked = -1 * blocked
                            unblocked = -1 * unblocked

                        direction = data_set[set_name][condition][
                            "which_direction_triangles"
                        ]
                        if direction == "topright":
                            if verbose:
                                print("correcting bias", set_name, condition, i)
                            blocked = -1 * blocked[::-1, ::-1]
                            unblocked = -1 * unblocked[::-1, ::-1]
                        elif direction == "topleft":
                            if verbose:
                                print("correcting bias", set_name, condition, i)
                            blocked = blocked[::-1]
                            unblocked = unblocked[::-1]
                        elif direction == "bottomleft":
                            pass
                        elif direction == "bottomright":
                            if verbose:
                                print("correcting bias", set_name, condition, i)
                            blocked = -1 * blocked[:, ::-1]
                            unblocked = -1 * unblocked[:, ::-1]

                        imgs.append(normalise_two_images(blocked, unblocked))

                        names.append(
                            str(ds_name)
                            + "_"
                            + str(set_name)
                            + "_"
                            + condition[:3]
                            + "bias_ind_"
                            + str(i)
                        )

                        label = (
                            1
                            if data_set[set_name][condition]["pauli_spin_blockade"][i]
                            else 0
                        )
                        labels.append(label)

                        device_name.append(data_set[set_name][condition]["device_name"])

            except:
                if verbose:
                    print(
                        "an error occured, probably not a full dataset?",
                        set_name,
                        condition,
                    )

    return imgs, labels, names, device_name


def cutout(img: np.ndarray, blob: np.ndarray, sidelength: int = 50) -> np.ndarray:
    """
    Cuts out a square region from the image centered around the blob.

    Args:
        img (np.ndarray): The input image.
        blob (np.ndarray): The blob around which to cut out.
        sidelength (int, optional): The sidelength of the square to cut out. Defaults to 50.

    Returns:
        np.ndarray: The cut out image.
    """
    x_bottom = (
        int(blob[0] - sidelength / 2) if int(blob[0] - sidelength / 2) >= 0 else 0
    )
    x_top = int(blob[0] + sidelength / 2) if int(blob[0] + sidelength / 2) >= 0 else 0
    y_bottom = (
        int(blob[1] - sidelength / 2) if int(blob[1] - sidelength / 2) >= 0 else 0
    )
    y_top = int(blob[1] + sidelength / 2) if int(blob[1] + sidelength / 2) >= 0 else 0
    img_save = img.copy()
    img = img[x_bottom:x_top, y_bottom:y_top]
    while img.shape[0] != img.shape[1]:
        sidelength -= 1
        img = cutout(img_save, blob, sidelength)
    return img


def cutout_box(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Cuts out a rectangular region from the image defined by the box.

    Args:
        img (np.ndarray): The input image.
        box (np.ndarray): The box defining the region to cut out.

    Returns:
        np.ndarray: The cut out image.
    """
    x_bottom = np.min(box[:, 1])
    x_top = np.max(box[:, 1])

    y_bottom = np.min(box[:, 0])
    y_top = np.max(box[:, 0])

    img = img[y_bottom:y_top, x_bottom:x_top]
    return img


def convert_voltage_coord_to_px_coordinates(
    coord: np.ndarray, d_file: Dict
) -> np.ndarray:
    """
    Converts voltage coordinates to pixel coordinates.

    Args:
        coord (np.ndarray): Voltage coordinates to be converted.
        d_file (Dict): File containing the mapping information from voltage to pixel coordinates.

    Returns:
        np.ndarray: Pixel coordinates.
    """
    bottom_x = d_file["chan0"]["vals"][1][0]
    res_x = np.abs(bottom_x - d_file["chan0"]["vals"][1][1])
    px_x = (coord[0] - bottom_x) / res_x
    bottom_y = d_file["chan0"]["vals"][0][0]
    res_y = np.abs(bottom_y - d_file["chan0"]["vals"][0][1])
    px_y = (coord[1] - bottom_y) / res_y

    return np.array([px_x, px_y], dtype=int)


def convert_vboxes_to_pxboxes(boxes: np.ndarray, d_file: Dict) -> np.ndarray:
    """
    Converts boxes defined in voltage coordinates to boxes defined in pixel coordinates.

    Args:
        boxes (np.ndarray): Boxes defined in voltage coordinates.
        d_file (Dict): File containing the mapping information from voltage to pixel coordinates.

    Returns:
        np.ndarray: Boxes defined in pixel coordinates.
    """
    px_boxes = []
    for box in boxes:
        px_boxes.append([])
        for coord in box:
            coord_px = convert_voltage_coord_to_px_coordinates(coord, d_file)
            px_boxes[-1].append(coord_px)
    return np.array(px_boxes)
