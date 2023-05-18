import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn
from typing import List
from skimage.draw import rectangle_perimeter

import bokeh
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColorBar, LinearColorMapper
from bokeh.io import output_notebook

output_notebook()
# from bokeh.palettes import Magma
cmap = "icefire"
colormap = cm.get_cmap(cmap)
bokehpalette = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]


def draw_boxes_in_img_pxspace(
    _img: np.ndarray, boxes: np.ndarray, condition: str
) -> np.ndarray:
    """
    Function to draw boxes on an image in pixel space.

    Parameters:
    _img (np.ndarray): The image on which boxes are to be drawn.
    boxes (np.ndarray): The coordinates of the boxes to be drawn.
    condition (str): Determines the color of the box. If 'positive bias', the color is set to the maximum color in the image. Otherwise, it's set to the minimum color in the image.

    Returns:
    np.ndarray: The image with boxes drawn on it.
    """
    img = _img.copy()
    for i in range(len(boxes)):
        start = boxes[i, 0]
        end = boxes[i, 1]
        rr, cc = rectangle_perimeter(start, end=end, shape=img.shape)
        if condition == "positive bias":
            img[rr, cc] = np.max(img)
        else:
            img[rr, cc] = np.min(img)

    p = figure(x_range=(0, img.shape[1]), y_range=(0, img.shape[0]))

    # mapper = LinearColorMapper(palette=bokehpalette,low=img.min(),high=img.max())
    p.image(
        image=[img], x=0, y=0, dw=img.shape[1], dh=img.shape[0], palette=bokehpalette
    )
    p.add_tools(HoverTool())
    show(p, notebook_handle=True)

    return img


def draw_box_in_img_pxspace(
    _img: np.ndarray, box: np.ndarray, condition: str
) -> np.ndarray:
    """
    Function to draw a single box on an image in pixel space.

    Parameters:
    _img (np.ndarray): The image on which the box is to be drawn.
    box (np.ndarray): The coordinates of the box to be drawn.
    condition (str): Determines the color of the box. If 'positive bias', the color is set to the maximum color in the image. Otherwise, it's set to the minimum color in the image.

    Returns:
    np.ndarray: The image with the box drawn on it.
    """
    img = _img.copy()
    start = box[0]
    end = box[1]
    rr, cc = rectangle_perimeter(start, end=end, shape=img.shape)
    if condition == "positive bias":
        img[rr, cc] = np.max(img)
    else:
        img[rr, cc] = np.min(img)

    p = figure(x_range=(0, img.shape[1]), y_range=(0, img.shape[0]))

    p.image(
        image=[img], x=0, y=0, dw=img.shape[1], dh=img.shape[0], palette=bokehpalette
    )
    p.add_tools(HoverTool())
    show(p, notebook_handle=True)

    return img


def cut_out(img: np.ndarray, box: np.ndarray):
    """
    Cuts out a portion of the image based on the given box coordinates.

    Parameters:
    img (np.ndarray): The original image.
    box(np.ndarray): The coordinates of the box to cut out.
    """
    resizer = iaa.Resize(self.nn_input_size)

    x_bottom = np.min(box[:, 1])
    x_top = np.max(box[:, 1])

    y_bottom = np.min(box[:, 0])
    y_top = np.max(box[:, 0])

    img = img[x_bottom:x_top, y_bottom:y_top]


def draw_in_voltagespace(
    full_data: dict,
    v_box=None,
    px_box=None,
    condition=None,
    draw_box=False,
    cmap="icefire",
):
    """
    Function to draw data in voltage space.

    Parameters:
    full_data (dict): Dictionary containing the data to be plotted.
    v_box (np.ndarray, optional): The voltage box coordinates. If given, the plot is cropped to that window. Defaults to None.
    px_box (np.ndarray, optional): The pixel box coordinates. Defaults to None.
    condition (str, optional): Determines the color of the box. If 'positive bias', the color is set to the maximum color in the image. Otherwise, it's set to the minimum color in the image. Defaults to None.
    draw_box (bool, optional): If true, draw a box on the plot. Defaults to False.
    cmap (str, optional): The colormap to use for the plot. Defaults to "icefire".

    Returns:
    None.
    """
    colormap = cm.get_cmap(cmap)
    bokehpalette = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]

    TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset"

    z_label = "chan0"
    full_data = full_data[z_label]
    x_label = full_data["varz"][0]
    y_label = full_data["varz"][1]

    TOOLTIPS = [(z_label, "@image"), ("(%s,%s)" % (x_label, y_label), "($x, $y)")]

    x_vals = full_data["vals"][0]
    y_vals = full_data["vals"][1]

    x_ran = [x_vals[0], x_vals[-1]]
    y_ran = [y_vals[0], y_vals[-1]]
    x_width = abs(np.diff(x_ran)[0])
    y_width = abs(np.diff(y_ran)[0])

    if not v_box is None:
        v_box_x = np.sort(v_box[:, 1])
        v_box_x = (v_box_x[0], v_box_x[1])

        v_box_y = np.sort(v_box[:, 0])
        v_box_y = (v_box_y[0], v_box_y[1])

        fig = figure(
            x_range=v_box_x,
            y_range=v_box_y,
            x_axis_label=x_label,
            y_axis_label=y_label,
            toolbar_location="right",
            tools=TOOLS,
            tooltips=TOOLTIPS,
        )

    else:
        fig = figure(
            x_range=x_ran,
            y_range=y_ran,
            x_axis_label=x_label,
            y_axis_label=y_label,
            toolbar_location="right",
            tools=TOOLS,
            tooltips=TOOLTIPS,
        )

    data = full_data["data"]

    if draw_box:
        data = data.copy()
        start = px_box[0]
        end = px_box[1]
        rr, cc = rectangle_perimeter(start, end=end, shape=data.shape)
        if condition == "positive bias":
            data[rr, cc] = np.max(data)
        else:
            data[rr, cc] = np.min(data)

    mapper = LinearColorMapper(palette=bokehpalette, low=data.min(), high=data.max())

    d_s = fig.image(
        image=[data],
        x=x_ran[0],
        y=y_ran[0],
        dw=x_width,
        dh=y_width,
        color_mapper=mapper,
    )

    color_bar = ColorBar(
        color_mapper=mapper, label_standoff=5, orientation="horizontal", location=(0, 0)
    )

    fig.add_layout(color_bar, "above")
    show(fig, notebook_handle=True)
