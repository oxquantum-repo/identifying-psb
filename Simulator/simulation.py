import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from tqdm.notebook import tqdm
from scipy.ndimage import gaussian_filter
from typing import Tuple, List, Dict, Optional, Union


def normalise(
    img: np.ndarray, maximum: float = None, minimum: float = None
) -> np.ndarray:
    """
    Normalise an image to [0, 1] range by subtracting the minimum and dividing by the range.

    Args:
        img (np.ndarray): Input image.
        maximum (float, optional): Maximum value of the image. If None, it's computed from the image. Defaults to None.
        minimum (float, optional): Minimum value of the image. If None, it's computed from the image. Defaults to None.

    Returns:
        np.ndarray: Normalised image.
    """
    img = np.asarray(img)
    if maximum == None:
        maximum = np.max(img)
    if minimum == None:
        minimum = np.min(img)
    return (img - minimum) / (maximum - minimum)


def get_current(
    tc: np.ndarray, gamma_s: np.ndarray, gamma_d: np.ndarray, epsilon: float
) -> np.ndarray:
    """
    Compute the stationary current through a double quantum dot system.

    Args:
        tc (np.ndarray): Tunnel coupling between two quantum dots.
        gamma_s (np.ndarray): Tunneling rate from source to the first dot.
        gamma_d (np.ndarray): Tunneling rate from the second dot to the drain.
        epsilon (float): Energy difference between two quantum dots

    Returns:
        np.ndarray: The stationary current.
    """
    gamma_s[gamma_s == 0] = 10 ** (-100)
    denominator = tc**2 * (2 + gamma_d / gamma_s) + 0.25 * gamma_d**2 + epsilon**2
    denominator[denominator == 0] = 10 ** (-100)
    current_ss = (tc**2 * gamma_d) / denominator
    current_ss = np.nan_to_num(current_ss)
    return current_ss


def fermi(
    x: Union[float, np.ndarray], mu: float, temp: float
) -> Union[float, np.ndarray]:
    """
    Calculate the Fermi-Dirac distribution.

    Args:
        x (Union[float, np.ndarray]): Energy levels where the Fermi function is evaluated.
        mu (float): Chemical potential or Fermi energy level.
        temp (float): Absolute temperature in Kelvin.

    Returns:
        Union[float, np.ndarray]: Fermi-Dirac distribution evaluated at energy levels 'x'.
    """
    k = 8.617e-5
    beta = (1e-3) / (k * temp)  # meV -> eV
    return 1 / (1 + np.exp((x - mu) * beta))


def get_charge_jump_noise() -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Generate the charge jump noise parameters.

    Returns:
        Tuple[int, int, np.ndarray, np.ndarray]: Number of rolls, axis of rolls, roll lengths and roll points.
    """
    n_rolls = np.random.choice((0, 1, 2))
    rollaxis = np.random.choice((0, 1))
    rolllengths = np.random.uniform(-0.05, 0.05, n_rolls)
    rollpoints = np.random.randint(0, 99, n_rolls)
    return n_rolls, rollaxis, rolllengths, rollpoints


def sample_factors() -> Dict[str, Union[float, np.ndarray, int, bool]]:
    """
    Sample different factors for the simulation.

    Returns:
        Dict[str, Union[float, np.ndarray, int, bool]]: A dictionary containing the sampled factors.
    """

    mu_s = np.random.uniform(-0.5, 0.5)
    bias = np.random.uniform(-2, -0.1)
    mu_d = mu_s + bias
    temp = np.random.uniform(0.1, 1)
    shift = np.random.uniform(0.01, 2)

    lever_arms = np.random.uniform(0.5, 1.5, 2)
    cross_talk = np.random.uniform(0.0, 0.5, 2)

    n_pots_1 = np.random.randint(2, 4)
    n_pots_2 = np.random.randint(2, 7)

    charging1 = np.random.uniform(bias - 0.2, bias + 0.2)
    epsilon = charging1 / 4
    charging11_deltas = np.array(
        [0] + [*np.random.uniform(-epsilon, epsilon, n_pots_1 - 1)]
    )

    charging2 = np.random.uniform(0.2, 0.5)
    epsilon = charging2 / 4
    charging21_deltas = np.array(
        [0] + [*np.random.uniform(-epsilon, epsilon, n_pots_2 - 1)]
    )

    dot1_pot_arrangement = charging11_deltas + [i * charging1 for i in range(n_pots_1)]

    dot2_pot_arrangement = charging21_deltas + [i * charging2 for i in range(n_pots_2)]

    gamma_s = np.random.uniform(0.01, 0.5, n_pots_1)
    gamma_d = np.random.uniform(0.01, 0.5, n_pots_2)

    jitter_var = np.random.uniform(0, 0.05)

    in_psb = np.random.choice((True, False))

    tc = np.random.uniform(0.01, 0.4, [n_pots_1, n_pots_2])

    gaussian_blur = np.random.uniform(0.8, 1.2)
    white_noise_level = np.random.uniform(3e-2, 7e-2)

    return {
        "tc": tc,
        "gamma_s": gamma_s,
        "gamma_d": gamma_d,
        "mu_s": mu_s,
        "mu_d": mu_d,
        "temp": temp,
        "lever_arms": lever_arms,
        "cross_talk": cross_talk,
        "shift": shift,
        "dot1_pot_arrangement": dot1_pot_arrangement,
        "dot2_pot_arrangement": dot2_pot_arrangement,
        "in_psb": in_psb,
        "jitter_var": jitter_var,
        "gaussian_blur": gaussian_blur,
        "white_noise_level": white_noise_level,
    }


def get_triangle(
    dot1_base: np.ndarray,
    dot2_base: np.ndarray,
    dot1_pot_arrangement: np.ndarray,
    dot2_pot_arrangement: np.ndarray,
    mu_s: float,
    mu_d: float,
    temp: float,
    gamma_s: float,
    gamma_d: float,
    tc: float,
    jitter_var: float,
    kT: float,
    temp_broadening: bool = True,
    in_psb: bool = None,
    **kwargs
) -> np.ndarray:
    """
    Compute the current through a quantum dot system.

    Args:
        dot1_base (np.ndarray): Base potential of the first dot.
        dot2_base (np.ndarray): Base potential of the second dot.
        dot1_pot_arrangement (np.ndarray): Potential arrangement of the first dot.
        dot2_pot_arrangement (np.ndarray): Potential arrangement of the second dot.
        mu_s (float): Chemical potential of the source.
        mu_d (float): Chemical potential of the drain.
        temp (float): Temperature of the system.
        gamma_s (float): Source tunneling rate.
        gamma_d (float): Drain tunneling rate.
        tc (float): Coherence time.
        jitter_var (float): Variance of the jitter.
        kT (float): Boltzmann constant times temperature.
        temp_broadening (bool): If True, applies temperature broadening.
        in_psb (bool): If True, includes Pauli spin blockade.
        kwargs: Additional parameters.

    Returns:
        np.ndarray: Computed current through the quantum dot system.
    """
    dot1_pots = dot1_base[:, :, np.newaxis] + dot1_pot_arrangement[np.newaxis, :]

    dot2_pots = dot2_base[:, :, np.newaxis] + dot2_pot_arrangement[np.newaxis, :]

    # to jitter each potential level randomly
    jitter = 1 + jitter_var * np.random.randn(*dot1_pots.shape)
    jitter2 = 1 + jitter_var * np.random.randn(*dot2_pots.shape)

    # source/drain tunneling rates
    rate_s = fermi(x=dot1_pots, mu=mu_s, temp=temp) * gamma_s
    rate_d = (1 - fermi(x=dot2_pots, mu=mu_d, temp=temp)) * gamma_d

    rate_s = rate_s[
        ..., np.newaxis
    ]  # shape: [first_voltage,second_voltage, firstdot, seconddot]
    rate_d = rate_d[
        :, :, np.newaxis
    ]  # shape: [first_voltage,second_voltage, firstdot, seconddot]

    # difference between levels
    epsilon = (jitter * dot1_pots)[..., np.newaxis] - (jitter2 * dot2_pots)[
        :, :, np.newaxis
    ]
    if in_psb:
        epsilon[:, :, 0, 0] = np.zeros(epsilon[:, :, 0, 0].shape)

    current = get_current(tc=tc, gamma_s=rate_s, gamma_d=rate_d, epsilon=epsilon)

    # tunneling only allowed in downward direction
    current = np.heaviside(epsilon, 0) * current
    # summing up over all possible channels
    current = np.sum(current, axis=(2, 3))

    # computing masks for bias window and PSB
    if temp_broadening:
        mu_s_window = mu_s + kT
        mu_d_window = mu_d - kT
    else:
        mu_s_window = mu_s
        mu_d_window = mu_d

    initial_setting = np.heaviside(mu_s_window - dot1_pots[:, :, 0], 0)
    interdot_matrix = np.heaviside(
        (
            (jitter[:, :, 0] * dot1_pots[:, :, 0])
            - jitter2[:, :, 0] * dot2_pots[:, :, 0]
        ),
        0,
    )
    dot2_setting = interdot_matrix * initial_setting
    final_matrix = np.heaviside(dot2_pots[:, :, 0] - mu_d_window, 0)

    bias_window_mask = final_matrix * dot2_setting

    psb_mask = np.heaviside(
        (
            (jitter[:, :, 0] * dot1_pots[:, :, 0])
            - jitter2[:, :, 1] * dot2_pots[:, :, 1]
        ),
        0,
    )

    current = bias_window_mask * current
    if in_psb:
        current = psb_mask * current
    return current


def get_voltage_extent(
    kT: float,
    lever_arms: List[float],
    cross_talk: List[float],
    shift: float,
    mu_s: float,
    mu_d: float,
    temp_broadening: bool = None,
    blank_space: float = 0,
    **kwargs
) -> List[List[float]]:
    """
    Compute the extent of the bias triangles in the voltage space.

    Args:
        kT (float): Boltzmann constant times temperature.
        lever_arms (List[float]): Lever arms for the quantum dots.
        cross_talk (List[float]): Cross talk coefficients for the quantum dots.
        shift (float): Energy shift.
        mu_s (float): Chemical potential of the source.
        mu_d (float): Chemical potential of the drain.
        temp_broadening (bool): If True, applies temperature broadening.
        blank_space (float): Fraction of additional blank space.
        kwargs: Additional parameters.

    Returns:
        List[List[float]]: Bounds of the voltages for the first and second quantum dots.
    """
    if temp_broadening:
        source_pot = mu_s + kT
        drain_pot = mu_d - kT
    else:
        source_pot = mu_s
        drain_pot = mu_d
    bias = source_pot - drain_pot
    # compute extent
    y_extent = []
    x_extent = []
    a = lever_arms[0]
    b = cross_talk[0]
    c = cross_talk[1]
    d = lever_arms[1]

    y_extent.append((source_pot * (1 - c / a)) / (d - (c * b / a)))
    x_extent.append((source_pot - b * y_extent[-1]) / a)

    y_extent.append(((drain_pot - shift) * (1 - c / a)) / (d - (c * b / a)))
    x_extent.append((drain_pot - shift - b * y_extent[-1]) / a)

    y_extent.append((drain_pot - source_pot * c / a) / (d - (c * b / a)))
    x_extent.append((source_pot - b * y_extent[-1]) / a)

    y_extent.append(
        (drain_pot - shift - (source_pot - shift) * c / a) / (d - (c * b / a))
    )
    x_extent.append((source_pot - shift - b * y_extent[-1]) / a)

    y_extent = [np.min(y_extent), np.max(y_extent)]
    x_extent = [np.min(x_extent), np.max(x_extent)]
    y_dis = y_extent[1] - y_extent[0]
    x_dis = x_extent[1] - x_extent[0]
    sidelength = (1 + blank_space) * np.max((x_dis, y_dis))

    dot1_voltage_bounds = [
        np.mean(x_extent) - sidelength / 2,
        np.mean(x_extent) + sidelength / 2,
    ]

    dot2_voltage_bounds = [
        np.mean(y_extent) - sidelength / 2,
        np.mean(y_extent) + sidelength / 2,
    ]
    return [dot1_voltage_bounds, dot2_voltage_bounds]


def get_base_pots(
    params: Dict[str, Union[float, np.ndarray, int, bool]],
    adjust_voltage_window: bool = True,
    blank_space: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the base potentials for the two dots.

    Args:
        params (Dict[str, Union[float, np.ndarray, int, bool]]): Parameters for the simulation.
        adjust_voltage_window (bool, optional): Flag to adjust the voltage window. Defaults to True.
        blank_space (float, optional): Extra blank space in voltage window. Defaults to 0.5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The base potentials for the two dots.
    """
    lever_arms = params["lever_arms"]
    cross_talk = params["cross_talk"]
    temp = params["temp"]
    kT = 8.617e-5 * temp * 1e3

    if adjust_voltage_window:
        voltage_bounds = get_voltage_extent(kT=kT, blank_space=blank_space, **params)
    else:
        voltage_bounds = params["voltage_bounds"]

    n_points = params["n_points"]

    dot1_voltages = np.linspace(voltage_bounds[0][0], voltage_bounds[0][1], n_points[0])
    dot2_voltages = np.linspace(voltage_bounds[1][0], voltage_bounds[1][1], n_points[1])

    xv, yv = np.meshgrid(dot1_voltages, dot2_voltages)

    dot1_base = lever_arms[0] * xv + cross_talk[0] * yv
    dot2_base = lever_arms[1] * yv + cross_talk[1] * xv
    return dot1_base, dot2_base


def get_bias_triangles(
    params: Dict,
    gaussian_blur: Optional[float] = None,
    white_noise_level: Optional[float] = None,
    adjust_voltage_window: bool = True,
    max_current: Optional[float] = None,
    blank_space: float = 0.5,
) -> np.ndarray:
    """
    Generate bias triangles for a quantum dot system.

    Args:
        params (Dict): Parameters of the quantum dot system including:
                       - 'n_points': number of points for the quantum dot system.
                       - 'temp': temperature of the system.
                       - 'shift': energy shift.
                       - 'gaussian_blur': standard deviation for Gaussian kernel.
                       - 'white_noise_level': standard deviation of the white noise.
        gaussian_blur (Optional[float]): Overrides the 'gaussian_blur' in params if provided.
        white_noise_level (Optional[float]): Overrides the 'white_noise_level' in params if provided.
        adjust_voltage_window (bool): If True, adjusts the voltage window.
        max_current (Optional[float]): Maximum current level. If None, the maximum current in the computed current will be used. Used because white noise is scaled to maximum current and needed for creating pairs of stability diagrams
        blank_space (float): Fraction of additional blank space around the triangles.

    Returns:
        np.ndarray: Computed current through the quantum dot system with applied noise and blur.
    """

    n_points = params["n_points"]
    kT = 8.617e-5 * params["temp"] * 1e3

    dot1_base, dot2_base = get_base_pots(
        params, adjust_voltage_window=adjust_voltage_window, blank_space=blank_space
    )

    first_triangle = get_triangle(
        **params, dot1_base=dot1_base, dot2_base=dot2_base, kT=kT
    )

    dot1_base += params["shift"]
    dot2_base += params["shift"]

    second_triangle = get_triangle(
        **params, dot1_base=dot1_base, dot2_base=dot2_base, kT=kT
    )

    first_triangle = first_triangle.reshape(-1)
    second_triangle = second_triangle.reshape(-1)
    current = np.max((first_triangle, second_triangle), axis=0).reshape(n_points)

    if gaussian_blur == None:
        gaussian_blur = params["gaussian_blur"]
    if white_noise_level == None:
        white_noise_level = params["white_noise_level"]

    # Blur current
    current = gaussian_filter(current, gaussian_blur)
    # Add noise
    if max_current == None:
        max_current = np.max(current)
    current = current + np.abs(
        np.random.normal(0, max_current * white_noise_level, current.shape)
    )

    n_rolls, rollaxis, rolllengths, rollpoints = get_charge_jump_noise()
    rolllengths_px = np.array(rolllengths * current.shape[rollaxis], dtype=int)

    for i in range(n_rolls):
        if rollaxis == 0:
            current[:, : rollpoints[i]] = np.roll(
                current[:, : rollpoints[i]], rolllengths_px[i], rollaxis
            )
        else:
            current[: rollpoints[i]] = np.roll(
                current[: rollpoints[i]], rolllengths_px[i], rollaxis
            )

    return current


def get_simulation(
    params: Dict[str, Union[float, np.ndarray, int, bool]],
    max_current: Optional[float] = None,
) -> np.ndarray:
    """
    Get a simulated image of bias triangles.

    Args:
        params (Dict[str, Union[float, np.ndarray, int, bool]]): Parameters for the simulation.
        max_current (Optional[float], optional): Maximum current. Defaults to None.

    Returns:
        np.ndarray: A simulated image of bias triangles.
    """
    params["n_points"] = [100, 100]
    current = get_bias_triangles(params, max_current=max_current, blank_space=0.5)
    current = np.rot90(current, 2)

    return current


def simulate(
    n_imgs: int,
    return_sampling_factors: bool = False,
    visualise: bool = False,
    sample_factors_func:Optional[callable]=sample_factors,
) -> Union[
    Tuple[
        List[np.ndarray],
        List[bool],
        List[Dict[str, Union[float, np.ndarray, int, bool]]],
    ],
    Tuple[List[np.ndarray], List[bool]],
]:
    """
    Generate a set of simulated images of bias triangles.

    Args:
        n_imgs (int): Number of images to generate.
        return_sampling_factors (bool, optional): Flag to return the sampling factors. Defaults to False.
        visualise (bool, optional): Flag to visualise the generated images. Defaults to False.
        sample_factors_func (Optional[callable], optional): Function to sample factors. Defaults to sample_factors().

    Returns:
        Union[Tuple[List[np.ndarray], List[bool], List[Dict[str, Union[float, np.ndarray, int, bool]]]], Tuple[List[np.ndarray], List[bool]]]: List of images, list of boolean flags indicating if PSB is present, and optionally list of sampled factors.
    """
    if sample_factors_func is not None:
        sample_factors = sample_factors_func
    sample_facs = []
    imgs = []
    psb_label = []

    for i in tqdm(range(n_imgs)):
        params = sample_factors()
        sample_facs.append(params)
        psb = params["in_psb"]

        params["in_psb"] = False
        img_no_psb = get_simulation(params)
        if visualise:
            plt.imshow(img_no_psb)
            plt.show()

        ###########
        params["in_psb"] = psb
        img = get_simulation(params, max_current=np.max(img_no_psb))
        if visualise:
            print("PSB:", in_psb)
            plt.imshow(img)
            plt.show()

        # sample_facs.append(sampled)
        this_imgs = normalise(np.array([img, img_no_psb]))
        # this_imgs=(np.array([img,img_no_psb]))
        imgs.append(this_imgs)
        psb_label.append(psb)
        if visualise:
            print("----")
    if return_sampling_factors:
        return imgs, psb_label, sample_facs
    else:
        return imgs, psb_label
