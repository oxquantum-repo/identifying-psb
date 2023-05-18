import pickle
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

import pandas as pd
import seaborn as sns
import sys

sys.path.append("..")

from Training import training_utils
from typing import Dict, List, Optional, Tuple, Union


def get_classic_results(
    data_set: Dict[str, Union[ndarray, List[ndarray]]]
) -> Dict[str, Union[ndarray, List[ndarray]]]:
    """
    Process the results of a classification task on a given data set.

    The function computes the predicted classes and their scores for each instance in the data set.
    The computed results are then stored in a dictionary.

    Args:
        data_set (dict): A dictionary containing the following keys:
            - "TPR": list of true positive rates.
            - "predictions": list of predicted labels.
            - "scores": list of scores.
            - "true_labels": list of actual labels.
            - "device_names": list of device names.
            - "triangle_names": list of triangle names.

    Returns:
        dict: A dictionary containing the results of the classification task.
    """
    new_results = training_utils.get_results_dict()
    for i in range(len(data_set["TPR"])):
        predicted = np.array(data_set["predictions"][i])
        scores = np.array(data_set["scores"][i])
        y_test_this_rep = np.array(data_set["true_labels"][i])
        _device_names = data_set["device_names"][i]
        _names = data_set["triangle_names"][i]
        new_results = training_utils.record_results(
            new_results,
            predicted,
            scores,
            y_test_this_rep,
            _device_names,
            _names,
            mode="classic",
        )

    return new_results


def get_ensemble_results(
    results_only_sim: Dict[str, Union[ndarray, List[ndarray]]],
    results_mixed_data: Dict[str, Union[ndarray, List[ndarray]]],
    results_only_real_data: Dict[str, Union[ndarray, List[ndarray]]],
) -> Dict[str, Union[ndarray, List[ndarray]]]:
    """
    Aggregate the results of multiple classification tasks using ensemble learning.

    The function computes the ensemble results for three different scenarios:
    - Only simulated data
    - Mixed real and simulated data
    - Only real data

    The computed results are then stored in a dictionary.

    Args:
        results_only_sim (dict): A dictionary containing the results of the classification task using only simulated data.
        results_mixed_data (dict): A dictionary containing the results of the classification task using mixed real and simulated data.
        results_only_real_data (dict): A dictionary containing the results of the classification task using only real data.

    Returns:
        dict: A dictionary containing the ensemble results of the classification tasks.
    """
    n_reps = 10
    names = np.unique(results_only_sim["triangle_names"][0])
    predictions_arr = {"sim": [], "mixed": [], "real": []}
    triangle_names_arr = []  # {'sim':[],'mixed':[],'real':[]}
    scores_arr = {"sim": [], "mixed": [], "real": []}
    device_names_arr = []  # {'sim':[],'mixed':[],'real':[]}
    true_labels_arr = []  # {'sim':[],'mixed':[],'real':[]}

    majority_vote = False  # True

    for i, name in enumerate(names):
        print(f"Processing {i+1}/{len(names)}: {name}")
        triangle_names_arr.append(name)
        idx = np.where(results_only_sim["triangle_names"][0] == name)
        device_names_arr.append(results_only_sim["device_names"][0][idx])
        true_labels_arr.append(results_only_sim["true_labels"][0][idx])

        data = results_only_sim
        gathered_pred = 0
        gathered_scores = 0
        for i_rep in range(len(data["triangle_names"])):
            # print(data['triangle_names'][i_rep])
            idx = np.where(data["triangle_names"][i_rep] == name)
            gathered_pred += data["predictions"][i_rep][idx]
            gathered_scores += data["scores"][i_rep][idx]
        # predictions_arr['sim'].append(gathered_pred.numpy()/n_reps)
        if majority_vote:
            predictions_arr["sim"].append(gathered_pred.numpy() / n_reps)
            scores_arr["sim"].append(gathered_pred.numpy())
        else:
            predictions_arr["sim"].append(gathered_pred / n_reps)
            scores_arr["sim"].append(gathered_scores / n_reps)

        data = results_mixed_data
        gathered_pred = 0
        gathered_scores = 0
        for i_rep in range(len(data["triangle_names"])):
            # print(data['triangle_names'][i_rep])
            idx = np.where(data["triangle_names"][i_rep] == name)
            gathered_pred += data["predictions"][i_rep][idx]
            gathered_scores += data["scores"][i_rep][idx]
        predictions_arr["mixed"].append(gathered_pred / n_reps)
        # scores_arr['mixed'].append(gathered_scores/n_reps)
        if majority_vote:
            scores_arr["mixed"].append(gathered_pred)
        else:
            scores_arr["mixed"].append(gathered_scores / n_reps)

        data = results_only_real_data
        gathered_pred = 0
        gathered_scores = 0
        for i_rep in range(len(data["triangle_names"])):
            # print(data['triangle_names'][i_rep])
            idx = np.where(data["triangle_names"][i_rep] == name)
            gathered_pred += data["predictions"][i_rep][idx]
            gathered_scores += data["scores"][i_rep][idx]
        predictions_arr["real"].append(gathered_pred / n_reps)

        if majority_vote:
            scores_arr["real"].append(gathered_pred)
        else:
            scores_arr["real"].append(gathered_scores / n_reps)

    cutoff = 0.5

    scores_arr["sim"] = np.array(scores_arr["sim"]).ravel()
    if majority_vote:
        predictions_arr["sim"] = np.array(predictions_arr["sim"]).ravel() > cutoff
    else:
        predictions_arr["sim"] = scores_arr["sim"] > cutoff

    scores_arr["mixed"] = np.array(scores_arr["mixed"]).ravel()
    if majority_vote:
        predictions_arr["mixed"] = np.array(predictions_arr["mixed"]).ravel() > cutoff
    else:
        predictions_arr["mixed"] = scores_arr["mixed"] > cutoff

    scores_arr["real"] = np.array(scores_arr["real"]).ravel()
    if majority_vote:
        predictions_arr["real"] = np.array(predictions_arr["real"]).ravel() > cutoff
    else:
        predictions_arr["real"] = scores_arr["real"] > cutoff

    true_labels_arr = np.array(true_labels_arr).ravel()
    device_names_arr = np.array(device_names_arr).ravel()

    results_ensemble_classic = training_utils.get_results_dict()
    training_utils.record_results(
        results_ensemble_classic,
        predictions_arr["sim"],
        scores_arr["sim"],
        true_labels_arr,
        device_names_arr,
        triangle_names_arr,
        mode="classic",
    )
    training_utils.record_results(
        results_ensemble_classic,
        predictions_arr["real"],
        scores_arr["real"],
        true_labels_arr,
        device_names_arr,
        triangle_names_arr,
        mode="classic",
    )
    training_utils.record_results(
        results_ensemble_classic,
        predictions_arr["mixed"],
        scores_arr["mixed"],
        true_labels_arr,
        device_names_arr,
        triangle_names_arr,
        mode="classic",
    )

    return results_ensemble_classic


def get_complete_plot(
    results_only_sim: Dict[str, List[Tuple[List[float], List[float]]]],
    results_mixed_data: Dict[str, List[Tuple[List[float], List[float]]]],
    results_only_real_data: Dict[str, List[Tuple[List[float], List[float]]]],
    results_ensemble: Dict[str, List[Tuple[List[float], List[float]]]],
    save_name: Optional[str] = None,
    arrange_horizontally: bool = False,
) -> None:
    """
    This function plots the performance of different data conditions (real, simulated, mixed) and the ensemble method.
    It also plots the ROC curves for each condition and their AUC and Accuracy scores.

    Args:
        results_only_sim (Dict[str, List[Tuple[List[float], List[float]]]]): A dictionary with performance metrics and ROC curve values of the model trained on simulated data.
        results_mixed_data (Dict[str, List[Tuple[List[float], List[float]]]]): A dictionary with performance metrics and ROC curve values of the model trained on mixed data.
        results_only_real_data (Dict[str, List[Tuple[List[float], List[float]]]]): A dictionary with performance metrics and ROC curve values of the model trained on real data.
        results_ensemble (Dict[str, List[Tuple[List[float], List[float]]]]): A dictionary with performance metrics and ROC curve values of the ensemble model.
        save_name (Optional[str], optional): The name of the file to save the plot. If None, the plot is not saved. Defaults to None.
        arrange_horizontally (bool, optional): If True, the plots are arranged horizontally. If False, the plots are arranged vertically. Defaults to False.

    Returns:
        None: This function does not return any value. It just plots the performance of models.
    """

    def get_df(metric: str) -> pd.DataFrame:
        """
        This function converts dictionaries of results into pandas dataframes based on the provided metric and concatenates them into a single dataframe.

        Args:
            metric (str): The metric of interest. It should be a key in the results dictionaries.

        Returns:
            pd.DataFrame: A DataFrame containing values of the specified metric for simulated, real, and mixed data.
        """
        results_only_real_df = pd.DataFrame.from_dict(results_only_real_data)
        results_only_real_df = results_only_real_df[[metric]]
        results_only_real_df["condition"] = "Real data"
        results_only_sim_df = pd.DataFrame.from_dict(results_only_sim)
        results_only_sim_df = results_only_sim_df[[metric]]
        results_only_sim_df["condition"] = "Simulated data"

        results_only_mixed_df = pd.DataFrame.from_dict(results_mixed_data)
        results_only_mixed_df = results_only_mixed_df[[metric]]
        results_only_mixed_df["condition"] = "Mixed data"

        total = pd.concat(
            [results_only_real_df, results_only_mixed_df, results_only_sim_df]
        )
        return total

    def get_plot(
        ax: plt.Axes,
        ordering: List[str],
        names_on_axes: List[str],
        df: pd.DataFrame,
        metric: str,
        ylim: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
    ) -> plt.Axes:
        """
        This function generates a seaborn swarmplot and boxplot on the provided axes, based on the specified metric, and adds ensemble results as horizontal lines.

        Args:
            ax (plt.Axes): Matplotlib Axes object on which to draw the plots.
            ordering (List[str]): The order of the conditions in the plot.
            names_on_axes (List[str]): The labels for the conditions on the x-axis.
            df (pd.DataFrame): The DataFrame containing values of the specified metric for different conditions.
            metric (str): The metric of interest.
            ylim (Optional[Tuple[float, float]]): The limits for the y-axis. If None, defaults will be used.
            title (Optional[str]): The title for the plot. If None, no title will be set.

        Returns:
            plt.Axes: The Axes object with the drawn plots.
        """
        sns.swarmplot(
            y=metric,
            x="condition",
            data=df,
            order=ordering,
            size=5,
            ax=ax,
            clip_on=False,
            alpha=1,
            color="black",
        )
        sns.boxplot(
            y=metric,
            x="condition",
            color="white",
            order=ordering,
            data=df,
            whis=np.inf,
            ax=ax,
            # medianprops={'color':'grey', 'linestyle':'dashdot'},
            showmeans=True,
            meanline=True,
            meanprops={"color": "green", "linestyle": (0, (3, 1))},
        )
        ax.set_xticklabels(names_on_axes, rotation=0)
        ax.set(title=title, ylim=ylim, xlabel=None)
        margin = 0.033

        ensemble = results_ensemble[metric]
        ensemble_c = "orange"
        ax.axhline(
            ensemble[0],
            xmin=margin,
            xmax=0.33 - margin,
            c=ensemble_c,
            zorder=10,
            clip_on=False,
        )
        ax.axhline(
            ensemble[1],
            xmin=0.33 + margin,
            xmax=0.67 - margin,
            c=ensemble_c,
            zorder=10,
            clip_on=False,
        )
        ax.axhline(
            ensemble[2],
            xmin=0.67 + margin,
            xmax=1 - margin,
            c=ensemble_c,
            zorder=10,
            clip_on=False,
        )

        mean_results = [
            np.mean(df[df["condition"] == ordering[0]])[0],
            np.mean(df[df["condition"] == ordering[1]])[0],
            np.mean(df[df["condition"] == ordering[2]])[0],
        ]

        return ax

    ordering = ["Simulated data", "Real data", "Mixed data"]
    names_on_axes = ["Simulated data", "Real data", "Mixed data"]

    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches

    from scipy import interp

    def get_subplot(
        data: Dict[str, Union[List[Tuple[float, float]], float]],
        ax: plt.Axes,
        color: str = "b",
        label: Optional[str] = None,
        title: Optional[str] = None,
        plot_means: bool = True,
        plot_individuals: bool = False,
    ) -> None:
        """
        This function plots ROC curves on the provided axes for the given data.

        Args:
            data (Dict[str, Union[List[Tuple[float, float]], float]]): The dictionary containing ROC curve data.
            ax (plt.Axes): Matplotlib Axes object on which to draw the plots.
            color (str, optional): Color for the plot lines. Defaults to 'b'.
            label (Optional[str], optional): Label for the plot lines. If None, no label will be set.
            title (Optional[str], optional): Title for the plot. If None, no title will be set.
            plot_means (bool, optional): Whether to plot the mean ROC curve. Defaults to True.
            plot_individuals (bool, optional): Whether to plot individual ROC curves. Defaults to False.
        """
        tprs = []
        base_fpr = np.linspace(0, 1, 101)

        for [fpr, tpr] in data["ROC Curve"]:
            if plot_individuals:
                ax.plot(fpr, tpr, c=color, alpha=0.35, clip_on=False, zorder=10)
            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)

        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std

        if plot_means:
            mean_c = "green"
            ax.plot(
                base_fpr,
                mean_tprs,
                mean_c,
                linestyle=(0, (3, 1)),
                label=label,
                clip_on=False,
                zorder=10,
            )
        # ax.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

        ax.set(
            title=title,
            xlim=[-0.0, 1.0],
            ylim=[-0.0, 1.0],
            ylabel="True positive\nrate",
            xlabel="False\npositive rate",
        )
        ax.xaxis.labelpad = -10  # 7
        ax.yaxis.labelpad = -7
        ax.set_yticks([0, 1])
        ax.set_xticks([0, 1])

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize

    if arrange_horizontally:
        f = plt.figure(figsize=(10, 4), constrained_layout=True)

        gs0 = gridspec.GridSpec(1, 2, figure=f)
        gs00 = gs0[0].subgridspec(100, 100)
        ax1 = f.add_subplot(gs00[:, :-1])

        gs01 = gs0[1].subgridspec(30, 3)

        ax5 = f.add_subplot(gs01[11:, :])
        ax2 = f.add_subplot(gs01[:10, 0])
        ax3 = f.add_subplot(gs01[:10, 1])
        ax4 = f.add_subplot(gs01[:10, 2])

        axs = [ax1, ax2, ax3, ax4, ax5]

        get_subplot(
            results_only_sim, axs[1], sns.color_palette()[0], label=names_on_axes[0]
        )
        axs[1].plot(results_ensemble["ROC Curve"][0])
        get_subplot(
            results_only_real_data,
            axs[2],
            sns.color_palette()[1],
            label=names_on_axes[1],
            title="(b)",
        )
        get_subplot(
            results_mixed_data, axs[3], sns.color_palette()[2], label=names_on_axes[2]
        )

        axs[2].set_yticklabels([])
        axs[2].set(ylabel=None)
        axs[3].set_yticklabels([])
        axs[3].set(ylabel=None)

        axs[1].set(title=None, xlabel=None)
        axs[3].set(xlabel=None)

        metric = "AUC"
        ax = axs[4]
        df = get_df(metric)
        ax = get_plot(
            ax, ordering, names_on_axes, df, metric, title="", ylim=[0.9, 1.003]
        )

        metric = "Accuracy"
        ax = axs[0]
        df = get_df(metric)
        ax = get_plot(
            ax, ordering, names_on_axes, df, metric, title="(a)", ylim=[0.7, 1.003]
        )
        ax.set(ylabel="Accuracy")
    else:
        print("starting to plot")
        f = plt.figure(figsize=(4, 5), constrained_layout=True)

        gs0 = gridspec.GridSpec(5, 1, figure=f)
        gs00 = gs0[:2].subgridspec(1, 1)
        ax1 = f.add_subplot(gs00[:, :])

        gs01 = gs0[2:].subgridspec(3, 3)

        ax5 = f.add_subplot(gs01[1:, :])
        ax2 = f.add_subplot(gs01[:1, 0])
        ax3 = f.add_subplot(gs01[:1, 1])
        ax4 = f.add_subplot(gs01[:1, 2])

        axs = [ax1, ax2, ax3, ax4, ax5]

        ensemble_c = "orange"
        get_subplot(
            results_only_sim, axs[1], sns.color_palette()[0], label=names_on_axes[0]
        )
        [fpr, tpr] = results_ensemble["ROC Curve"][0]
        axs[1].plot(fpr, tpr, c=ensemble_c, clip_on=False, zorder=9)
        get_subplot(
            results_only_real_data,
            axs[2],
            sns.color_palette()[1],
            label=names_on_axes[1],
            title="(b)",
        )
        [fpr, tpr] = results_ensemble["ROC Curve"][1]
        axs[2].plot(fpr, tpr, c=ensemble_c, clip_on=False, zorder=9)
        get_subplot(
            results_mixed_data, axs[3], sns.color_palette()[2], label=names_on_axes[2]
        )
        [fpr, tpr] = results_ensemble["ROC Curve"][2]
        axs[3].plot(fpr, tpr, c=ensemble_c, clip_on=False, zorder=9)

        axs[2].set_yticklabels([])
        axs[2].set(ylabel=None)
        axs[3].set_yticklabels([])
        axs[3].set(ylabel=None)
        axs[1].set(title=None, xlabel=None)
        axs[3].set(xlabel=None)
        metric = "AUC"
        ax = axs[4]
        df = get_df(metric)
        ax = get_plot(
            ax, ordering, names_on_axes, df, metric, title="", ylim=[0.8, 1.0]
        )

        ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])

        metric = "Accuracy"
        ax = axs[0]
        df = get_df(metric)
        ax = get_plot(
            ax, ordering, names_on_axes, df, metric, title="(a)", ylim=[0.8, 1.0]
        )
        ax.set(ylabel="Accuracy")
        ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])

        class AnyObject(object):
            pass

        class data_handler(object):
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                scale = fontsize / 10
                x0, y0 = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                patch_sq = mpatches.Circle(
                    [x0 + 0.33 * width - height / 2, y0 + height / 2],
                    height / 2 * scale,
                    facecolor="black",
                    edgecolor="none",
                    transform=handlebox.get_transform(),
                )
                patch_circ = mpatches.Circle(
                    [x0 + 0.66 * width - height / 2, y0 + height / 2],
                    height / 2 * scale,
                    facecolor="black",
                    edgecolor="none",
                    transform=handlebox.get_transform(),
                )
                patch_circ_2 = mpatches.Circle(
                    [x0 + width - height / 2, y0 + height / 2],
                    height / 2 * scale,
                    facecolor="black",
                    edgecolor="none",
                    transform=handlebox.get_transform(),
                )

                handlebox.add_artist(patch_sq)
                handlebox.add_artist(patch_circ)
                handlebox.add_artist(patch_circ_2)
                return patch_sq

        ax.legend(
            [
                AnyObject(),
                Line2D([0], [0], color="green", linestyle=(0, (3, 1))),
                Line2D([0], [0], color="orange"),
            ],
            ["Individuals", "Mean\nindividual", "Ensemble"],
            handler_map={AnyObject: data_handler()},
            loc="lower right",
        )

    plt.savefig(save_name + ".pdf")
    plt.savefig(save_name + ".svg")
    plt.show()


def process_data_sample_size_comparison(
    classic_different_sample_sizes: Dict[str, dict],
    results_different_sample_sizes: Dict[str, dict],
    names: List[str],
    results_only_sim: Dict[str, List[np.ndarray]],
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    This function processes data results from different sample sizes to prepare them for comparison.

    Args:
        classic_different_sample_sizes (Dict[str, dict]): A dictionary of classic data results with different sample sizes.
        results_different_sample_sizes (Dict[str, dict]): A dictionary of simulation results with different sample sizes.
        names (List[str]): Names of the samples.
        results_only_sim (Dict[str, List[np.ndarray]]): A dictionary of simulation results.

    Returns:
        Tuple[Dict[str, dict], Dict[str, dict]]: A tuple containing processed classic data and ensemble data results.
    """
    n_reps = 10
    predictions_arr = {}
    scores_arr = {}
    for key, item in results_different_sample_sizes.items():
        predictions_arr[key] = []
        scores_arr[key] = []

    triangle_names_arr = []

    device_names_arr = []
    true_labels_arr = []

    majority_vote = False

    for i, name in enumerate(names):
        triangle_names_arr.append(name)
        idx = np.where(results_only_sim["triangle_names"][0] == name)
        device_names_arr.append(results_only_sim["device_names"][0][idx])
        true_labels_arr.append(results_only_sim["true_labels"][0][idx])

        for key, item in results_different_sample_sizes.items():
            data = results_different_sample_sizes[key]
            gathered_pred = 0
            gathered_scores = 0
            for i_rep in range(len(data["triangle_names"])):
                # print(data['triangle_names'][i_rep])
                idx = np.where(data["triangle_names"][i_rep] == name)
                gathered_pred += data["predictions"][i_rep][idx]
                gathered_scores += data["scores"][i_rep][idx]
            # predictions_arr['sim'].append(gathered_pred.numpy()/n_reps)
            if majority_vote:
                predictions_arr[key].append(gathered_pred.numpy() / n_reps)
                scores_arr[key].append(gathered_pred.numpy())
            else:
                predictions_arr[key].append(gathered_pred / n_reps)
                scores_arr[key].append(gathered_scores / n_reps)
    cutoff = 0.5

    for key, item in results_different_sample_sizes.items():
        scores_arr[key] = np.array(scores_arr[key]).ravel()
        if majority_vote:
            predictions_arr[key] = np.array(predictions_arr[key]).ravel() > cutoff
        else:
            predictions_arr[key] = scores_arr[key] > cutoff

    true_labels_arr = np.array(true_labels_arr).ravel()
    device_names_arr = np.array(device_names_arr).ravel()

    results_ensemble_compare_samplesize = training_utils.get_results_dict()
    for key, item in results_different_sample_sizes.items():
        training_utils.record_results(
            results_ensemble_compare_samplesize,
            predictions_arr[key],
            scores_arr[key],
            true_labels_arr,
            device_names_arr,
            triangle_names_arr,
            mode="classic",
        )

    return classic_different_sample_sizes, results_ensemble_compare_samplesize


def plot_sample_size_comparison(
    classic_different_sample_sizes: Dict[str, dict],
    results_different_sample_sizes: Dict[str, dict],
    results_ensemble_compare_samplesize: Dict[str, dict],
    save_name: str = "sample_size_comparison",
) -> None:
    """
    This function plots a comparison of classic data results, simulation results and ensemble results from different sample sizes.

    Args:
        classic_different_sample_sizes (Dict[str, dict]): A dictionary of classic data results with different sample sizes.
        results_different_sample_sizes (Dict[str, dict]): A dictionary of simulation results with different sample sizes.
        results_ensemble_compare_samplesize (Dict[str, dict]): A dictionary of ensemble data results.
        save_name (str, optional): The name of the file where the plot is saved. Defaults to "sample_size_comparison".

    Returns:
        None
    """

    def get_df(metric: str) -> pd.DataFrame:
        """
        This function converts dictionaries of classic data results into pandas dataframes based on the provided metric and concatenates them into a single dataframe.

        Args:
            metric (str): The metric of interest. It should be a key in the results dictionaries.

        Returns:
            pd.DataFrame: A DataFrame containing values of the specified metric for different sample sizes.
        """
        dataframes = []
        for key, item in classic_different_sample_sizes.items():
            dataframes.append(pd.DataFrame.from_dict(item))
            dataframes[-1] = dataframes[-1][[metric]]
            dataframes[-1]["condition"] = key

        total = pd.concat([df for df in dataframes])
        return total

    def get_plot(
        ax: plt.Axes,
        ordering: List[str],
        names_on_axes: List[str],
        df: pd.DataFrame,
        metric: str,
        ylim: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
    ) -> plt.Axes:
        """
        This function generates a seaborn swarmplot and boxplot on the provided axes, based on the specified metric, and adds ensemble results as horizontal lines.

        Args:
            ax (plt.Axes): Matplotlib Axes object on which to draw the plots.
            ordering (List[str]): The order of the conditions in the plot.
            names_on_axes (List[str]): The labels for the conditions on the x-axis.
            df (pd.DataFrame): The DataFrame containing values of the specified metric for different conditions.
            metric (str): The metric of interest.
            ylim (Optional[Tuple[float, float]]): The limits for the y-axis. If None, defaults will be used.
            title (Optional[str]): The title for the plot. If None, no title will be set.

        Returns:
            plt.Axes: The Axes object with the drawn plots.
        """
        sns.swarmplot(
            y=metric,
            x="condition",
            data=df,
            order=ordering,
            size=5,
            ax=ax,
            clip_on=False,
            alpha=1,
            color="black",
        )

        sns.boxplot(
            y=metric,
            x="condition",
            color="white",
            order=ordering,
            data=df,
            whis=np.inf,
            ax=ax,
            showmeans=True,
            meanline=True,
            meanprops={"color": "green", "linestyle": (0, (3, 1))},
        )
        ax.set_xticklabels(names_on_axes, rotation=0)
        ax.set(title=title, ylim=ylim, xlabel=None)

        ensemble = results_ensemble_compare_samplesize[metric]
        ensemble_c = "orange"

        margin = 0.02
        mid_point = 1 / len(ordering)

        for i in range(len(ordering)):
            ax.axhline(
                ensemble[i],
                xmin=(i) * mid_point + margin,
                xmax=(i + 1) * mid_point - margin,
                c=ensemble_c,
                zorder=10,
                clip_on=False,
            )

        return ax

    ordering = list(results_different_sample_sizes.keys())
    names_on_axes = list(results_different_sample_sizes.keys())

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize

    print("starting to plot")
    f = plt.figure(figsize=(4, 4), constrained_layout=True)
    ax = f.add_subplot(1, 1, 1)

    metric = "Accuracy"
    df = get_df(metric)
    ax = get_plot(ax, ordering, names_on_axes, df, metric, title="(a)", ylim=[0.5, 1.0])
    ax.set(ylabel="Accuracy")
    ax.set_yticks([0.50, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])

    plt.savefig(save_name + ".svg")
    plt.show()


def get_ensemble_preds_each(
    results_only_sim: Dict[str, np.ndarray],
    results_mixed_data: Dict[str, np.ndarray],
    results_only_real_data: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Function to get ensemble predictions for each unique triangle name from three different datasets.

    Args:
        results_only_sim (Dict[str, np.ndarray]): Results dictionary containing triangle names, predictions, and scores from only simulated data.
        results_mixed_data (Dict[str, np.ndarray]): Results dictionary containing triangle names, predictions, and scores from mixed data.
        results_only_real_data (Dict[str, np.ndarray]): Results dictionary containing triangle names, predictions, and scores from only real data.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary with triangle names as keys, and a dictionary of prediction scores for simulated, mixed, and real data as values.
    """
    names = np.unique(results_only_sim["triangle_names"][0])

    data = results_only_sim

    majority_vote = False

    predictions = {}
    for i, name in enumerate(names):
        predictions[name] = {"sim": 0, "real": 0, "mixed": 0}

    for i, name in enumerate(names):
        data = results_only_sim
        gathered_pred = 0
        for i_rep in range(len(data["triangle_names"])):
            # print(data['triangle_names'][i_rep])
            idx = np.where(data["triangle_names"][i_rep] == name)
            if majority_vote:
                gathered_pred += data["predictions"][i_rep][idx]
            else:
                gathered_pred += data["scores"][i_rep][idx] / 10
        predictions[name]["sim"] += gathered_pred

        data = results_mixed_data
        gathered_pred = 0
        for i_rep in range(len(data["triangle_names"])):
            # print(data['triangle_names'][i_rep])
            idx = np.where(data["triangle_names"][i_rep] == name)
            if majority_vote:
                gathered_pred += data["predictions"][i_rep][idx]
            else:
                gathered_pred += data["scores"][i_rep][idx] / 10
        predictions[name]["mixed"] += gathered_pred

        data = results_only_real_data
        gathered_pred = 0
        for i_rep in range(len(data["triangle_names"])):
            # print(data['triangle_names'][i_rep])
            idx = np.where(data["triangle_names"][i_rep] == name)
            if majority_vote:
                gathered_pred += data["predictions"][i_rep][idx]
            else:
                gathered_pred += data["scores"][i_rep][idx] / 10
        predictions[name]["real"] += gathered_pred
    return predictions


def plot_all_examples(
    predictions: Dict[str, Dict[str, float]],
    X: np.ndarray,
    y: np.ndarray,
    triangle_names: np.ndarray,
    device_names: np.ndarray,
    save_name: Optional[str] = None,
) -> None:
    """
    Function to plot all examples of predictions from different conditions.

    Args:
        predictions (Dict[str, Dict[str, float]]): Dictionary with triangle names as keys, and a dictionary of prediction scores for simulated, mixed, and real data as values.
        X (np.ndarray): Input data examples.
        y (np.ndarray): Corresponding labels for the input data examples.
        triangle_names (np.ndarray): Names of the triangles in the input data.
        device_names (np.ndarray): Names of the devices in the input data.
        save_name (Optional[str], optional): Name of the file to save the plot. If None, plot is not saved. Defaults to None.

    Returns:
        None: This function doesn't return anything; it only creates and optionally saves a plot.
    """

    def get_examples(f, gs00, gs01, this_dname, cmap="icefire"):
        index_example_pos = 0
        index_example_neg = 0

        for name in predictions:
            idx = np.where(triangle_names == name)
            example = X[idx][0]
            device_name = device_names[idx][0]
            dname = device_name_to_idx[device_name]
            true_label = y[idx][0]

            if dname == this_dname:
                if true_label:
                    gs000 = gs00[index_example_pos].subgridspec(5, 1)
                    index_example_pos += 1
                else:
                    gs000 = gs01[index_example_neg].subgridspec(5, 1)
                    index_example_neg += 1

                label_ax = f.add_subplot(gs000[0, :])
                ax1 = f.add_subplot(gs000[1:3, :])
                ax2 = f.add_subplot(gs000[3:, :])

                ax1.imshow(
                    example[0], origin="lower", cmap=cmap, aspect="auto", clim=(0, 1)
                )
                ax2.imshow(
                    example[1], origin="lower", cmap=cmap, aspect="auto", clim=(0, 1)
                )

                sns.barplot(
                    x=["S", "R", "M"],
                    y=[
                        np.array(predictions[name]["sim"])[0],
                        predictions[name]["real"][0],
                        predictions[name]["mixed"][0],
                    ],
                    ax=label_ax,
                )

                label_ax.set_yticks([0, 0.5, 1])

                ax1.grid(False)
                ax2.grid(False)
                ax1.axis("off")
                ax2.axis("off")
                label_ax.set(ylim=(0, 1))
                label_ax.set_yticklabels([])
                label_ax.set_xticklabels([])

                xlim = label_ax.get_xlim()
                label_ax.set_yticks([0, 0.5, 1])
                label_ax.set(xlim=xlim)

    device_name_to_idx = {
        "Tuor2E_chiplet_10_device_J": "i",
        "Tuor6A_chiplet_5_device_C_cooldown_1": "ii",
        "Tuor6A_chiplet_5_device_C_cooldown_2": "ii",
        "Tuor6A_chiplet_6_device_E": "iii",
        "Tuor6A_chiplet_7_device_A": "iv",
    }

    f = plt.figure(figsize=(12, 16), constrained_layout=True)

    gs = gridspec.GridSpec(10, 10, figure=f)

    gs0 = gs[1:, 1:].subgridspec(8, 2)

    top_labels_gs = gs[0, 1:].subgridspec(1, 2)
    left_labels_gs = gs[1:, 0].subgridspec(8, 1)

    label_ax = f.add_subplot(top_labels_gs[0])
    label_ax.grid(False)
    label_ax.set_xticklabels([])
    label_ax.set_yticklabels([])
    label_ax.text(0.5, 0.5, "PSB", ha="center", va="center")

    label_ax = f.add_subplot(top_labels_gs[1])
    label_ax.grid(False)
    label_ax.set_xticklabels([])
    label_ax.set_yticklabels([])
    label_ax.text(0.5, 0.5, "No PSB", ha="center", va="center")

    label_ax = f.add_subplot(left_labels_gs[0])
    label_ax.grid(False)
    label_ax.set_xticklabels([])
    label_ax.set_yticklabels([])
    label_ax.text(0.5, 0.5, "Device i", ha="center", va="center")

    label_ax = f.add_subplot(left_labels_gs[1:4])
    label_ax.grid(False)
    label_ax.set_xticklabels([])
    label_ax.set_yticklabels([])
    label_ax.text(0.5, 0.5, "Device ii", ha="center", va="center")

    label_ax = f.add_subplot(left_labels_gs[4:7])
    label_ax.grid(False)
    label_ax.set_xticklabels([])
    label_ax.set_yticklabels([])
    label_ax.text(0.5, 0.5, "Device iii", ha="center", va="center")

    label_ax = f.add_subplot(left_labels_gs[7:])
    label_ax.grid(False)
    label_ax.set_xticklabels([])
    label_ax.set_yticklabels([])
    label_ax.text(0.5, 0.5, "Device iv", ha="center", va="center")

    gs00 = gs0[0, 0].subgridspec(1, 6)
    gs01 = gs0[0, 1].subgridspec(1, 6)

    get_examples(f, gs00, gs01, "i")
    print("device i done")

    gs00 = gs0[1:4, 0].subgridspec(3, 6)
    gs01 = gs0[1:4, 1].subgridspec(3, 6)

    get_examples(f, gs00, gs01, "ii")
    print("device ii done")

    gs00 = gs0[4:7, 0].subgridspec(3, 6)
    gs01 = gs0[4:7, 1].subgridspec(3, 6)

    get_examples(f, gs00, gs01, "iii")
    print("device iii done")

    gs00 = gs0[7:, 0].subgridspec(1, 6)
    gs01 = gs0[7:, 1].subgridspec(1, 6)

    get_examples(f, gs00, gs01, "iv")
    print("device iv done")

    gs_legend = gs[0, 0].subgridspec(2, 1)

    legend_ax = f.add_subplot(gs_legend[0])

    names_on_axes = ["Simulated\ndata", "Real\ndata", "Mixed\ndata"]
    sns.barplot(ax=legend_ax, x=names_on_axes, y=[1, 1, 1])
    xlim = legend_ax.get_xlim()

    plt.setp(legend_ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(legend_ax.get_yticklabels(), rotation=0, fontsize=6)
    legend_ax.set_ylabel("average\nprediction", fontsize=7)
    legend_ax.set_yticks([0, 0.5, 1])

    legend_ax.set_yticks([0, 0.5, 1])

    legend_ax.set(ylim=(0, 1), xlim=xlim)
    # plt.savefig('all_examples.pdf')
    plt.savefig(save_name + ".svg", dpi=1000)

    plt.show()
