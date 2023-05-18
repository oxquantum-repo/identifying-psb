from torch.utils.data import Dataset
import torchvision
import torch
from torch import nn, optim
from typing import Optional, List, Union, Dict, Callable, Any, Tuple

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    balanced_accuracy_score,
)

import numpy as np

import matplotlib.pyplot as plt


class Triangles(torch.utils.data.Dataset):
    def __init__(
        self, imgs: np.ndarray, labels: np.ndarray, transform: Optional[Callable] = None
    ) -> None:
        """
        A custom PyTorch Dataset for storing images and their labels.

        Args:
            imgs (np.ndarray): Array of images.
            labels (np.ndarray): Corresponding labels for images.
            transform (Callable, optional): Optional transform to be applied on a sample.
        """
        self.imgs = imgs
        self.labels = labels

        self.transform = transform

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(
        self, idx: Union[int, torch.Tensor]
    ) -> Dict[str, Union[np.ndarray, int]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.imgs[idx]
        label = self.labels[idx]

        sample = {"image": img, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample


def build_batches(x: np.ndarray, n: int) -> np.ndarray:
    """
    Builds batches of given size from an input array.

    Args:
        x (np.ndarray): Input array.
        n (int): Size of the batch.

    Returns:
        np.ndarray: Array reshaped into batches.
    """
    x = np.asarray(x)
    m = (x.shape[0] // n) * n
    return x[:m].reshape(-1, n, *x.shape[1:])


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        """
        Initialize the LeNet5 model for image classification. Assumes 100x100 pixel images.

        Args:
            num_classes (int, optional): The number of output classes. Defaults to 2.
        """
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(7744, 120)  # assumes input of 100x100 images
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the LeNet5 model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


def get_net_optimiser_scheduler_criterion(
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
    model_type: str = "resnet18",
) -> Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler.ReduceLROnPlateau, nn.Module]:
    """
    Gets the network, optimizer, scheduler, and criterion based on the specified parameters.

    Args:
        device (torch.device): The device to which the model should be sent.
        class_weights (torch.Tensor, optional): A tensor of class weights. If None, all classes are assumed to have equal weight.
        model_type (str, optional): The type of the model to use. Can be 'resnet18' or 'lenet'. Defaults to 'resnet18'.

    Returns:
        Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler.ReduceLROnPlateau, nn.Module]: The model, optimizer, scheduler, and criterion.
    """
    if model_type == "resnet18":
        net = torchvision.models.resnet18(pretrained=False)
        # net = torchvision.models.resnet34(pretrained=False)
        net.conv1 = nn.Conv2d(
            2,
            net.conv1.out_channels,
            net.conv1.kernel_size,
            net.conv1.stride,
            net.conv1.padding,
            bias=net.conv1.bias,
        )

        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 2)
    elif model_type == "lenet":
        net = LeNet5()
    else:
        raise (f"model_type {model_type} not considered")
    net = net.to(device)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    return net, optimizer, scheduler, criterion


def get_results_dict() -> Dict[str, List[Any]]:
    """
    Initializes a dictionary to store the results of the model.

    Returns:
        Dict[str, List[Any]]: A dictionary with keys for various performance metrics and values as empty lists.
    """
    d = {
        "confusion matrix": [],
        "TPR": [],
        "TNR": [],
        "Accuracy": [],
        "Balanced Accuracy": [],
        "AUC": [],
        "ROC Curve": [],
        "true_labels": [],
        "predictions": [],
        "scores": [],
        "triangle_names": [],
        "device_names": [],
    }
    return d


def record_results(
    results: Dict[str, List[Any]],
    predicted: np.ndarray,
    scores: np.ndarray,
    true_labels: np.ndarray,
    device_names: Optional[np.ndarray] = None,
    triangle_names: Optional[np.ndarray] = None,
    mode: str = "classic",
    latest_basel_idx: Optional[int] = None,
) -> Dict[str, List[Any]]:
    """
    Records the results of the model in the results dictionary.
    
    'classic' mode is the one used in the paper. It weighs each sample the same while the 
    alternative gives each fold the same weight, leading to different weights of the individual
    samples due to differences in the fold size.

    Args:
        results (Dict[str, List[Any]]): The results dictionary.
        predicted (np.ndarray): The predicted labels.
        scores (np.ndarray): The prediction scores.
        true_labels (np.ndarray): The true labels.
        device_names (np.ndarray, optional): The names of the devices. Defaults to None.
        triangle_names (np.ndarray, optional): The names of the triangles. Defaults to None.
        mode (str, optional): The mode to use when recording results. Can be 'classic' or 'other'. Defaults to 'classic'.
        latest_basel_idx (int, optional): The latest basel index. Defaults to None.

    Returns:
        Dict[str, List[Any]]: The updated results dictionary.
    """
    if mode == "classic":
        results["triangle_names"].append(triangle_names)
        results["device_names"].append(device_names)

        cm = confusion_matrix(true_labels, predicted, labels=[0, 1])

        results["confusion matrix"].append(cm)
        tn, fp, fn, tp = cm.ravel()

        tnr = tn / (tn + fp)
        tpr = tp / (tp + fn)
        results["Accuracy"].append((tp + tn) / (tn + fp + fn + tp))
        results["Balanced Accuracy"].append(
            balanced_accuracy_score(true_labels, predicted)
        )
        results["TPR"].append(tpr)
        results["TNR"].append(tnr)

        fpr, tpr, thresholds = roc_curve(true_labels, scores)

        results["scores"].append(scores)
        results["ROC Curve"].append([fpr, tpr])

        AUC = roc_auc_score(true_labels, scores)

        results["AUC"].append(AUC)

        results["predictions"].append(predicted)
        results["true_labels"].append(true_labels)

        return results
    else:
        results["triangle_names"].append(triangle_names)
        results["predictions"].append(predicted)
        results["scores"].append(scores)
        results["device_names"].append(device_names)
        results["true_labels"].append(true_labels)

        # weigh each device equally
        test_devices = [
            "Tuor6A_chiplet_5_device_C",
            "Tuor6A_chiplet_6_device_E",
            "Tuor6A_chiplet_7_device_A",
        ]

        _tnr = []
        _tpr = []
        _accuracy = []
        _balanced_accuracy = []
        _roc_curve = []
        _auc = []
        _cms = {}

        for test_device_name in test_devices:
            print("testing", test_device_name)
            if test_device_name == "Tuor6A_chiplet_5_device_C":
                test_index = np.logical_or(
                    device_names == "Tuor6A_chiplet_5_device_C_cooldown_1",
                    device_names == "Tuor6A_chiplet_5_device_C_cooldown_2",
                )
            else:
                test_index = device_names == test_device_name

            cm = confusion_matrix(
                true_labels[test_index], predicted[test_index], labels=[0, 1]
            )

            _cms[test_device_name] = cm
            tn, fp, fn, tp = cm.ravel()
            _accuracy.append((tp + tn) / (tn + fp + fn + tp))
            _balanced_accuracy.append(
                balanced_accuracy_score(true_labels[test_index], predicted[test_index])
            )

            _tnr.append(tn / (tn + fp))
            _tpr.append(tp / (tp + fn))

            _roc_curve.append(roc_curve(true_labels[test_index], scores[test_index]))
            # fpr, tpr, thresholds

            _auc.append(roc_auc_score(true_labels[test_index], scores[test_index]))

        results["confusion matrix"].append(_cms)
        results["Accuracy"].append(np.mean(_accuracy))
        results["Balanced Accuracy"].append(np.mean(_balanced_accuracy))
        results["TPR"].append(np.mean(_tpr))
        results["TNR"].append(np.mean(_tnr))
        results["AUC"].append(np.mean(_auc))

        tprs = []
        # aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        for i, (fpr, tpr, thresholds) in enumerate(_roc_curve):
            # mean_fpr = np.linspace(0, 1, 1000)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        results["ROC Curve"].append([mean_fpr, mean_tpr])

        return results


def report_results(results: Dict[str, List[Any]]) -> None:
    """
    Prints and plots the results of the model.

    Args:
        results (Dict[str, List[Any]]): The results dictionary.
    """
    print("Confusion matrix: ", results["confusion matrix"][-1])
    print("TPR and TNR:", results["TPR"][-1], results["TNR"][-1])
    print("Accuracy:", results["Accuracy"][-1])
    print("AUC:", results["AUC"][-1])
    print("")
    print("Mean/Std AUC", np.mean(results["AUC"]), np.std(results["AUC"]))
    print(
        "Mean/Std accuracy", np.mean(results["Accuracy"]), np.std(results["Accuracy"])
    )
    print(
        "Mean/Std balanced accuracy",
        np.mean(results["Balanced Accuracy"]),
        np.std(results["Balanced Accuracy"]),
    )

    plt.plot(results["Accuracy"], label="Accuracy", color="tab:red")
    plt.plot(results["Balanced Accuracy"], label="Balanced Accuracy", color="tab:pink")
    plt.plot(results["TPR"], label="TPR", color="tab:green")
    plt.plot(results["TNR"], label="TNR", color="tab:blue")

    plt.plot(results["AUC"], label="AUC", color="tab:orange", linestyle="-.")

    plt.legend()
    plt.show()

    [fpr, tpr] = results["ROC Curve"][-1]
    plt.plot(fpr, tpr, color="tab:green", label="latest run")

    if len(results["ROC Curve"]) > 1:
        for [fpr, tpr] in results["ROC Curve"][:-1]:
            plt.plot(fpr, tpr, color="tab:grey", alpha=0.1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()
