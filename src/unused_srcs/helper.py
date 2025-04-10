import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms


from resnet import resnet18, resnet34, resnet50

def get_model(arch, bins, pretrained=False, inference_mode=False):
    """Return the model based on the specified architecture."""
    if arch == 'resnet18':
        model = resnet18(pretrained=pretrained, num_classes=bins)
    elif arch == 'resnet34':
        model = resnet34(pretrained=pretrained, num_classes=bins)
    elif arch == 'resnet50':
        model = resnet50(pretrained=pretrained, num_classes=bins)
    # elif arch == "mobilenetv2":
    #     model = mobilenet_v2(pretrained=pretrained, num_classes=bins)
    # elif arch == "mobileone_s0":
    #     model = mobileone_s0(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    # elif arch == "mobileone_s1":
    #     model = mobileone_s1(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    # elif arch == "mobileone_s2":
    #     model = mobileone_s2(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    # elif arch == "mobileone_s3":
    #     model = mobileone_s3(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    # elif arch == "mobileone_s4":
    #     model = mobileone_s4(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    else:
        raise ValueError(f"Please choose available model architecture, currently chosen: {arch}")
    return model


def angular_error(gaze_vector, label_vector):
    dot_product = np.dot(gaze_vector, label_vector)
    norm_product = np.linalg.norm(gaze_vector) * np.linalg.norm(label_vector)
    cosine_similarity = min(dot_product / norm_product, 0.9999999)

    return np.degrees(np.arccos(cosine_similarity))


def gaze_to_3d(gaze):
    yaw = gaze[0]   # Horizontal angle
    pitch = gaze[1]  # Vertical angle

    gaze_vector = np.zeros(3)
    gaze_vector[0] = -np.cos(pitch) * np.sin(yaw)
    gaze_vector[1] = -np.sin(pitch)
    gaze_vector[2] = -np.cos(pitch) * np.cos(yaw)

    return gaze_vector

