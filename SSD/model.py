from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead, SSDRegressionHead
import torch

def build_SSD(num_classes):
    # pre-trained SSD model
    model = ssd300_vgg16(weights="DEFAULT")

    # dummy input to pass through the backbone
    dummy_input = torch.rand(1, 3, 300, 300)  # Batch size 1, RGB image, 300x300 resolution
    feature_maps = model.backbone(dummy_input)

    # Determining the output channels for each feature map
    out_channels = [feature.shape[1] for feature in feature_maps.values()]

    # Generating the number of anchors per location
    num_anchors = model.anchor_generator.num_anchors_per_location()

    # Replacing the classification head with updated dimensions
    model.head.classification_head = SSDClassificationHead(out_channels, num_anchors, num_classes)

    # Replacing the regression head (bbox prediction)
    model.head.regression_head = SSDRegressionHead(out_channels, num_anchors)

    return model


