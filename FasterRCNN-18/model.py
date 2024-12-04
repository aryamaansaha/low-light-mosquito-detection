import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def build_model(num_classes):
    # pre-trained ResNet-18 backbone
    backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])  # Removing the FC and avg pool layers
    backbone.out_channels = 512  # Output channels of ResNet-18

    # RPN anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    # ROI pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"], output_size=7, sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )
    return model
