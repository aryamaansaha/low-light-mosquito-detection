import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import torch
import os
import numpy as np
from torchvision.ops import box_iou
from tqdm import tqdm

def plot_predictions(model, dataloader, device, num_samples=9, grid_size=(3, 3), confidence_threshold=0.5, save_folder=None, dpi=300):
    """
    Plot predicted and ground truth bounding boxes and labels for samples from the validation dataset in a grid layout.

    Args:
        model: Trained Faster R-CNN model.
        dataloader: DataLoader for validation data.
        device: Device for computation (e.g., "cuda" or "cpu").
        class_labels: List of class labels corresponding to class IDs.
        num_samples: Total number of samples to visualize.
        grid_size: Tuple defining the grid layout (rows, cols).
        confidence_threshold: Minimum confidence score for displaying predictions.
        save_folder: Folder to save the plot.
        dpi: Resolution for saving the plot.
    """

    class_labels = ["background", 'aegypti', 'albopictus', 'anopheles', 'culex', 'culiseta', 'japonicus/koreicus']
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
    model.eval()  # Set model to evaluation mode
    rows, cols = grid_size
    total_images = min(num_samples, rows * cols)  # Total images to display

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()

    with torch.no_grad():
        for idx in range(total_images):
            ax = axes[idx]

            img, target = dataloader.dataset[idx]
            img = img.to(device).unsqueeze(0)  # Add batch dimension
            output = model(img)[0]  # Model returns a list of outputs

            # Extract predictions
            pred_boxes = output["boxes"].cpu().numpy()
            pred_labels = output["labels"].cpu().numpy()
            pred_scores = output["scores"].cpu().numpy()

            # Extract ground truth
            true_boxes = target["boxes"].cpu().numpy()
            true_labels = target["labels"].cpu().numpy()

            # Convert the image tensor to a PIL image for plotting
            img = to_pil_image(img.squeeze(0).cpu())

            ax.imshow(img)
            ax.axis("off")

            # Plot ground truth bounding boxes
            for box, label in zip(true_boxes, true_labels):
                x_min, y_min, x_max, y_max = box

                # Draw the bounding box (green for ground truth)
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                     edgecolor="green", facecolor="none", linewidth=1)
                ax.add_patch(rect)

                # Add the label
                label_text = f"True: {class_labels[label]}"
                ax.text(x_min, y_min - 5, label_text, color="green", fontsize=15, 
                        backgroundcolor="white", bbox=dict(facecolor="white", alpha=0.5))

            # Plot predicted bounding boxes
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                if score > confidence_threshold:  # Plot only confident predictions
                    x_min, y_min, x_max, y_max = box

                    # Draw the bounding box (red for predictions)
                    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                         edgecolor="red", facecolor="none", linewidth=1, linestyle="--")
                    ax.add_patch(rect)

                    # Add the label and score
                    label_text = f"Pred: {class_labels[label]} ({score:.2f})"
                    ax.text(x_min, y_max + 5, label_text, color="red", fontsize=15, 
                            backgroundcolor="white", bbox=dict(facecolor="white", alpha=0.5))

            # ax.set_title(f"Sample {idx + 1}", fontsize=10)

        for idx in range(total_images, len(axes)):
            axes[idx].axis("off")

    if save_folder:
        save_path = os.path.join(save_folder, f"predictions_vs_ground_truth.png")
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved at: {save_path}")
    plt.tight_layout(pad=0.1, w_pad=0.05, h_pad=0.05)
    plt.show()



def calculate_f1_score(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels, iou_threshold=0.5, score_threshold=0.5):
    """
    Calculate F1 score for a single image's predictions.

    Args:
        pred_boxes: Predicted bounding boxes (N x 4, numpy array).
        pred_labels: Predicted class labels (N, numpy array).
        pred_scores: Predicted confidence scores (N, numpy array).
        true_boxes: Ground truth bounding boxes (M x 4, numpy array).
        true_labels: Ground truth class labels (M, numpy array).
        iou_threshold: IoU threshold for matching predictions to ground truth.
        score_threshold: Confidence score threshold for valid predictions.

    Returns:
        tp: True positives.
        fp: False positives.
        fn: False negatives.
    """
    # Filter predictions based on confidence score
    valid_preds = pred_scores > score_threshold
    pred_boxes = pred_boxes[valid_preds]
    pred_labels = pred_labels[valid_preds]

    tp = 0
    fp = 0
    fn = 0

    if len(pred_boxes) > 0 and len(true_boxes) > 0:
        ious = box_iou(torch.tensor(pred_boxes), torch.tensor(true_boxes))
        max_ious, max_indices = ious.max(dim=1)

        for pred_idx, max_iou in enumerate(max_ious):
            if max_iou > iou_threshold and pred_labels[pred_idx] == true_labels[max_indices[pred_idx]]:
                tp += 1
            else:
                fp += 1

        # Count false negatives (unmatched ground truth boxes)
        matched_gt = max_ious > iou_threshold
        fn = len(true_boxes) - matched_gt.sum().item()
    else:
        # All predictions are false positives if no ground truth exists
        fp = len(pred_boxes)
        # All ground truth boxes are false negatives if no predictions exist
        fn = len(true_boxes)

    return tp, fp, fn


def calculate_map(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels, iou_threshold=0.5):
    """
    Calculate Mean Average Precision (mAP) for a single image's predictions.

    Args:
        pred_boxes: Predicted bounding boxes (N x 4, numpy array).
        pred_labels: Predicted class labels (N, numpy array).
        pred_scores: Predicted confidence scores (N, numpy array).
        true_boxes: Ground truth bounding boxes (M x 4, numpy array).
        true_labels: Ground truth class labels (M, numpy array).
        iou_threshold: IoU threshold for matching predictions to ground truth.

    Returns:
        Average precision for this image.
    """
    if len(pred_boxes) == 0 or len(true_boxes) == 0:
        return 0.0

    # Sort predictions by confidence score (descending)
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_indices]
    pred_labels = pred_labels[sorted_indices]

    ious = box_iou(torch.tensor(pred_boxes), torch.tensor(true_boxes))

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    matched_gt = np.zeros(len(true_boxes), dtype=bool)

    for i, box in enumerate(pred_boxes):
        if len(true_boxes) == 0:
            fp[i] = 1
            continue

        # Find the highest IoU for this prediction
        max_iou, max_idx = ious[i].max(0)
        if max_iou >= iou_threshold and not matched_gt[max_idx] and pred_labels[i] == true_labels[max_idx]:
            tp[i] = 1
            matched_gt[max_idx] = True
        else:
            fp[i] = 1

    # Cumulative sums for precision/recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = tp_cumsum / len(true_boxes)

    # Average Precision (AP)
    ap = 0.0
    for t in np.linspace(0, 1, 11):  # Standard mAP evaluation at 11 points
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11

    return ap

def get_metrics(model, dataloader, device, iou_threshold=0.5, score_threshold=0.5):
    """
    Evaluate F1 Score and Mean Average Precision (mAP) for the validation dataset.

    Args:
        model: Trained Faster R-CNN model.
        dataloader: DataLoader for validation data.
        device: Device for computation (e.g., "cuda" or "cpu").
        iou_threshold: IoU threshold for matching predictions to ground truth.
        score_threshold: Confidence score threshold for valid predictions.

    Returns:
        f1_score: Overall F1 Score for the dataset.
        mean_ap: Mean Average Precision (mAP) for the dataset.
    """
    model.eval()

    all_tp, all_fp, all_fn = 0, 0, 0
    all_aps = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get model predictions
            outputs = model(images)

            for output, target in zip(outputs, targets):
                # Predicted values
                pred_boxes = output["boxes"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()

                # Ground truth values
                true_boxes = target["boxes"].cpu().numpy()
                true_labels = target["labels"].cpu().numpy()

                # Calculate F1 components
                tp, fp, fn = calculate_f1_score(pred_boxes, pred_labels, pred_scores, 
                                                true_boxes, true_labels, 
                                                iou_threshold, score_threshold)
                all_tp += tp
                all_fp += fp
                all_fn += fn

                # Calculate AP for this image
                ap = calculate_map(pred_boxes, pred_labels, pred_scores, 
                                   true_boxes, true_labels, 
                                   iou_threshold)
                all_aps.append(ap)

    # F1 Score Calculation
    precision = all_tp / (all_tp + all_fp + 1e-6)
    recall = all_tp / (all_tp + all_fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    # Mean Average Precision
    mean_ap = np.mean(all_aps)

    return f1_score, mean_ap, (f1_score + mean_ap) / 2, precision, recall
