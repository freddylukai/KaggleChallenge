import numpy as np

'''
    prediction and ground_truth are bounding boxes
    represented as arrays in the form (x, y, width, height).

    This returns the intersection area divided by the 
    union area.
'''


def intersection_over_union(prediction, ground_truth):
    pred_x1, pred_y1, pred_width, pred_height = prediction
    gt_x1, gt_y1, gt_width, gt_height = ground_truth

    # Opposite corners
    pred_x2 = pred_x1 + pred_width
    pred_y2 = pred_y1 + pred_height
    gt_x2 = gt_x1 + gt_width
    gt_y2 = gt_y1 + gt_height

    # Intersection points
    int_x1 = max(pred_x1, gt_x1)
    int_y1 = max(pred_y1, gt_y1)
    int_x2 = min(pred_x2, gt_x2)
    int_y2 = min(pred_y2, gt_y2)

    if int_x2 > int_x1 or int_y2 > int_y1:
        # Sum of both areas
        inion_area = (pred_width * pred_height) + (gt_height * gt_width)

        # intersection width and height multiplied
        int_area = (int_x2 - int_x1) * (int_y2 - int_y1)

        # substract intersection to get total area
        return int_area / (inion_area - int_area)

    else:
        return 0.0


'''
    predictions -> array of shape (n_predictions, x, y, width, height)
    confidence -> array of shape (n_predictions,)
    ground_truth -> array of shape (n_boxes, x, y, width, height)
'''


def rsna_metric(predictions, confidences, ground_truth, thresholds=(0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75)):
    assert (len(predictions) == len(confidences))

    if len(ground_truth) == 0 and len(predictions) == 0:
        return 1.0  # Special case to avoid division by zero

    total = 0.0

    for t in thresholds:
        matched_ground_truth = set()
        true_positives = 0.0
        false_positives = 0.0

        for i, gt in enumerate(ground_truth):
            for j in np.argsort(confidences):
                pred_box = predictions[j]
                if intersection_over_union(pred_box, gt) >= t:
                    # Uses the first match sorted by confidence as TP
                    matched_ground_truth.add(i)
                    true_positives += 1
                    break
            else:
                # no match found, is a FP
                false_positives += 1

        # False negatives are ground_truth objects that were not matched
        false_negatives = len(ground_truth) - len(matched_ground_truth)
        total += true_positives / (true_positives + false_positives + false_negatives)

    # Average over all thresholds
    return total / len(thresholds)


'''
    predictions -> List of arrays with shape (n_predictions, x, y, width, height)
    confidence ->  List of arrays with shape (n_predictions,)
    ground_truth -> List of arrays with shape (n_boxes, x, y, width, height)

    predictions, confidence, and ground_truth must have the same length.

    Each element on the list represents the predictions/ground_truth of a single image.
'''


def mean_rsna_metric(predictions, confidences, ground_truth, thresholds=(0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75)):
    assert (len(predictions) == len(confidences) and len(confidences) == len(ground_truth))
    return sum((rsna_metric(predictions[i], confidences[i], ground_truth[i], thresholds) for i in
                range(len(predictions)))) / len(predictions)
