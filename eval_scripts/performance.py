import sklearn.metrics
import numpy as np


def ptsort(tu):
    return tu[0]


def AUPRC(pts):
    true_labels = [int(x[1]) for x in pts]
    predicted_probs = [x[0] for x in pts]
    return sklearn.metrics.average_precision_score(true_labels, predicted_probs)


def AUROC(truth, scores, average="macro", multi_class="ovr"):
    truth_np = truth.cpu().numpy() if hasattr(truth, "cpu") else np.asarray(truth)
    scores_np = scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)

    if scores_np.ndim == 1:
        return sklearn.metrics.roc_auc_score(truth_np, scores_np)

    if scores_np.ndim != 2:
        raise ValueError(f"Expected AUROC scores to have 1 or 2 dims, got shape {scores_np.shape}.")

    num_classes = scores_np.shape[1]
    if num_classes == 1:
        return sklearn.metrics.roc_auc_score(truth_np, scores_np[:, 0])
    if num_classes == 2:
        return sklearn.metrics.roc_auc_score(truth_np, scores_np[:, 1])

    per_class_scores = []
    for class_idx in range(num_classes):
        binary_truth = (truth_np == class_idx).astype(np.int32)
        if np.unique(binary_truth).size < 2:
            continue
        class_score = sklearn.metrics.roc_auc_score(binary_truth, scores_np[:, class_idx])
        per_class_scores.append(class_score)

    if not per_class_scores:
        raise ValueError("AUROC is undefined because the labels contain fewer than two distinguishable classes.")

    if average == "macro":
        return float(np.mean(per_class_scores))

    return sklearn.metrics.roc_auc_score(truth_np, scores_np, average=average, multi_class=multi_class)


def f1_score(truth, pred, average):
    return sklearn.metrics.f1_score(truth.cpu().numpy(), pred.cpu().numpy(), average=average)


def accuracy(truth, pred):
    return sklearn.metrics.accuracy_score(truth.cpu().numpy(), pred.cpu().numpy())


def eval_affect(truths, results, exclude_zero=True):
    if type(results) is np.ndarray:
        test_preds = results
        test_truth = truths
    else:
        test_preds = results.cpu().numpy()
        test_truth = truths.cpu().numpy()

    non_zeros = np.array([i for i, e in enumerate(
        test_truth) if e != 0 or (not exclude_zero)])

    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    return sklearn.metrics.accuracy_score(binary_truth, binary_preds)
