import numpy as np

from failure_gallery import iou_matrix, match


def test_iou_identical_and_disjoint():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]], float)
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 0.0, 50 / 150], atol=1e-6)


def test_match_counts_tp_fp_and_misses():
    gt = [[0, 0, 10, 10], [50, 50, 60, 60]]
    gt_cls = [2, 1]
    preds = [[1, 1, 10, 10], [0, 0, 10, 10], [100, 100, 110, 110]]
    pred_cls = [2, 2, 3]
    conf = [0.9, 0.8, 0.7]
    tp, fp, missed = match(gt, gt_cls, preds, pred_cls, conf)
    # best-confidence box claims the helmet; the duplicate and the stray vest are FPs; the head is missed
    assert tp.tolist() == [True, False, False]
    assert fp.tolist() == [False, True, True]
    assert missed.tolist() == [False, True]


def test_class_mismatch_is_not_a_match():
    tp, fp, missed = match([[0, 0, 10, 10]], [1], [[0, 0, 10, 10]], [2], [0.9])
    assert tp.tolist() == [False] and missed.tolist() == [True]


def test_empty_inputs():
    tp, fp, missed = match([], [], [], [], [])
    assert len(tp) == len(fp) == len(missed) == 0
