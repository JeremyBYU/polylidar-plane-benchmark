import logging
import numpy as np

logger = logging.getLogger('PPB')
SYNPEB_VALID_INDICES = 10

np.set_printoptions(threshold=3600, linewidth=250, precision=2, suppress=True)


def evaluate(gt_image, planes_ms, tcomp=0.70):
    # Note when I use the word class/label it means a unique plane
    # This creates a flattened array of each pixels class
    gt_pixel_labels = gt_image[:, :, 3].flatten().astype(np.int)

    # This returns the the number of unique classes and their counts (# pixels with such a class)
    gt_unique_labels, counts = np.unique(gt_pixel_labels, return_counts=True)
    # print(gt_unique_labels)

    # This is a list of tuples, each tuple having a numpy array and a integer label id.
    # Each numpy array represents a unique plane and is a set of point INDICES
    # This list only accepts ground truth planes whose label greater than 10
    # https://github.com/acschaefer/ppe/blob/8804bba91debd1917188b509246b64b5e1401d87/matlab/experiments/segcompeval.m#L16
    gt_planes_filtered = [(np.flatnonzero(gt_pixel_labels == gt_unique_label), gt_unique_label) for gt_unique_label in
                          gt_unique_labels if gt_unique_label >= SYNPEB_VALID_INDICES]

    n_gt = len(gt_planes_filtered)  # num of gt planes
    n_ms = len(planes_ms)           # num of machine segmented planes

    # filtered ground truth labels, only used at end for debugging
    gt_unique_labels_filtered = np.array(
        [gt_unique_label for gt_unique_label in gt_unique_labels if gt_unique_label >= SYNPEB_VALID_INDICES], dtype=np.int)

    # Number of points in each ground truth plane
    point_count_gt = np.array([count for gt_unique_label, count in zip(
        gt_unique_labels, counts) if gt_unique_label >= SYNPEB_VALID_INDICES])
    # Number of points in each machine segmented plane
    point_count_ms = np.array([plane_ms['point_indices'].size for plane_ms in planes_ms])
    # and n_gtXn_ms matrix to store the pixel overlap count
    overlap = np.zeros((n_gt, n_ms), dtype=np.int)

    logging.info("There are %d unique gt planes and %d machine segmented planes", n_gt, n_ms)

    # print(n_gt)
    # print(n_ms)
    # print(point_count_gt, np.sum(point_count_gt))
    # print(point_count_ms, np.sum(point_count_ms))

    for gt_index, (gt_point_idx_set, gt_unique_label) in enumerate(gt_planes_filtered):
        # print(gt_unique_label)
        for ms_index, plane_ms in enumerate(planes_ms):
            ms_point_idx_set = plane_ms['point_indices']
            # bad_indices = np.setdiff1d(ms_point_idx_set, gt_point_idx_set)
            # if (gt_unique_label == 13) and (ms_index) == 34:
            #     print("bad indices")
            #     print(repr(bad_indices))
            n_overlap = np.intersect1d(gt_point_idx_set, ms_point_idx_set).size
            overlap[gt_index, ms_index] = n_overlap

    # print(overlap)
    # print(point_count_ms)
    overlap_fraction_gt = overlap / point_count_gt[:, None]
    overlap_fraction_ms = (overlap.T / point_count_ms[:, None]).T
    print(overlap_fraction_ms)
    print(overlap_fraction_gt)
    # import ipdb; ipdb.set_trace()

    # which ROWS of overlap_fraction_ms have at least 2 columns greater than tcomp (0.80)
    # should be a row vector of length n_gt with 0/1 indicating if this ground truth plane is undersegmented
    fraction_overseg_mask = (np.sum(overlap_fraction_ms > tcomp, axis=1) >= 2).astype(np.float64)
    # of the rows having this condition, and of columns greater than tcomp, sum up the column in ovlerap_fraction_gt
    fraction_overseg_union = np.sum((overlap_fraction_ms > tcomp) * overlap_fraction_gt, axis=1) * fraction_overseg_mask

    fraction_underseg_mask = (np.sum(overlap_fraction_gt > tcomp, axis=0) >= 2).astype(np.float64)
    fraction_underseg_union = np.sum((overlap_fraction_gt > tcomp) *
                                     overlap_fraction_ms, axis=0) * fraction_underseg_mask

    # a boolean matrix indicating which ms/gt planes were correctly segmented
    correct_seg = (overlap_fraction_gt > tcomp) & (overlap_fraction_ms > tcomp)
    # a boolean row mask indicating which row in gt is over segmented
    over_seg = fraction_overseg_union > tcomp
    # a boolean column mask indicating which columns in ms are under segmenting
    under_seg = fraction_underseg_union > tcomp

    # some rows and column may be participating in these three classifcations
    # we must exclusify them by first calculating average fractions
    avg_correct_seg = (overlap_fraction_gt * correct_seg + overlap_fraction_ms * correct_seg) / 2

    avg_over_seg_ms = np.sum(overlap_fraction_ms * (overlap_fraction_ms > tcomp), 1)
    avg_over_seg_gt = fraction_overseg_union * over_seg
    avg_over_seg = (avg_over_seg_ms + avg_over_seg_gt) / (np.sum(overlap_fraction_ms > tcomp, 1) + 1) * over_seg

    avg_under_seg_gt = np.sum(overlap_fraction_gt * (overlap_fraction_gt > tcomp), 0)
    avg_under_seg_ms = fraction_underseg_union * under_seg
    avg_under_seg = (avg_under_seg_ms + avg_under_seg_gt) / (np.sum(overlap_fraction_gt > tcomp, 0) + 1) * under_seg

    # avg_over_seg_cause = sum((fms>tcomp) .* overseg .* fms, 1);
    avg_over_seg_cause_temp = (overlap_fraction_ms > tcomp) * overlap_fraction_ms * over_seg[:, None]
    avg_over_seg_cause = np.sum(avg_over_seg_cause_temp, 0)

    avg_under_seg_cause_temp = (((overlap_fraction_gt > tcomp) * overlap_fraction_gt).T * under_seg[:, None]).T
    avg_under_seg_cause = np.sum(avg_under_seg_cause_temp, 1)

    # Exclusify
    correct_seg_final = (avg_correct_seg > avg_over_seg[:, None]) & ((avg_correct_seg.T > avg_under_seg[:, None]).T)
    over_seg_final = (avg_over_seg > np.sum(avg_correct_seg, 1)) & (avg_over_seg > avg_under_seg_cause)
    under_seg_final = (avg_under_seg > np.sum(avg_correct_seg, 0)) & (avg_under_seg > avg_over_seg_cause)

    # Get Remaining Planes
    over_seg_cause = np.sum((overlap_fraction_ms > tcomp) * over_seg[:, None], 0)
    under_seg_cause = np.sum(((overlap_fraction_gt > tcomp).T * under_seg[:, None]).T, 1)

    missed_seg = 1 - (np.sum(correct_seg_final, 1) | over_seg_final | under_seg_cause)
    noise_seg = 1 - (np.sum(correct_seg_final, 0) | under_seg_final | over_seg_cause)

    # Calculate Metrics
    n_ms_all = n_ms
    n_corr_seg = np.sum(np.sum(correct_seg_final))
    n_over_seg = np.asscalar(np.sum(over_seg_final))
    n_under_seg = np.asscalar(np.sum(under_seg_final))
    n_missed_seg = np.asscalar(np.sum(missed_seg))
    n_noise_seg = np.asscalar(np.sum(noise_seg))
    logger.info("n_corr: %d; n_over_seg: %d; n_under_seg: %d; n_missed_seg: %d; n_noise_seg: %d",
                n_corr_seg, n_over_seg, n_under_seg, n_missed_seg, n_noise_seg)

    # gt and ms plane ids for each category for debugging purposes
    gt_labels_missed = gt_unique_labels_filtered[np.ma.make_mask(missed_seg)]
    ms_labels_noise = np.where(np.ma.make_mask(noise_seg))[0]
    gt_labels_over_seg = gt_unique_labels_filtered[np.ma.make_mask(over_seg_final)]
    ms_labels_under_seg = np.where(np.ma.make_mask(under_seg_final))[0]

    results = dict(n_ms_all=n_ms_all, n_corr_seg=n_corr_seg, n_over_seg=n_over_seg, n_under_seg=n_under_seg, n_missed_seg=n_missed_seg, n_noise_seg=n_noise_seg)
    auxiliary = dict(gt_labels_missed=gt_labels_missed, ms_labels_noise=ms_labels_noise, gt_labels_over_seg=gt_labels_over_seg, ms_labels_under_seg=ms_labels_under_seg)

    return results, auxiliary
