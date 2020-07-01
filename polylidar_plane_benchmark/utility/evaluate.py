"""This code contains evaluation metrics for the benchmark
Only the function `evaluate` should be used.
"""
import logging
import numpy as np

from .plane_fit import fit_plane_and_get_rmse

logger = logging.getLogger('PPB')
# Any ground truth label < SYNPEB_VALID_INDICES is removed from ground truth set
# This number comes from SYNPEB evaluation source code (I believe it matches segcomp as well)
# I have verified that SYNPEB ground truth labels less than 10 are noise/spurious
SYNPEB_VALID_INDICES = 10
# NOTE - I also noticed that some files will have **more** ground truth labels
#       between different variances (same point cloud scene). Example synpeb/test/var1/pc_05.pcd vs. synpeb/test/var2/pc_05.pcd
#       Ground truth label 61 is missing in **var1**
#       I don't know why this is the case. It doesn't really effect evaluation

np.set_printoptions(threshold=3600, linewidth=350, precision=6, suppress=True)


# NOTE - Do not use this function, use `evaluate` below. Left for posterity
def evaluate_incorrect(gt_image, planes_ms, tcomp=0.80, misc=''):
    """Reports evaluation metrics of ground truth vs machine segmented planes
    The key metrics are
        n_corr_seg = # of Correctly segmented ground truth planes
        n_over_seg = # of oversegmented ground truth planes
        n_under_seg = # of machine segmented planes covering multiple ground planes
        n_missed_seg = # of ground truth planes that don't participate in any of the classifications
        n_noise_seg = # of of machine segmented planes that are just noise
    
    This implementation is translated from matlab from here:
    https://github.com/acschaefer/ppe/blob/8804bba91debd1917188b509246b64b5e1401d87/matlab/experiments/segcompeval.m
    I believe this implementation has a small bug, please see `evaluate` function below for more details

    Arguments:
        gt_image {ndarray} -- A numpy array of N X N X 4. The last dimension is (x,y,z,class)
        planes_ms {List[MachineSegmentedPlanes]} -- A list of machine segmented planes
    
    Keyword Arguments:
        tcomp {float} -- Overlap fractions must be greater than tcomp for classification (default: {0.80})
    
    Returns:
        [tuple(Dict, Dict) -- The first dict provides evaluation metrics, the second is used for debugging
    """
    # Used for miscellaneous error reporting
    if misc:
        misc['tcomp'] = tcomp
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
    # and n_gt X n_ms matrix to store the pixel overlap count
    overlap = np.zeros((n_gt, n_ms), dtype=np.int)

    logger.info("There are %d unique gt planes and %d machine segmented planes", n_gt, n_ms)

    # print(n_gt)
    # print(n_ms)
    # print(point_count_gt, np.sum(point_count_gt))
    # print(point_count_ms, np.sum(point_count_ms))

    for gt_index, (gt_point_idx_set, gt_unique_label) in enumerate(gt_planes_filtered):
        for ms_index, plane_ms in enumerate(planes_ms):
            ms_point_idx_set = plane_ms['point_indices']
            n_overlap = np.intersect1d(gt_point_idx_set, ms_point_idx_set).size
            overlap[gt_index, ms_index] = n_overlap


    overlap_fraction_gt = overlap / point_count_gt[:, None]
    overlap_fraction_ms = (overlap.T / point_count_ms[:, None]).T
    # print(overlap_fraction_ms)
    # print(overlap_fraction_gt)


    # Oversegmentation has as *common denominator* of ground truth planes
    # A ground truth plane is oversegmented if at least 2 machine segmented planes cover it
    # Therefore many oversegmented calculations are computed row wise (rows = ground truth)

    # Undersegmentation has a *common denominator* of machine segmented planes
    # A machine segmented plane is undersegmented if it covers at least 2 ground truth planes
    # Therefore many undersegmented calculations are computed column wise (columns = machine segmented)

    # which ROWS of overlap_fraction_ms have at least 2 columns greater than tcomp (0.80)
    # should be a row vector of length n_gt with 0/1 indicating if this ground truth plane is oversegmented
    fraction_overseg_mask = (np.sum(overlap_fraction_ms > tcomp, axis=1) >= 2).astype(np.float64)
    # of the rows having this condition, and of columns greater than tcomp, sum up the column in overlap_fraction_gt
    # and mark each row as oversegmented if their sum is > tcomp
    fraction_overseg_union = np.sum((overlap_fraction_ms > tcomp) * overlap_fraction_gt, axis=1) * fraction_overseg_mask
    # n_gt X 1 matrix

    # same idea as above but for undersegmented
    fraction_underseg_mask = (np.sum(overlap_fraction_gt > tcomp, axis=0) >= 2).astype(np.float64)
    fraction_underseg_union = np.sum((overlap_fraction_gt > tcomp) *
                                     overlap_fraction_ms, axis=0) * fraction_underseg_mask
    # 1 X n_ms matrix

    # a boolean matrix indicating which ms/gt planes were correctly segmented
    correct_seg = (overlap_fraction_gt > tcomp) & (overlap_fraction_ms > tcomp)

    # a boolean row mask indicating which row in gt is over segmented
    over_seg = fraction_overseg_union > tcomp

    # a boolean column mask indicating which columns in ms are under segmenting
    under_seg = fraction_underseg_union > tcomp

    # Important Note
    # Some rows and column may be participating in more than one classification (correct, over/under segmented)
    # we must make their class exclusive
    # We start with this by calculating the average fractions each gt/ms plane contributes to each classification
    
    # this makes an n_gt X n_ms matrix specified the average fraction of belonging to correct classification
    avg_correct_seg = (overlap_fraction_gt * correct_seg + overlap_fraction_ms * correct_seg) / 2


    avg_over_seg_ms = np.sum(overlap_fraction_ms * (overlap_fraction_ms > tcomp), 1)
    avg_over_seg_gt = fraction_overseg_union * over_seg
    avg_over_seg = (avg_over_seg_ms + avg_over_seg_gt) / (np.sum(overlap_fraction_ms > tcomp, 1) + 1) * over_seg
    # The end result is n_gt X 1 matrix, 

    avg_under_seg_gt = np.sum(overlap_fraction_gt * (overlap_fraction_gt > tcomp), 0)
    avg_under_seg_ms = fraction_underseg_union * under_seg
    avg_under_seg = (avg_under_seg_ms + avg_under_seg_gt) / (np.sum(overlap_fraction_gt > tcomp, 0) + 1) * under_seg
    # The end result is 1 X n_ms matrix

    # of the machine segmented planes that could possibly contributed to a gt plane being oversegmented
    # sum all its overlap_fraction_ms on the rows that could have possibly contributed to over segmentation
    # this can never be more than 1
    avg_over_seg_cause_temp = (overlap_fraction_ms > tcomp) * overlap_fraction_ms * over_seg[:, None]
    avg_over_seg_cause = np.sum(avg_over_seg_cause_temp, 0)
    # the end result is an 1 X n_ms matrix

    # of the gt planes that could possibly contributed to a ms plane undersegmenting
    # sum all its overlap_fraction_gt on the cols that could have possibly contributed to undersegmentation
    # this can never be more than 1
    avg_under_seg_cause_temp = (((overlap_fraction_gt > tcomp) * overlap_fraction_gt).T * under_seg[:, None]).T
    avg_under_seg_cause = np.sum(avg_under_seg_cause_temp, 1)

    # Exclusify
    # import ipdb; ipdb.set_trace()
    correct_seg_final = (avg_correct_seg >= avg_over_seg[:, None]) & ((avg_correct_seg.T >= avg_under_seg[:, None]).T) & (avg_correct_seg > 0)
    # the end result is a n_gt X n_ms matrix

    over_seg_final = (avg_over_seg > np.sum(avg_correct_seg, 1)) & (avg_over_seg > avg_under_seg_cause)

    under_seg_final = (avg_under_seg > np.sum(avg_correct_seg, 0)) & (avg_under_seg > avg_over_seg_cause)

    # Get Remaining Planes
    over_seg_cause = np.sum((overlap_fraction_ms > tcomp) * over_seg_final[:, None], 0)
    under_seg_cause = np.sum(((overlap_fraction_gt > tcomp).T * under_seg_final[:, None]).T, 1)

    missed_seg = 1 - (np.sum(correct_seg_final, 1) | over_seg_final | under_seg_cause)
    noise_seg = 1 - (np.sum(correct_seg_final, 0) | under_seg_final | over_seg_cause)

    # Calculate Metrics
    n_ms_all = n_ms
    n_corr_seg = np.sum(np.sum(correct_seg_final))
    n_over_seg = np.asscalar(np.sum(over_seg_final))
    n_under_seg = np.asscalar(np.sum(under_seg_final))
    n_missed_seg = np.asscalar(np.sum(missed_seg))
    n_noise_seg = np.asscalar(np.sum(noise_seg))

    f_corr_seg = n_corr_seg / n_gt
    total_points_correct_gt = np.sum(correct_seg_final, axis=1) * point_count_gt 
    f_weighted_corr_seg = np.sum(total_points_correct_gt) / np.sum(point_count_gt)


    test_gt = np.column_stack([np.sum(correct_seg_final, axis=1), over_seg_final, under_seg_cause, missed_seg])
    test_ms = np.row_stack([np.sum(correct_seg_final, axis=0), under_seg_final, over_seg_cause, noise_seg])

    if not np.array_equal(np.sum(test_gt, 1), np.ones((n_gt, ), dtype=np.int)):
        logger.error('Ground-truth classification is not consistent! Misc: %r', misc)

    if not np.array_equal(np.sum(test_ms, 0), np.ones((n_ms, ), dtype=np.int)):
        logger.error('Measurement classification is not consistent! Misc: %r', misc)

    # gt and ms plane ids for each category for debugging purposes
    gt_labels_missed = gt_unique_labels_filtered[np.ma.make_mask(missed_seg)]
    ms_labels_noise = np.where(np.ma.make_mask(noise_seg))[0]
    gt_labels_over_seg = gt_unique_labels_filtered[np.ma.make_mask(over_seg_final)]
    ms_labels_under_seg = np.where(np.ma.make_mask(under_seg_final))[0]

    results = dict(n_gt=n_gt, n_ms_all=n_ms_all,f_weighted_corr_seg=f_weighted_corr_seg, f_corr_seg=f_corr_seg, n_corr_seg=n_corr_seg, n_over_seg=n_over_seg, n_under_seg=n_under_seg, n_missed_seg=n_missed_seg, n_noise_seg=n_noise_seg)
    auxiliary = dict(gt_labels_missed=gt_labels_missed, ms_labels_noise=ms_labels_noise, gt_labels_over_seg=gt_labels_over_seg, ms_labels_under_seg=ms_labels_under_seg)


    logger.info("f_corr: %.2f; f_weighted_corr: %.2f; n_corr: %d; n_over_seg: %d; n_under_seg: %d; n_missed_seg: %d; n_noise_seg: %d",
            f_corr_seg, f_weighted_corr_seg, n_corr_seg, n_over_seg, n_under_seg, n_missed_seg, n_noise_seg)

    return results, auxiliary


def evaluate_rmse(gt_image, planes_ms, outlier_rmse=10.0):
    """Evaluates RMSE error for each machine segmented plane

    Args:
        gt_image (ndarray): Point indices for Ground Truth
        planes_ms (list(dict)): Machine Segmented planes
        outlier_rmse (float, optional): Don't include planes with an RMSE value greater than this. 
                                        This value is taken from SynPEB: https://github.com/acschaefer/ppe/blob/8804bba91debd1917188b509246b64b5e1401d87/matlab/experiments/evalexp.m#L160
                                        Defaults to 10.0

    Returns:
        [float]: The RSME of all predicted MS planes
    """
    gt_flat = gt_image.reshape((gt_image.shape[0] * gt_image.shape[1], 4)) 
    rmse_np = []
    for ms_index, plane_ms in enumerate(planes_ms):
        ms_point_idx_set = plane_ms['point_indices']
        pc_ms = gt_flat[ms_point_idx_set][:, :3]
        _, _, distance, rmse = fit_plane_and_get_rmse(pc_ms)
        if rmse < outlier_rmse:
            rmse_np.append(distance) # distance is signed distance
        else:
            # NOTE - this is never triggered. The default RMSE value is too high (verified 06/30/2020)
            logger.warning("Outlier!")
    rmse_np = np.concatenate(rmse_np, axis=0)  
    # now we get RMSE of all machine segmented planes at once. This is the way its done in PPE evaluation.
    rmse_total = np.sqrt(np.mean(rmse_np ** 2))
    return rmse_total


def evaluate(gt_image, planes_ms, tcomp=0.80, misc=''):
    """Reports evaluation metrics of ground truth vs machine segmented planes
    The key metrics are
        n_corr_seg = # of Correctly segmented ground truth planes
        n_over_seg = # of oversegmented ground truth planes
        n_under_seg = # of machine segmented planes covering multiple ground planes
        n_missed_seg = # of ground truth planes that don't participate in any of the classifications
        n_noise_seg = # of of machine segmented planes that are just noise

    This function differs from evalute_incorrect by using the correct definition of "average of its
    metric pair" (An Experimental Comparision of Range Image Segmentation Algorithms,  Hoover Et Al,
    Section 2.4 Performance Metrics, pg 677, Second to Last Paragraph )
    Basically we take the average of one measure (e.g. defintion 2(a)) the average of the second measure (e.g. 2(b)) 
    and then average this metric pair together. This differs from `evaluate_incorrect` above
    which lumps all the results of all measures together and then averages (line 135 and 136 above).
    This interpretation of "average of its  metric pair" is taken from `compare.c` in C code evaluation 
    tool released by Hoover. (Line 403 and 404).
    
    Arguments:
        gt_image {ndarray} -- A numpy array of N X N X 4. The last dimension is (x,y,z,class)
        planes_ms {List[MachineSegmentedPlanes]} -- A list of machine segmented planes
    
    Keyword Arguments:
        tcomp {float} -- Overlap fractions must be greater than tcomp for classification (default: {0.80})
    
    Returns:
        [tuple(Dict, Dict) -- The first dict provides evaluation metrics, the second is used for debugging
    """
    # Used for miscellaneous error reporting
    if misc:
        misc['tcomp'] = tcomp
    # Note when I use the word class/label it means a unique plane
    # This creates a flattened array of each pixels class
    gt_pixel_labels = gt_image[:, :, 3].flatten().astype(np.int)

    # This returns the the number of unique classes and their counts (# pixels with such a class)
    gt_unique_labels, counts = np.unique(gt_pixel_labels, return_counts=True)
    # print(gt_unique_labels)

    # This is a list of tuples, each tuple having a numpy array and a integer label id.
    # Each numpy array represents a unique plane and is a set of point INDICES
    # This list only accepts ground truth planes whose label is greater than 10
    # https://github.com/acschaefer/ppe/blob/8804bba91debd1917188b509246b64b5e1401d87/matlab/experiments/segcompeval.m#L16
    gt_planes_filtered = [(np.flatnonzero(gt_pixel_labels == gt_unique_label), gt_unique_label) for gt_unique_label in
                          gt_unique_labels if gt_unique_label >= SYNPEB_VALID_INDICES]

    n_gt = len(gt_planes_filtered)  # num of gt planes
    n_ms = len(planes_ms)           # num of machine segmented planes

    # Shortcut if there are no predictions
    if n_ms == 0:
        results = dict(n_gt=n_gt, n_ms_all=0, f_weighted_corr_seg=0.0, f_corr_seg=0.0, n_corr_seg=0, n_over_seg=0, n_under_seg=0, n_missed_seg=n_gt, n_noise_seg=0)
        auxiliary = dict(gt_labels_missed=[], ms_labels_noise=[], gt_labels_over_seg=[], ms_labels_under_seg=[])
        return results, auxiliary


    # filtered ground truth labels, only used at end for debugging
    gt_unique_labels_filtered = np.array(
        [gt_unique_label for gt_unique_label in gt_unique_labels if gt_unique_label >= SYNPEB_VALID_INDICES], dtype=np.int)

    # Number of points in each ground truth plane
    point_count_gt = np.array([count for gt_unique_label, count in zip(
        gt_unique_labels, counts) if gt_unique_label >= SYNPEB_VALID_INDICES])
    # Number of points in each machine segmented plane
    point_count_ms = np.array([plane_ms['point_indices'].size for plane_ms in planes_ms])
    # and n_gt X n_ms matrix to store the pixel overlap count
    overlap = np.zeros((n_gt, n_ms), dtype=np.int)

    logger.info("There are %d unique gt planes and %d machine segmented planes", n_gt, n_ms)

    for gt_index, (gt_point_idx_set, gt_unique_label) in enumerate(gt_planes_filtered):
        for ms_index, plane_ms in enumerate(planes_ms):
            ms_point_idx_set = plane_ms['point_indices']
            n_overlap = np.intersect1d(gt_point_idx_set, ms_point_idx_set).size
            overlap[gt_index, ms_index] = n_overlap


    overlap_fraction_gt = overlap / point_count_gt[:, None]
    overlap_fraction_ms = (overlap.T / point_count_ms[:, None]).T


    # Oversegmentation has as *common denominator* of ground truth planes
    # A ground truth plane is oversegmented if at least 2 machine segmented planes cover it
    # Therefore many oversegmented calculations are computed row wise (rows = ground truth)

    # Undersegmentation has a *common denominator* of machine segmented planes
    # A machine segmented plane is undersegmented if it covers at least 2 ground truth planes
    # Therefore many undersegmented calculations are computed column wise (columns = machine segmented)

    # which ROWS of overlap_fraction_ms have at least 2 columns greater than tcomp (0.80)
    # should be a row vector of length n_gt with 0/1 indicating if this ground truth plane is oversegmented
    fraction_overseg_mask = (np.sum(overlap_fraction_ms > tcomp, axis=1) >= 2).astype(np.float64)
    # of the rows having this condition, and of columns greater than tcomp, sum up the column in overlap_fraction_gt
    # and mark each row as oversegmented if their sum is > tcomp
    fraction_overseg_union = np.sum((overlap_fraction_ms > tcomp) * overlap_fraction_gt, axis=1) * fraction_overseg_mask
    # n_gt X 1 matrix

    # same idea as above but for undersegmented
    fraction_underseg_mask = (np.sum(overlap_fraction_gt > tcomp, axis=0) >= 2).astype(np.float64)
    fraction_underseg_union = np.sum((overlap_fraction_gt > tcomp) *
                                     overlap_fraction_ms, axis=0) * fraction_underseg_mask
    # 1 X n_ms matrix

    # a boolean matrix indicating which ms/gt planes were correctly segmented
    correct_seg = (overlap_fraction_gt > tcomp) & (overlap_fraction_ms > tcomp)

    # a boolean row mask indicating which row in gt is over segmented
    over_seg = fraction_overseg_union > tcomp

    # a boolean column mask indicating which columns in ms are under segmenting
    under_seg = fraction_underseg_union > tcomp

    # Important Note
    # Some rows and column may be participating in more than one classification (correct, over/under segmented)
    # we must make their class exclusive
    # We start with this by calculating the average fractions each gt/ms plane contributes to each classification
    
    ## Correct Segmentation average measrues
    # this makes an n_gt X n_ms matrix specified the average fraction of belonging to correct classification
    avg_correct_seg = (overlap_fraction_gt * correct_seg + overlap_fraction_ms * correct_seg) / 2

    ## Over Segmentation average measures
    # Sum of machine segmented planes overlap fraction with oversegmentation
    cum_over_seg_ms = np.sum(overlap_fraction_ms * (overlap_fraction_ms > tcomp), 1)
    # The number of machine segmented planes involved in oversegmentation per ground truth plane
    count_over_seg_ms = np.sum(overlap_fraction_ms > tcomp, 1)
    # To prevent divide by zero for rows in which there was NO oversegmentation, this will be removed with over_seg mask 4 lines below
    count_over_seg_ms[count_over_seg_ms == 0] = 1.0
    # The average overlap fraction of all machine segments involved in oversegmentation of a ground truth plane
    # Called Measure 1 in `compare.c`, or 2(a) in Hoover pg. 677
    avg_over_seg_ms = cum_over_seg_ms  / count_over_seg_ms * over_seg
    # The cumulative GT overlap of machine segmented planes, 
    # Called Measure 2 `compare.c`, or 2(b) in Hoover pg. 677
    avg_over_seg_gt = fraction_overseg_union * over_seg
    avg_over_seg = (avg_over_seg_ms + avg_over_seg_gt) / 2 * over_seg
    # The end result is n_gt X 1 matrix, 

    ## Under Segmentation average measures
    # Sum of the gt planes overlap fraction with undersegmentation 
    cum_under_seg_gt = np.sum(overlap_fraction_gt * (overlap_fraction_gt > tcomp), 0)
    # The number of ground truth planes involved in undersegmentation
    count_under_seg_gt = np.sum(overlap_fraction_gt > tcomp, 0)
    # To prevent divide by zero for cols in which there was NO undersegmentation, this will be removed with under_seg mask 4 lines below
    count_under_seg_gt[count_under_seg_gt == 0] = 1.0
    # The average overlap fraction of all ground truth planes involved in undersegmentation with a machine segmented plane
    # Called Measure 2 in `compare.c`, or 3(b) in Hoover pg. 677
    avg_under_seg_gt = cum_under_seg_gt  / count_under_seg_gt * under_seg
    # Called Measure 1 `compare.c`, or 3(a) in Hoover pg. 677
    avg_under_seg_ms = fraction_underseg_union * under_seg
    avg_under_seg = (avg_under_seg_gt + avg_under_seg_ms) / 2 * under_seg
    # The end result is 1 X n_ms matrix

    ## Take the average fraction we just calculated and associate with the their counterparts
    ## which *caused* them to occur

    # Correspond the average fractions of oversegmentation (gt planes) to the cause (ms planes)
    over_seg_2D_mask = overlap_fraction_ms > tcomp
    # Ensure that the mask is only true if all requirements have been made
    over_seg_2D_mask[~over_seg, :] = False
    # The rows (gt planes) that are oversegmented and their **corresponding** machine segmented planes **causing** this
    (over_seg_rows_gt, over_seg_cols_ms) = np.where(over_seg_2D_mask) 
    avg_over_seg_cause = np.zeros((n_ms,))
    for idx, ms_plane_idx in enumerate(over_seg_cols_ms):
        gt_plane_idx = over_seg_rows_gt[idx]
        avg_over_seg_cause[ms_plane_idx] = avg_over_seg[gt_plane_idx]
    # the end result is an 1 X n_ms matrix
    # each element has the average fraction of over segmentation this machine segmented plane had if it participated in oversegmentation


    # Correspond the average fractions of undersegmentation (ms planes) to the cause (gt planes)
    under_seg_2D_mask = overlap_fraction_gt > tcomp
    # Ensure that the mask is only true if all requirements have been made
    under_seg_2D_mask[:, ~under_seg] = False
    # The rows (gt planes) that are oversegmented and their **corresponding** machine segmented planes **causing** this
    (under_seg_rows_gt, under_seg_cols_ms) = np.where(under_seg_2D_mask) 
    avg_under_seg_cause = np.zeros((n_gt,))
    for idx, gt_plane_idx in enumerate(under_seg_rows_gt):
        ms_plane_idx = under_seg_cols_ms[idx]
        avg_under_seg_cause[gt_plane_idx] = avg_under_seg[ms_plane_idx]
    # the end result is an gt X 1 matrix

    # Exclusify
    # import ipdb; ipdb.set_trace()
    correct_seg_final = (avg_correct_seg >= avg_over_seg[:, None]) & ((avg_correct_seg.T >= avg_under_seg[:, None]).T) & (avg_correct_seg > 0)
    # the end result is a n_gt X n_ms matrix

    over_seg_final = (avg_over_seg > np.sum(avg_correct_seg, 1)) & (avg_over_seg >= avg_under_seg_cause)

    under_seg_final = (avg_under_seg > np.sum(avg_correct_seg, 0)) & (avg_under_seg > avg_over_seg_cause)

    # Get Remaining Planes
    over_seg_cause = np.sum((overlap_fraction_ms > tcomp) * over_seg_final[:, None], 0)
    under_seg_cause = np.sum(((overlap_fraction_gt > tcomp).T * under_seg_final[:, None]).T, 1)

    missed_seg = 1 - (np.sum(correct_seg_final, 1) | over_seg_final | under_seg_cause)
    noise_seg = 1 - (np.sum(correct_seg_final, 0) | under_seg_final | over_seg_cause)

    # Calculate Metrics
    n_ms_all = n_ms
    n_corr_seg = np.sum(np.sum(correct_seg_final))
    n_over_seg = np.asscalar(np.sum(over_seg_final))
    n_under_seg = np.asscalar(np.sum(under_seg_final))
    n_missed_seg = np.asscalar(np.sum(missed_seg))
    n_noise_seg = np.asscalar(np.sum(noise_seg))

    f_corr_seg = n_corr_seg / n_gt
    total_points_correct_gt = np.sum(correct_seg_final, axis=1) * point_count_gt
    # this is k in PPE paper
    f_weighted_corr_seg = np.sum(total_points_correct_gt) / np.sum(point_count_gt)
    # now fit a plane anc calculate rmse
    rmse = evaluate_rmse(gt_image, planes_ms)


    test_gt = np.column_stack([np.sum(correct_seg_final, axis=1), over_seg_final, under_seg_cause, missed_seg])
    test_ms = np.row_stack([np.sum(correct_seg_final, axis=0), under_seg_final, over_seg_cause, noise_seg])

    if not np.array_equal(np.sum(test_gt, 1), np.ones((n_gt, ), dtype=np.int)):
        logger.error('Ground-truth classification is not consistent! Misc: %r', misc)

    if not np.array_equal(np.sum(test_ms, 0), np.ones((n_ms, ), dtype=np.int)):
        logger.error('Measurement classification is not consistent! Misc: %r', misc)

    # gt and ms plane ids for each category for debugging purposes
    gt_labels_missed = gt_unique_labels_filtered[np.ma.make_mask(missed_seg)]
    ms_labels_noise = np.where(np.ma.make_mask(noise_seg))[0]
    gt_labels_over_seg = gt_unique_labels_filtered[np.ma.make_mask(over_seg_final)]
    ms_labels_under_seg = np.where(np.ma.make_mask(under_seg_final))[0]

    # get a mapping for the gt colors ids for the *correctly** segmented planes
    rows, cols = np.where(correct_seg_final)
    map_gt_id_to_ms = np.full((n_ms,), 100, dtype=np.int)
    map_gt_id_to_ms[cols] = gt_unique_labels_filtered[rows]

    results = dict(n_gt=n_gt, n_ms_all=n_ms_all,f_weighted_corr_seg=f_weighted_corr_seg, f_corr_seg=f_corr_seg, n_corr_seg=n_corr_seg, n_over_seg=n_over_seg, n_under_seg=n_under_seg, n_missed_seg=n_missed_seg, n_noise_seg=n_noise_seg, rmse=rmse)
    auxiliary = dict(gt_labels_missed=gt_labels_missed, ms_labels_noise=ms_labels_noise, gt_labels_over_seg=gt_labels_over_seg, ms_labels_under_seg=ms_labels_under_seg, map_gt_id_to_ms=map_gt_id_to_ms)

    logger.info("f_corr: %.2f; f_weighted_corr: %.2f; rmse: %.2f;  n_corr: %d; n_over_seg: %d; n_under_seg: %d; n_missed_seg: %d; n_noise_seg: %d",
            f_corr_seg, f_weighted_corr_seg, rmse, n_corr_seg, n_over_seg, n_under_seg, n_missed_seg, n_noise_seg)

    return results, auxiliary


