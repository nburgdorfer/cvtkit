import numpy as np

def abs_error(input_depth, fused_depth, gt_depth):
    input_signed_error = input_depth - gt_depth
    fused_signed_error = fused_depth - gt_depth

    # compute gt mask and number of valid pixels
    gt_mask = np.not_equal(gt_depth, 0.0).astype(np.double)
    input_abs_error = np.abs(input_signed_error) * gt_mask
    fused_abs_error = np.abs(fused_signed_error) * gt_mask

    return input_abs_error, fused_abs_error
