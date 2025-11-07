# cvt/geometry.py

"""Module including geometric routines."""

import numpy as np
import cv2
import open3d as o3d
from typing import Tuple, List, Any
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
import torchvision.transforms as tvt
from torch.cuda.amp import autocast
from torch import Tensor
from numpy.typing import NDArray


from cvtkit.camera import intrinsic_pyramid
from cvtkit.common import groupwise_correlation
from cvtkit.io import read_pfm, read_single_cam_sfm


def depths_to_points(view, depthmap):
    c2w = (view.P).inverse()
    W, H = view.width, view.height
    ndc2pix = (
        torch.tensor([[W / 2, 0, 0, (W) / 2], [0, H / 2, 0, (H) / 2], [0, 0, 0, 1]])
        .float()
        .cuda()
        .T
    )
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T

    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device="cuda").float(),
        torch.arange(H, device="cuda").float(),
        indexing="xy",
    )
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(
        -1, 3
    )
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(view, depth):
    """
    view: view camera
    depth: depthmap
    """
    height, width = depth.shape
    points = project_depth_map(depth, view.KP_inv).reshape(height, width, 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


def edge_mask(depth, near_depth, far_depth):
    down_gt = F.interpolate(
        depth,
        scale_factor=0.5,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )
    down_up_gt = F.interpolate(
        down_gt,
        scale_factor=2,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )
    res = torch.abs(depth - down_up_gt)
    high_frequency_mask = res > (0.001 * (far_depth - near_depth)[:, None, None, None])
    valid_gt_mask = (
        -F.max_pool2d(-depth, kernel_size=5, stride=1, padding=2)
    ) > near_depth[:, None, None, None]
    high_frequency_mask = high_frequency_mask * valid_gt_mask
    high_frequency_mask = (
        (1 - high_frequency_mask.to(torch.int32)) * valid_gt_mask
    ).to(torch.int32)
    return high_frequency_mask


def get_epipolar_inds(x0, y0, x1, y1, x_lim, y_lim, max_patches):
    dx = x1 - x0
    dy = y1 - y0
    yi = torch.ones_like(y0)

    negative_mask = torch.where(dy < 0, -1, 1)
    yi *= negative_mask
    dy *= negative_mask

    D = (2 * dy) - dx
    y = y0
    x = x0

    batch, h, w = x.shape

    epipolar_grid = torch.zeros((batch, max_patches, h, w, 2)).to(x0)
    for i in range(max_patches):
        # build valid indices mask
        valid_mask = torch.where(
            (x < x_lim).to(torch.bool) & (x >= 0).to(torch.bool), 1, 0
        )
        valid_mask *= torch.where(
            (y < y_lim).to(torch.bool) & (y >= 0).to(torch.bool), 1, 0
        )
        valid_mask = valid_mask.unsqueeze(-1).repeat(1, 1, 1, 2)

        # stack xy and apply valid indices mask
        xy = torch.stack([x, y], dim=-1)
        epipolar_grid[:, i, :, :, :] = (xy * valid_mask) - (1 - valid_mask)

        mask = torch.where(D > 0, 1, 0)
        y = (y + yi) * mask + y * (1 - mask)
        D = ((D + (2 * (dy - dx))) * mask) + ((D + (2 * dy)) * (1 - mask))
        x += 1

    return epipolar_grid[:, :, :, :, 0], epipolar_grid[:, :, :, :, 1]


def epipolar_patch_retrieval(
    imgs: Tensor, intrinsics: Tensor, extrinsics: Tensor, patch_size: int
):
    assert isinstance(imgs, Tensor)
    assert isinstance(intrinsics, Tensor)
    assert isinstance(extrinsics, Tensor)

    batch_size, _, _, height, width = imgs.shape
    half_patch_size = patch_size // 2

    x_flat = torch.arange((half_patch_size), width + 1)[::patch_size].to(imgs)
    y_flat = torch.arange((half_patch_size), height + 1)[::patch_size].to(imgs)

    xgrid, ygrid = torch.meshgrid([x_flat, y_flat], indexing="xy")
    xy = torch.stack(
        [xgrid, ygrid, torch.ones_like(xgrid)], dim=-1
    )  # [patched_height, patch_width, 3]
    patched_height, patched_width, _ = xy.shape
    xy = (
        xy.unsqueeze(0).repeat(batch_size, 1, 1, 1).unsqueeze(-1)
    )  # [batch_size, patched_height, patch_width, 3, 1]

    max_patches = patched_height + patched_width - 1

    patch_volumes = []
    for j in range(imgs.shape[1]):
        # view_patches = [imgs[:,j].unsqueeze(2)]
        view_patches = []

        K = intrinsics[:, j]
        P_src = extrinsics[:, j]

        for i in range(imgs.shape[1]):
            if i == j:
                continue

            x_lim = (
                torch.ones((batch_size, patched_height, patched_width)) * patched_width
            ).to(imgs)
            y_lim = (
                torch.ones((batch_size, patched_height, patched_width)) * patched_height
            ).to(imgs)

            P_tgt = extrinsics[:, i]
            Fm = fundamental_from_KP(K, P_src, P_tgt)

            assert isinstance(Fm, Tensor)
            Fm = Fm.reshape(batch_size, 1, 1, 3, 3).repeat(
                1, patched_height, patched_width, 1, 1
            )  # [batch_size, patched_height, patch_width, 3, 1]
            line = torch.matmul(Fm, xy).squeeze(
                -1
            )  # [batch_size, patched_height, patch_width, 3]

            ## Start Point ##
            # initial x coordinate, comput y corrdinate
            x0 = (
                x_flat[0]
                .reshape(1, 1, 1)
                .repeat(batch_size, patched_height, patched_width)
            )
            y0 = (-(line[:, :, :, 0] / line[:, :, :, 1]) * x0) - (
                line[:, :, :, 2] / line[:, :, :, 1]
            )
            # check for invalid y coordinates
            y_mask_lt = torch.where(y0 < 0, 1, 0)
            y_mask_gt = torch.where(y0 >= height, 1, 0)
            y_mask_out = y_mask_lt + y_mask_gt
            # adjust for invalid y coordinates
            y0 = y0 * (1 - y_mask_lt) + y_flat[0] * y_mask_lt
            y0 = y0 * (1 - y_mask_gt) + y_flat[-1] * y_mask_gt
            x0 = (x0 * (1 - y_mask_out)) + (
                (
                    (-(line[:, :, :, 1] / line[:, :, :, 0]) * y0)
                    - (line[:, :, :, 2] / line[:, :, :, 0])
                )
                * (y_mask_out)
            )
            # if x coordinate is invalid
            valid_mask = torch.where(
                (x0 >= 0).to(torch.bool) & (x0 < width).to(torch.bool), 1, 0
            )
            x0 = (x0 * valid_mask) - (1 - valid_mask)
            y0 = (y0 * valid_mask) - (1 - valid_mask)

            ## End Point ##
            # initial x coordinate, comput y corrdinate
            x1 = (
                x_flat[-1]
                .reshape(1, 1, 1)
                .repeat(batch_size, patched_height, patched_width)
            )
            y1 = (-(line[:, :, :, 0] / line[:, :, :, 1]) * x1) - (
                line[:, :, :, 2] / line[:, :, :, 1]
            )
            # check for invalid y coordinates
            y_mask_lt = torch.where(y1 < 0, 1, 0)
            y_mask_gt = torch.where(y1 >= height, 1, 0)
            y_mask_out = y_mask_lt + y_mask_gt
            # adjust for invalid y coordinates
            y1 = y1 * (1 - y_mask_lt) + y_flat[0] * y_mask_lt
            y1 = y1 * (1 - y_mask_gt) + y_flat[-1] * y_mask_gt
            x1 = (x1 * (1 - y_mask_out)) + (
                (
                    (-(line[:, :, :, 1] / line[:, :, :, 0]) * y1)
                    - (line[:, :, :, 2] / line[:, :, :, 0])
                )
                * (y_mask_out)
            )
            # if x coordinate is invalid
            valid_mask = torch.where(
                (x1 >= 0).to(torch.bool) & (x1 < width).to(torch.bool), 1, 0
            )
            x1 = (x1 * valid_mask) - (1 - valid_mask)
            y1 = (y1 * valid_mask) - (1 - valid_mask)

            # convert image indices into patch indices
            x0 = torch.round((x0 - (half_patch_size)) / patch_size)
            x1 = torch.round((x1 - (half_patch_size)) / patch_size)
            y0 = torch.round((y0 - (half_patch_size)) / patch_size)
            y1 = torch.round((y1 - (half_patch_size)) / patch_size)

            # compute x and y slopes
            slope_x = torch.abs(x1 - x0)
            slope_y = torch.abs(y1 - y0)

            # flip x's and y's depending on slope
            small_slope_mask = torch.where(slope_y < slope_x, 1, 0)
            x0_temp = x0 * small_slope_mask + y0 * (1 - small_slope_mask)
            y0 = y0 * small_slope_mask + x0 * (1 - small_slope_mask)
            x0 = x0_temp
            x1_temp = x1 * small_slope_mask + y1 * (1 - small_slope_mask)
            y1 = y1 * small_slope_mask + x1 * (1 - small_slope_mask)
            x1 = x1_temp
            x_lim_temp = x_lim * small_slope_mask + y_lim * (1 - small_slope_mask)
            y_lim = y_lim * small_slope_mask + x_lim * (1 - small_slope_mask)
            x_lim = x_lim_temp

            # flip start and end points so start is smaller
            small_end_mask = torch.where(x1 < x0, 1, 0)
            x0_temp = x0 * (1 - small_end_mask) + x1 * small_end_mask
            x1 = x1 * (1 - small_end_mask) + x0 * small_end_mask
            x0 = x0_temp
            y0_temp = y0 * (1 - small_end_mask) + y1 * small_end_mask
            y1 = y1 * (1 - small_end_mask) + y0 * small_end_mask
            y0 = y0_temp

            # grab nearest patch indices
            x_grid, y_grid = get_epipolar_inds(
                x0, y0, x1, y1, x_lim, y_lim, max_patches
            )

            # flip x and y indices back where necessary (using small_slope_mask)
            small_slope_mask = small_slope_mask.reshape(
                batch_size, 1, patched_height, patched_width
            ).repeat(1, max_patches, 1, 1)
            x_grid_temp = x_grid * small_slope_mask + y_grid * (1 - small_slope_mask)
            y_grid = y_grid * small_slope_mask + x_grid * (1 - small_slope_mask)
            x_grid = x_grid_temp
            epipolar_grid = torch.stack([x_grid, y_grid], dim=-1)

            # convert patch indices into image indices
            epipolar_grid = epipolar_grid * patch_size + (half_patch_size)
            epipolar_grid = torch.where(epipolar_grid < 0, -1, epipolar_grid)
            patch_grid = epipolar_grid.cpu().numpy()

            # duplicate patch center indices over entire patch
            epipolar_grid = torch.repeat_interleave(epipolar_grid, patch_size, dim=2)
            epipolar_grid = torch.repeat_interleave(epipolar_grid, patch_size, dim=3)

            # apply center offset matrix
            valid_mask = torch.where(epipolar_grid >= 0, 1, 0)
            patch_offset = torch.arange(-half_patch_size, half_patch_size).to(imgs)
            x_offset, y_offset = torch.meshgrid(
                [patch_offset, patch_offset], indexing="xy"
            )
            patch_offset = torch.stack([x_offset, y_offset], dim=-1)
            patch_offset = torch.tile(patch_offset, (patched_height, patched_width, 1))
            patch_offset = patch_offset.reshape(1, 1, height, width, 2).repeat(
                batch_size, max_patches, 1, 1, 1
            )
            epipolar_grid += patch_offset
            epipolar_grid = (epipolar_grid * valid_mask) - (1 - valid_mask)

            # normalize coordinate grid
            min_coord = torch.tensor([0, 0]).to(imgs)
            min_coord = min_coord.reshape(1, 1, 1, 1, 2).repeat(
                batch_size, max_patches, height, width, 1
            )
            max_coord = torch.tensor([width - 1, height - 1]).to(imgs)
            max_coord = max_coord.reshape(1, 1, 1, 1, 2).repeat(
                batch_size, max_patches, height, width, 1
            )
            norm_grid = (epipolar_grid - min_coord) / (max_coord - min_coord)
            norm_grid = (norm_grid * 2) - 1

            # aggregate image patches
            img_patches = F.grid_sample(
                imgs[:, i],
                norm_grid.reshape(batch_size, max_patches, height * width, 2),
                mode="nearest",
                padding_mode="zeros",
            )
            img_patches = img_patches.reshape(batch_size, 3, max_patches, height, width)
            view_patches.append(img_patches)

            #   #### visual
            #   r,c = patched_height//2, patched_width//2
            #   #for k in range(max_patches):
            #   #    x_k,y_k,_ = xy[0,r,c,:,0].cpu().numpy()
            #   #    fig = plt.figure()
            #   #    ax = fig.add_subplot(111)
            #   #    ax.imshow(torch.movedim(img_patches[0,:,k], (0,1,2), (2,0,1)).cpu().numpy())
            #   #    rect_k = Rectangle((x_k-(half_patch_size),y_k-(half_patch_size)), patch_size, patch_size, color='red', fc = 'none', lw = 0.5)
            #   #    ax.add_patch(rect_k)
            #   #    plt.savefig(f"patches/ref{j:02d}_src{i:02d}_patch{k:04d}.png")
            #   #    plt.close()

            #   # plot src patches
            #   ep = patch_grid[0,:,r,c,:]
            #   fig = plt.figure()
            #   ax = fig.add_subplot(111)
            #   ax.imshow(torch.movedim(imgs[0,i],(0,1,2),(2,0,1)).cpu().numpy())
            #   for x_i,y_i in ep:
            #       if x_i>=0 and y_i>=0:
            #           rect_i = Rectangle((x_i-(half_patch_size),y_i-(half_patch_size)), patch_size, patch_size, color='red', fc = 'none', lw = 0.5)
            #           ax.add_patch(rect_i)
            #   plt.axis('off')
            #   plt.savefig(f"patches/ref{j:02d}_src{i:02d}.png", bbox_inches='tight', dpi=300)
            #   plt.close()
            #   #### visual
        #   #### visual
        #   # plot ref point
        #   ref_pix = xy[0,r,c,:,0].cpu().numpy()
        #   plt.imshow(torch.movedim(imgs[0,j],(0,1,2),(2,0,1)).cpu().numpy())
        #   plt.plot(ref_pix[0].item(),ref_pix[1].item(),'ro')
        #   plt.axis('off')
        #   plt.savefig(f"patches/ref{j:02d}.png", bbox_inches='tight', dpi=300)
        #   plt.close()
        #   sys.exit()
        #   #### visual

        patch_volumes.append(torch.cat(view_patches, dim=2))
    return torch.stack(patch_volumes, dim=1)


def essential_from_features(
    src_image_file: str, tgt_image_file: str, K: np.ndarray
) -> np.ndarray:
    """Computes the essential matrix between two images using image features.

    Parameters:
        src_image_file: Input file for the source image.
        tgt_image_file: Input file for the target image.
        K: Intrinsics matrix of the two cameras (assumed to be constant between views).

    Returns:
        The essential matrix betweent the two image views.
    """
    src_image = cv2.imread(src_image_file)
    tgt_image = cv2.imread(tgt_image_file)

    # compute matching features
    (src_points, tgt_points) = match_features(src_image, tgt_image)

    # Compute fundamental matrix
    E, mask = cv2.findEssentialMat(src_points, tgt_points, K, method=cv2.RANSAC)

    return E


def fundamental_from_KP(
    K: NDArray[Any] | Tensor, P_src: NDArray[Any] | Tensor, P_tgt: NDArray[Any] | Tensor
) -> NDArray[Any] | Tensor:
    if isinstance(K, Tensor):
        assert isinstance(P_src, Tensor)
        assert isinstance(P_tgt, Tensor)
        return _fundamental_from_KP_torch(K, P_src, P_tgt)
    elif isinstance(K, np.ndarray):
        assert isinstance(P_src, np.ndarray)
        assert isinstance(P_tgt, np.ndarray)
        return _fundamental_from_KP_numpy(K, P_src, P_tgt)
    else:
        raise Exception(f"Unknown data type '{type(K)}'")


def _fundamental_from_KP_torch(K: Tensor, P_src: Tensor, P_tgt: Tensor) -> Tensor:
    """Computes the fundamental matrix between two images using camera parameters.

    Parameters:
        K: Intrinsics matrix of the two cameras (assumed to be constant between views).
        P_src: Extrinsics matrix for the source view.
        P_tgt: Extrinsics matrix for the target view.

    Returns:
        The fundamental matrix betweent the two cameras.
    """
    F_mats = []
    for i in range(K.shape[0]):
        R1 = P_src[i, 0:3, 0:3]
        t1 = P_src[i, 0:3, 3]
        R2 = P_tgt[i, 0:3, 0:3]
        t2 = P_tgt[i, 0:3, 3]

        t1aug = torch.tensor([t1[0], t1[1], t1[2], 1]).to(K)
        epi2 = torch.matmul(P_tgt[i], t1aug)
        epi2 = torch.matmul(K[i], epi2[0:3])

        R = torch.matmul(R2, torch.t(R1))
        t = t2 - torch.matmul(R, t1)
        K1inv = torch.linalg.inv(K[i])
        K2invT = torch.t(K1inv)
        tx = torch.tensor([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]).to(K)
        F = torch.matmul(K2invT, torch.matmul(tx, torch.matmul(R, K1inv)))
        F = F / (torch.max(F) + 1e-10)
        F_mats.append(F)

    return torch.stack(F_mats, dim=0)


def _fundamental_from_KP_numpy(
    K: np.ndarray, P_src: np.ndarray, P_tgt: np.ndarray
) -> np.ndarray:
    """Computes the fundamental matrix between two images using camera parameters.

    Parameters:
        K: Intrinsics matrix of the two cameras (assumed to be constant between views).
        P_src: Extrinsics matrix for the source view.
        P_tgt: Extrinsics matrix for the target view.

    Returns:
        The fundamental matrix betweent the two cameras.
    """
    R1 = P_src[0:3, 0:3]
    t1 = P_src[0:3, 3]
    R2 = P_tgt[0:3, 0:3]
    t2 = P_tgt[0:3, 3]

    t1aug = np.array([t1[0], t1[1], t1[2], 1])
    epi2 = np.matmul(P_tgt, t1aug)
    epi2 = np.matmul(K, epi2[0:3])

    R = np.matmul(R2, np.transpose(R1))
    t = t2 - np.matmul(R, t1)
    K1inv = np.linalg.inv(K)
    K2invT = np.transpose(K1inv)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.matmul(K2invT, np.matmul(tx, np.matmul(R, K1inv)))
    F = F / np.amax(F)

    return F


def fundamental_from_features(src_image_file: str, tgt_image_file: str) -> np.ndarray:
    """Computes the fundamental matrix between two images using image features.

    Parameters:
        src_image_file: Input file for the source image.
        tgt_image_file: Input file for the target image.

    Returns:
        The fundamental matrix betweent the two image views.
    """
    src_image = cv2.imread(src_image_file)
    tgt_image = cv2.imread(tgt_image_file)

    # compute matching features
    (src_points, tgt_points) = match_features(src_image, tgt_image)

    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(src_points, tgt_points, cv2.FM_8POINT)

    return F


def geometric_consistency_error(
    src_depth: NDArray[Any],
    src_cam: NDArray[Any],
    tgt_depth: NDArray[Any],
    tgt_cam: NDArray[Any],
) -> NDArray[Any]:
    """Computes the geometric consistency error between a source and target depth map.

    Parameters:
        src_depth: Depth map for the source view.
        src_cam: Camera parameters for the source depth map viewpoint.
        tgt_depth: Depth map for the target view.
        tgt_cam: Camera parameters for the target depth map viewpoint.

    Returns:
        The binary consistency mask encoding depth consensus between source and target depth maps.
    """
    height, width = src_depth.shape
    x_src, y_src = np.meshgrid(np.arange(0, width), np.arange(0, height))

    _, coords_reprojected, _ = reproject(
        src_depth=src_depth,
        src_K=src_cam[1],
        src_P=src_cam[0],
        tgt_depth=tgt_depth,
        tgt_K=tgt_cam[1],
        tgt_P=tgt_cam[0],
    )
    assert isinstance(coords_reprojected, np.ndarray)

    dist = np.sqrt(
        (coords_reprojected[:, :, 0] - x_src) ** 2
        + (coords_reprojected[:, :, 1] - y_src) ** 2
    )

    return dist


def geometric_consistency_mask(
    src_depth: NDArray[Any] | Tensor,
    src_K: NDArray[Any] | Tensor,
    src_P: NDArray[Any] | Tensor,
    tgt_depth: NDArray[Any] | Tensor,
    tgt_K: NDArray[Any] | Tensor,
    tgt_P: NDArray[Any] | Tensor,
    pixel_th: float,
) -> NDArray[Any] | Tensor:
    if isinstance(src_depth, Tensor):
        assert isinstance(src_K, Tensor)
        assert isinstance(src_P, Tensor)
        assert isinstance(tgt_depth, Tensor)
        assert isinstance(tgt_K, Tensor)
        assert isinstance(tgt_P, Tensor)
        return _geometric_consistency_mask_torch(
            src_depth, src_K, src_P, tgt_depth, tgt_K, tgt_P, pixel_th
        )
    elif isinstance(src_depth, np.ndarray):
        assert isinstance(src_K, np.ndarray)
        assert isinstance(src_P, np.ndarray)
        assert isinstance(tgt_depth, np.ndarray)
        assert isinstance(tgt_K, np.ndarray)
        assert isinstance(tgt_P, np.ndarray)
        return _geometric_consistency_mask_numpy(
            src_depth, src_K, src_P, tgt_depth, tgt_K, tgt_P, pixel_th
        )
    else:
        raise Exception(f"Unknown data type '{type(src_depth)}'")


def _geometric_consistency_mask_torch(
    src_depth: Tensor,
    src_K: Tensor,
    src_P: Tensor,
    tgt_depth: Tensor,
    tgt_K: Tensor,
    tgt_P: Tensor,
    pixel_th: float,
) -> Tensor:
    """Computes the geometric consistency mask between a source and target depth map.

    Parameters:
        src_depth: Depth map for the source view.
        src_K: Intrinsic camera parameters for the source depth map viewpoint.
        src_P: Extrinsic camera parameters for the source depth map viewpoint.
        tgt_depth: Target depth map used for re-projection.
        tgt_K: Intrinsic camera parameters for the target depth map viewpoint.
        tgt_P: Extrinsic camera parameters for the target depth map viewpoint.
        pixel_th: Pixel re-projection threshold to determine matching depth estimates.

    Returns:
        The binary consistency mask encoding depth consensus between source and target depth maps.
    """
    batch_size, _, height, width = src_depth.shape
    _, coords_reprojected, _ = reproject(
        src_depth, src_K, src_P, tgt_depth, tgt_K, tgt_P
    )
    assert isinstance(coords_reprojected, Tensor)

    x_src, y_src = torch.meshgrid(
        torch.arange(0, width), torch.arange(0, height), indexing="xy"
    )
    x_src = x_src.unsqueeze(0).repeat(batch_size, 1, 1).to(src_depth)
    y_src = y_src.unsqueeze(0).repeat(batch_size, 1, 1).to(src_depth)
    dist = torch.sqrt(
        (coords_reprojected[:, :, :, 0] - x_src) ** 2
        + (coords_reprojected[:, :, :, 1] - y_src) ** 2
    )

    mask = torch.where(dist < pixel_th, 1, 0)
    return mask


def _geometric_consistency_mask_numpy(
    src_depth: NDArray[Any],
    src_K: NDArray[Any],
    src_P: NDArray[Any],
    tgt_depth: NDArray[Any],
    tgt_K: NDArray[Any],
    tgt_P: NDArray[Any],
    pixel_th: float,
) -> NDArray[Any]:
    """Computes the geometric consistency mask between a source and target depth map.

    Parameters:
        src_depth: Depth map for the source view.
        src_K: Intrinsic camera parameters for the source depth map viewpoint.
        src_P: Extrinsic camera parameters for the source depth map viewpoint.
        tgt_depth: Target depth map used for re-projection.
        tgt_K: Intrinsic camera parameters for the target depth map viewpoint.
        tgt_P: Extrinsic camera parameters for the target depth map viewpoint.
        pixel_th: Pixel re-projection threshold to determine matching depth estimates.

    Returns:
        The binary consistency mask encoding depth consensus between source and target depth maps.
    """
    height, width = src_depth.shape
    x_src, y_src = np.meshgrid(np.arange(0, width), np.arange(0, height))

    _, coords_reprojected, _ = reproject(
        src_depth, src_K, src_P, tgt_depth, tgt_K, tgt_P
    )
    assert isinstance(coords_reprojected, np.ndarray)

    dist = np.sqrt(
        (coords_reprojected[:, :, 0] - x_src) ** 2
        + (coords_reprojected[:, :, 1] - y_src) ** 2
    )
    mask = np.where(dist < pixel_th, 1, 0)

    return mask


def get_uncovered_mask(data, output):
    hypos = output["hypos"]
    intervals = output["intervals"]
    _, H, W = data["target_depth"].shape
    levels = len(hypos)

    uncovered_masks = torch.zeros(levels, H, W)
    for l in range(levels):
        hypo = hypos[l][0].squeeze(0)
        planes, h, w = hypo.shape
        interval = intervals[l][0].squeeze(0)
        target_depth = tvf.resize(
            data["target_depth"][0:1].unsqueeze(0), [h, w]
        ).reshape(h, w)

        ### compute coverage
        diff = torch.abs(hypo - target_depth)
        min_interval = (
            interval[:, 0:1] * 0.5
        )  # intervals are bin widths, divide by 2 for radius
        coverage = torch.clip(
            torch.where(diff <= min_interval, 1, 0).sum(dim=0, keepdim=True), 0, 1
        )
        uncovered = torch.clip(
            torch.where(coverage <= 0, 1, 0).sum(dim=0, keepdim=True), 0, 1
        )
        valid_targets = torch.where(target_depth > 0, 1, 0)
        uncovered *= valid_targets
        uncovered_masks[l] = (
            tvf.resize(
                uncovered.reshape(1, 1, h, w),
                [H, W],
                interpolation=tvt.InterpolationMode.NEAREST,
            )
        ).reshape(H, W)

    return uncovered_masks


def homography(src_image_file: str, tgt_image_file: str) -> np.ndarray:
    """Computes a homography transformation between two images using image features.

    Parameters:
        src_image_file: Input file for the source image.
        tgt_image_file: Input file for the target image.

    Returns:
        The homography matrix to warp the target image to the source image.
    """
    src_image = cv2.imread(src_image_file)
    tgt_image = cv2.imread(tgt_image_file)

    (height, width, _) = src_image.shape

    (src_points, tgt_points) = match_features(src_image, tgt_image)

    # Compute fundamental matrix
    H, mask = cv2.findHomography(tgt_points, src_points, method=cv2.RANSAC)

    return H


def psv(cfg, images, intrinsics, extrinsics, depth_hypos):
    """Performs homography warping to create a Plane Sweeping Volume (PSV).
    Parameters:
        cfg: Configuration dictionary containing configuration parameters.
        images: image maps to be warped into a PSV.
        intrinsics: intrinsics matrices.
        extrinsics: extrinsics matrices.
        depth_hypos: Depth hypotheses to use for homography warping.

    Returns:
        The Plane Sweeping Volume computed via feature matching cost.
    """
    depth_hypos = depth_hypos.squeeze(1)
    _, planes, _, _ = depth_hypos.shape
    B, views, C, H, W = images.shape

    pairwise_psv = []
    for v in range(1, views):
        with torch.no_grad():
            src_proj = torch.matmul(intrinsics[:, v], extrinsics[:, v, 0:3])
            ref_proj = torch.matmul(intrinsics[:, 0], extrinsics[:, 0, 0:3])
            last = torch.tensor([[[0, 0, 0, 1.0]]]).repeat(B, 1, 1).to(images.device)
            src_proj = torch.cat((src_proj, last), 1)
            ref_proj = torch.cat((ref_proj, last), 1)

            proj = torch.matmul(src_proj, torch.inverse(ref_proj))
            rot = proj[:, :3, :3]  # [B,3,3]
            trans = proj[:, :3, 3:4]  # [B,3,1]

            y, x = torch.meshgrid(
                [
                    torch.arange(0, H, dtype=torch.float32, device=images.device),
                    torch.arange(0, W, dtype=torch.float32, device=images.device),
                ],
                indexing="ij",
            )
            y, x = y.contiguous(), x.contiguous()
            y, x = y.view(H * W), x.view(H * W)
            xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
            xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
            rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(
                1, 1, planes, 1
            ) * depth_hypos.view(
                B, 1, planes, H * W
            )  # [B, 3, Ndepth, H*W]
            proj_xyz = rot_depth_xyz + trans.view(B, 3, 1, 1)  # [B, 3, Ndepth, H*W]
            proj_xy = (
                proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]
            )  # [B, 2, Ndepth, H*W]
            proj_x_normalized = proj_xy[:, 0, :, :] / ((W - 1) / 2) - 1
            proj_y_normalized = proj_xy[:, 1, :, :] / ((H - 1) / 2) - 1
            proj_xy = torch.stack(
                (proj_x_normalized, proj_y_normalized), dim=3
            )  # [B, Ndepth, H*W, 2]
            grid = proj_xy
        grid = grid.type(images.dtype)

        warped_src_image = F.grid_sample(
            images[:, v],
            grid.view(B, planes * H, W, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        pairwise_psv.append(warped_src_image.view(B, C, planes, H, W))

    return pairwise_psv


def homogeneous_pixel_coords(batch_size, height, width, device):
    with torch.no_grad():
        y, x = torch.meshgrid(
            [
                torch.arange(
                    0, height, dtype=torch.float32, device=device
                ),
                torch.arange(
                    0, width, dtype=torch.float32, device=device
                ),
            ],
            indexing="ij",
        )
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch_size, 1, 1)  # [B, 3, H*W]
    return xyz

def homography_warp_variance(
    features,
    intrinsics,
    extrinsics,
    hypotheses,
    reference_index,
    memory_mode,
):
    """Performs homography warping to create a Plane Sweeping Volume (PSV).
    Parameters:
        cfg: Configuration dictionary containing configuration parameters.
        features: Feature maps to be warped into a PSV.
        intrinsics: Intrinsics matrices for all views.
        extrinsics: Extrinsics matrices for all views.
        hypotheses: Depth hypotheses to use for homography warping.
        reference_index: The index for the reference view.

    Returns:
        The Plane Sweeping Volume computed via feature matching cost.
    """
    hypotheses = hypotheses.squeeze(1)
    _, planes, _, _ = hypotheses.shape
    batch_size, channels, height, width = features[:,reference_index].shape
    num_views = features.shape[1]
    device = features.device

    ref_volume = features[:,reference_index].unsqueeze(2).repeat(1, 1, planes, 1, 1)

    # maintain rolling variance
    cost_sum = torch.zeros_like(ref_volume)
    cost_sq_sum = torch.zeros_like(ref_volume)
    cost_sum = cost_sum + ref_volume
    cost_sq_sum = cost_sq_sum + torch.pow(ref_volume, 2)

    if memory_mode:
        del ref_volume
        torch.cuda.empty_cache()

    # build reference projection matrix
    ref_proj = torch.matmul(intrinsics[:, reference_index], extrinsics[:, reference_index, 0:3])
    last = torch.tensor([[[0, 0, 0, 1.0]]]).repeat(batch_size, 1, 1).cuda()
    ref_proj = torch.cat((ref_proj, last), 1)

    # build coordinates grid
    y, x = torch.meshgrid(
        [
            torch.arange(0, height, dtype=torch.float32, device=device),
            torch.arange(0, width, dtype=torch.float32, device=device),
        ],
        indexing="ij",
    )
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))
    xyz = torch.unsqueeze(xyz, 0).repeat(batch_size, 1, 1)

    for v in range(num_views):
        if v == reference_index:
            continue

        with torch.no_grad():
            # build source projection matrix
            src_proj = torch.matmul(intrinsics[:, v], extrinsics[:, v, 0:3])
            src_proj = torch.cat((src_proj, last), 1)

            # compute full projection matrix between views
            proj = torch.matmul(src_proj, torch.inverse(ref_proj))
            rot = proj[:, :3, :3]
            trans = proj[:, :3, 3:4]

            # Build plane-sweeping coordinates grid between views
            rot_xyz = torch.matmul(rot, xyz)
            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(
                1, 1, planes, 1
            ) * hypotheses.view(batch_size, 1, planes, height * width)
            proj_xyz = rot_depth_xyz + trans.view(batch_size, 3, 1, 1)
            proj_xy = proj_xyz[:, :2] / proj_xyz[:, 2:3]
            proj_x_normalized = proj_xy[:, 0] / ((width - 1) / 2) - 1
            proj_y_normalized = proj_xy[:, 1] / ((height - 1) / 2) - 1
            proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)
            grid = proj_xy
            grid = grid.type(torch.float32)

        src_feature = features[:,v]
        warped_features = F.grid_sample(
            src_feature,
            grid.view(batch_size, planes * height, width, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        warped_features = warped_features.view(batch_size, channels, planes, height, width)

        cost_sum = torch.add(cost_sum, warped_features)
        cost_sq_sum = torch.add(cost_sq_sum, torch.pow(warped_features, 2))

        if memory_mode:
            del src_feature
            del warped_features
            torch.cuda.empty_cache()

    psv = torch.div(cost_sq_sum, num_views) - torch.pow(torch.div(cost_sum, num_views), 2)

    if memory_mode:
        del cost_sq_sum
        del cost_sum
        torch.cuda.empty_cache()

    return psv


def homography_warp(
    features,
    intrinsics,
    extrinsics,
    hypotheses,
    group_channels,
    vwa_net,
    reference_index,
):
    """Performs homography warping to create a Plane Sweeping Volume (PSV).
    Parameters:
        cfg: Configuration dictionary containing configuration parameters.
        features: Feature maps to be warped into a PSV.
        intrinsics: Intrinsics matrices for all views.
        extrinsics: Extrinsics matrices for all views.
        hypotheses: Depth hypotheses to use for homography warping.
        group_channels: Feature channel sizes used in group-wise correlation (GWC).
        vwa_net: Network used for visibility weighting.
        vis_weights: Pre-computed visibility weights.
        virtual: If True, reference camera is not used in computing feature correlation (virtual reference camera)

    Returns:
        The Plane Sweeping Volume computed via feature matching cost.
    """
    hypotheses = hypotheses.squeeze(1)
    _, planes, _, _ = hypotheses.shape
    batch_size, C, height, width = features[:,reference_index].shape
    num_views = features.shape[1]
    device = features.device

    ref_volume = features[:,reference_index].unsqueeze(2).repeat(1, 1, planes, 1, 1)

    cost_volume_sum = torch.zeros(
        (batch_size, group_channels, planes, height, width),
        dtype=torch.float32,
        device=device,
    )
    view_weights_sum = torch.zeros(
        (batch_size, 1, planes, height, width), dtype=torch.float32, device=device
    )

    # build reference projection matrix
    ref_proj = torch.matmul(intrinsics[:, reference_index], extrinsics[:, reference_index, 0:3])
    last = torch.tensor([[[0, 0, 0, 1.0]]]).repeat(batch_size, 1, 1).cuda()
    ref_proj = torch.cat((ref_proj, last), 1)

    # build coordinates grid
    y, x = torch.meshgrid(
        [
            torch.arange(0, height, dtype=torch.float32, device=device),
            torch.arange(0, width, dtype=torch.float32, device=device),
        ],
        indexing="ij",
    )
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))
    xyz = torch.unsqueeze(xyz, 0).repeat(batch_size, 1, 1)

    for v in range(num_views):
        if v == reference_index:
            continue

        with torch.no_grad():
            # build source projection matrix
            src_proj = torch.matmul(intrinsics[:, v], extrinsics[:, v, 0:3])
            src_proj = torch.cat((src_proj, last), 1)

            # compute full projection matrix between views
            proj = torch.matmul(src_proj, torch.inverse(ref_proj))
            rot = proj[:, :3, :3]
            trans = proj[:, :3, 3:4]

            # Build plane-sweeping coordinates grid between views
            rot_xyz = torch.matmul(rot, xyz)
            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(
                1, 1, planes, 1
            ) * hypotheses.view(batch_size, 1, planes, height * width)
            proj_xyz = rot_depth_xyz + trans.view(batch_size, 3, 1, 1)
            proj_xy = proj_xyz[:, :2] / proj_xyz[:, 2:3]
            proj_x_normalized = proj_xy[:, 0] / ((width - 1) / 2) - 1
            proj_y_normalized = proj_xy[:, 1] / ((height - 1) / 2) - 1
            proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)
            grid = proj_xy
            grid = grid.type(torch.float32)

        src_feature = features[:,v]
        warped_features = F.grid_sample(
            src_feature,
            grid.view(batch_size, planes * height, width, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        warped_features = warped_features.view(batch_size, C, planes, height, width)

        # compute Pairwise Plane-Sweeping Volume using GWC
        pairwise_plane_sweep_volume = groupwise_correlation(warped_features, ref_volume, group_channels)

        view_weights = vwa_net(pairwise_plane_sweep_volume)
        cost_volume_sum = torch.add(cost_volume_sum, (pairwise_plane_sweep_volume * view_weights.unsqueeze(1)))
        view_weights_sum = torch.add(view_weights_sum, view_weights.unsqueeze(1))

    cost_volume = cost_volume_sum.div_(view_weights_sum)

    return cost_volume


#   def homography_warp_coords(cfg, features, level, ref_in, src_in, ref_pose, src_pose, depth_hypos, coords, H, W, gwc_groups, vwa_net=None, vis_weights=None):
#       batch_size, c, h, w = features[0][level].shape
#       _, _, num_planes, _, _ = depth_hypos.shape
#       _, num_pixels, _ = coords.shape
#       num_src_views = len(features)-1
#
#       K_ref = torch.zeros(batch_size, 4, 4).to(ref_in)
#       K_ref[:, :3,:3] = ref_in
#       K_ref[:, 3, 3] = 1
#
#       K_src = torch.zeros(batch_size, num_src_views, 4, 4).to(ref_in)
#       for v in range(num_src_views):
#           K_src[:, v, :3, :3] = src_in[:,v]
#       K_src[:, :, 3, 3] = 1
#
#       vis_weight_list = []
#       cost_volume = None
#       reweight_sum = None
#
#       # build coordinates vector
#       depth_hypos = depth_hypos.reshape(batch_size, num_planes, num_pixels) # batch_size x num_planes x num_pixels
#       coords = torch.movedim(coords, (0,1,2), (0,2,1)).to(torch.float32) # batch_size x 2 x num_pixels
#       x_coords = coords[:,1,:]
#       y_coords = coords[:,0,:]
#       xyz = torch.stack((x_coords, y_coords, torch.ones_like(x_coords)), dim=1) # batch_size, 3 x num_pixels
#
#       # sample reference features
#       x_normalized = ((x_coords / (W-1)) * 2) - 1
#       y_normalized = ((y_coords / (H-1)) * 2) - 1
#       xy = torch.stack((x_normalized, y_normalized), dim=-1)  # [B, num_pixels, 2]
#       ref_features = features[0][level]
#       ref_features = F.grid_sample(ref_features,
#                               xy.view(batch_size, 1, num_pixels, 2),
#                               mode='nearest',
#                               padding_mode='zeros',
#                               align_corners=False)
#       ref_features = ref_features.repeat(1,1,num_planes,1) # [B x C x num_pixels x num_planes]
#
#       for src in range(num_src_views):
#           with torch.no_grad():
#               src_proj = torch.matmul(K_src[:,src],src_pose[:,src])
#               ref_proj = torch.matmul(K_ref,ref_pose)
#
#               proj = torch.matmul(src_proj, torch.inverse(ref_proj))
#               rot = proj[:, :3, :3]  # [B,3,3]
#               trans = proj[:, :3, 3:4]  # [B,3,1]
#
#               rot_xyz = torch.matmul(rot, xyz)  # [B, 3, num_pixels]
#               rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_planes, 1) * depth_hypos.view(batch_size, 1, num_planes, num_pixels)  # [B, 3, num_planes, num_pixels]
#               proj_xyz = rot_depth_xyz + trans.view(batch_size, 3, 1, 1)  # [B, 3, num_planes, num_pixels]
#               proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, num_planes, num_pixels]
#               proj_x_normalized = ((proj_xy[:, 0, :, :] / (W-1)) * 2) - 1
#               proj_y_normalized = ((proj_xy[:, 1, :, :] / (H-1)) * 2) - 1
#
#               proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, num_planes, num_pixels, 2]
#               grid = proj_xy
#
#           grid = grid.type(ref_features.dtype)
#           # Note: The shape of the input map is 4D: Batch x Channels x Height x Width.
#           #       The input coordinates must be in [x, y] format where x->width, y->height.
#           #       These coordinates must be normalized between [-1, 1].
#           src_features = features[src+1][level]
#           src_features = F.grid_sample(src_features,
#                                   grid.view(batch_size, num_planes, num_pixels, 2),
#                                   mode='bilinear',
#                                   padding_mode='zeros',
#                                   align_corners=False)
#           two_view_cost_volume = groupwise_correlation(src_features, ref_features, gwc_groups[level]) #B,C,num_planes,num_pixels
#           two_view_cost_volume = two_view_cost_volume.reshape(batch_size, gwc_groups[level], num_planes, H, W)
#
#           # Estimate visability weight for init level
#           if vwa_net is not None:
#               reweight = vwa_net(two_view_cost_volume) #B, H, W
#               vis_weight_list.append(reweight)
#               reweight = reweight.unsqueeze(1).unsqueeze(2) #B, 1, 1, H, W
#               two_view_cost_volume = reweight*two_view_cost_volume
#
#           # Use estimated visability weights for refine levels
#           elif vis_weights is not None:
#               reweight = vis_weights[src].unsqueeze(1)
#               if reweight.shape[2] < two_view_cost_volume.shape[3]:
#                   reweight = F.interpolate(reweight,scale_factor=2,mode='bilinear',align_corners=False)
#               vis_weight_list.append(reweight.squeeze(1))
#               reweight = reweight.unsqueeze(2)
#               two_view_cost_volume = reweight*two_view_cost_volume
#
#           if cost_volume == None:
#               cost_volume = two_view_cost_volume
#               reweight_sum = reweight
#           else:
#               cost_volume = cost_volume + two_view_cost_volume
#               reweight_sum = reweight_sum + reweight
#
#       cost_volume = cost_volume/(reweight_sum+1e-5)
#       return cost_volume, vis_weight_list


def match_features(
    src_image: np.ndarray, tgt_image: np.ndarray, max_features: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """Computer matching ORB features between a pair of images.

    Args:
        src_image: The source image to compute and match features.
        tgt_image: The target image to compute and match features.
        max_features: The maximum number of features to retain.

    Returns:
        src_points: The set of matched point coordinates for the source image.
        tgt_points: The set of matched point coordinates for the target image.
    """
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    tgt_image = cv2.cvtColor(tgt_image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)

    src_keypoints, src_descriptors = orb.detectAndCompute(src_image, None)
    tgt_keypoints, tgt_descriptors = orb.detectAndCompute(tgt_image, None)

    matcher = cv2.BFMatcher(crossCheck=True)
    matches = list(matcher.match(src_descriptors, tgt_descriptors))
    matches.sort(key=lambda x: x.distance)

    src_points = []
    tgt_points = []
    for i in range(8):
        m = matches[i]

        src_points.append(src_keypoints[m.queryIdx].pt)
        tgt_points.append(tgt_keypoints[m.trainIdx].pt)
    src_points = np.asarray(src_points)
    tgt_points = np.asarray(tgt_points)

    return (src_points, tgt_points)


def plane_coords(K, P, depth_hypos, H, W):
    """Batched PyTorch version"""
    batch_size, _, _ = K.shape
    num_planes = depth_hypos.shape[0]

    xyz = torch.movedim(
        torch.tensor(
            [[0, 0, 1], [W - 1, 0, 1], [0, H - 1, 1], [W - 1, H - 1, 1]],
            dtype=torch.float32,
        ),
        0,
        1,
    ).to(P)
    xyz = xyz.reshape(1, 3, 4).repeat(batch_size, 1, 1)
    if K.shape[1] == 3:
        K_44 = torch.zeros((batch_size, 4, 4)).to(P)
        K_44[:, :3, :3] = K[:, :3, :3]
        K_44[:, 3, 3] = 1
        K = K_44
    proj = K @ P
    inv_proj = torch.linalg.inv(proj)

    planes = torch.zeros(num_planes, 3, 4).to(inv_proj)
    for p in range(num_planes):
        planes[p] = (inv_proj[0, :3, :3] @ xyz) * depth_hypos[p]
        planes[p] += inv_proj[0, :3, 3:4]

    return planes


def _plane_coords(K, P, near, far, H, W):
    """Numpy version"""
    xyz = np.asarray(
        [[0, 0, 1], [W - 1, 0, 1], [0, H - 1, 1], [W - 1, H - 1, 1]], dtype=np.float32
    ).transpose()
    if K.shape[0] == 3:
        K_44 = np.zeros((4, 4))
        K_44[:3, :3] = K[:3, :3]
        K_44[3, 3] = 1
        K = K_44
    proj = K @ P

    near_plane = (np.linalg.inv(proj)[:3, :3] @ xyz) * near
    near_plane += np.linalg.inv(proj)[:3, 3:4]
    far_plane = (np.linalg.inv(proj)[:3, :3] @ xyz) * far
    far_plane += np.linalg.inv(proj)[:3, 3:4]

    return near_plane, far_plane


def _points_from_depth(
    depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """Creates a point array from a single depth map.

    Parameters:
        depth: Depth map to project to 3D.
        intrinsics: Intrinsic camera parameters for the given depth map viewpoint.
        extrinsics: Extrinsic camera parameters for the given depth map viewpoint.

    Returns:
        An array of 3D points corresponding to the input depth map.
    """
    # project depth map to point cloud
    height, width = depth.shape
    x = np.linspace(0, width - 1, num=width)
    y = np.linspace(0, height - 1, num=height)
    x, y = np.meshgrid(x, y, indexing="xy")
    x = x.flatten()
    y = y.flatten()
    depth = depth.flatten()
    valid_inds = np.argwhere(depth > 0)[:, 0]
    xyz_cam = np.matmul(
        np.linalg.inv(intrinsics[:3, :3]), np.vstack((x, y, np.ones_like(x))) * depth
    )
    xyz_world = np.matmul(
        np.linalg.inv(extrinsics), np.vstack((xyz_cam, np.ones_like(x)))
    )[:3]
    points = xyz_world.transpose((1, 0))
    return (points, valid_inds)


def rigid_transform(points: NDArray[Any], transform: NDArray[Any]) -> NDArray[Any]:
    """Apply's a rigid transform to a collection of 3D points.

    Parameters:
        points: Array of 3D points of size [N x 3]
        transform: rigid transform to be pre-multiplied of size [4x4]

    Returns:
        The transformed 3D points.
    """
    N = points.shape[0]
    homog_coords = np.concatenate(points, np.ones_like(points[:, 0:1])).reshape(N, 4, 1)
    transform = np.tile(transform.reshape(1, 4, 4), (N, 1, 1))

    return np.matmul(transform, homog_coords)[:, :3, 0]

def project_points(points: torch.Tensor, depth_map: torch.Tensor, P: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Projects a depth map into a list of 3D points

    Parameters:
        depth: Input depth map to project.
        cam: Camera parameters for input depth map.

    Returns:
        A float Tensor of 3D points corresponding to the projected depth values.
    """
    _, _, height, width = depth_map.shape
    batch_size, N, _ = points.shape
    K_aug = torch.zeros_like(P)
    K_aug[:,:3,:3] = K[:,:3,:3]
    K_aug[:,3,:] = torch.tensor([0,0,0,1])

    # construct projection from camera matrices
    projection = torch.matmul(K_aug, P).to(torch.float32)
    projection = projection.reshape(batch_size, 1,  4, 4).repeat(1,N,1,1)

    points = torch.cat((points, torch.ones_like(points[:,:,0:1])), dim=2).unsqueeze(-1)
    pixels = torch.matmul(projection, points)[:,:,:3,0]
    pixels = pixels.div_(pixels[:,:,2:3])

    proj_x_normalized = pixels[:, :, 0] / ((width - 1) / 2) - 1
    proj_y_normalized = pixels[:, :, 1] / ((height - 1) / 2) - 1
    grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=2).to(torch.float32)

    projected_depths = F.grid_sample(
        depth_map,
        grid.view(batch_size, height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return projected_depths.view(batch_size, 1, height, width)

def project_depth_map(depth: torch.Tensor, P: torch.Tensor, K: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Projects a depth map into a list of 3D points

    Parameters:
        depth: Input depth map to project.
        cam: Camera parameters for input depth map.

    Returns:
        A float Tensor of 3D points corresponding to the projected depth values.
    """
    batch_size, height, width = depth.shape
    K_aug = torch.zeros_like(P)
    K_aug[:,:3,:3] = K[:,:3,:3]
    K_aug[:,3,:] = torch.tensor([0,0,0,1])

    # construct back-projection from invers matrices
    # separate into rotation and translation components
    bwd_projection = torch.matmul(torch.inverse(P), torch.inverse(K_aug)).to(torch.float32)
    bwd_rotation = bwd_projection[:,:3,:3]
    bwd_translation = bwd_projection[:,:3,3:4]

    # build 2D homogeneous coordinates tensor: [B, 3, H*W]
    with torch.no_grad():
        row_span = torch.arange(0, height, dtype=torch.float32).cuda()
        col_span = torch.arange(0, width, dtype=torch.float32).cuda()
        r,c = torch.meshgrid(row_span, col_span, indexing="ij")
        r,c = r.contiguous(), c.contiguous()
        r,c = r.reshape(height*width), c.reshape(height*width)
        coords = torch.stack((c,r,torch.ones_like(c)))
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1)

    # compute 3D coordinates using the depth map: [B, H*W, 3]
    world_coords = torch.matmul(bwd_rotation, coords)
    depth = depth.reshape(batch_size, 1, -1)
    world_coords = world_coords * depth
    world_coords = world_coords + bwd_translation
    world_coords = torch.movedim(world_coords, 1, 2)

    # mask world points optionally
    if (mask != None):
        mask = mask.reshape(batch_size, -1)
        world_coords = world_coords[mask].unsqueeze(0)

    return world_coords

def project_depth_map_np(depth: NDArray[np.float32], P: NDArray[np.float32], K: NDArray[np.float32], mask: NDArray[np.float32] | None = None) -> NDArray[np.float32]:
    """Projects a depth map into a list of 3D points

    Parameters:
        depth: Input depth map to project.
        cam: Camera parameters for input depth map.

    Returns:
        A float Tensor of 3D points corresponding to the projected depth values.
    """
    height, width = depth.shape
    K_aug = np.zeros_like(P)
    K_aug[:3,:3] = K[:3,:3]
    K_aug[3,:] = np.array([0,0,0,1])

    # construct back-projection from invers matrices
    # separate into rotation and translation components
    bwd_projection = np.matmul(np.linalg.inv(P), np.linalg.inv(K_aug)).astype(np.float32)
    bwd_rotation = bwd_projection[:3,:3]
    bwd_translation = bwd_projection[:3,3:4]

    # build 2D homogeneous coordinates tensor: [B, 3, H*W]
    
    row_span = np.arange(0, height, dtype=np.float32)
    col_span = np.arange(0, width, dtype=np.float32)
    r,c = np.meshgrid(row_span, col_span, indexing="ij")
    r,c = r.reshape(height*width), c.reshape(height*width)
    coords = np.stack((c,r,np.ones_like(c)))

    # compute 3D coordinates using the depth map: [B, H*W, 3]
    world_coords = np.matmul(bwd_rotation, coords)
    depth = depth.reshape(1, -1)
    world_coords = world_coords * depth
    world_coords = world_coords + bwd_translation
    world_coords = np.moveaxis(world_coords, 0, 1)

    # filter world coordinates where depth is <= 0
    depth = depth.squeeze(0)
    filter_mask = depth > 0
    
    # mask world points optionally
    if (mask != None):
        filter_mask *= mask.reshape(-1)

    return world_coords[depth > 0]
    # return world_coords


def project_renderer(
    renderer: o3d.visualization.rendering.OffscreenRenderer,
    K: np.ndarray,
    P: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray:
    """Projects the scene in an Open3D Offscreen Renderer to the 2D image plane.

    Parameters:
        renderer: Geometric scene to be projected.
        K: Camera intrinsic parameters.
        P: Camera extrinsic parameters.
        width: Desired image width.
        height: Desired image height.

    Returns:
        The rendered image for the scene at the specified camera viewpoint.
    """
    # set up the renderer
    intrins = o3d.camera.PinholeCameraIntrinsic(
        width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    )
    renderer.setup_camera(intrins, P)

    # render image
    image = np.asarray(renderer.render_to_image())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


# def render_custom_values(points: NDArray[], values: np.ndarray, image_shape: Tuple[int,int], cam: np.ndarray) -> np.ndarray:
#     """Renders a point cloud into a 2D camera plane using custom values for each pixel.

#     Parameters:
#         points: List of 3D points to be rendered.
#         values: List of values to be written in the rendered image.
#         image_shape: Desired shape (height,width) of the rendered image.
#         cam: Camera parameters for the image viewpoint.

#     Returns:
#         The rendered image for the list of points using the sepcified corresponding values.
#     """
#     rendered_img = rd.render(list(image_shape), points.tolist(), values.astype(float).tolist(), cam.flatten().tolist())

#     return rendered_img

# def _render_point_cloud(cloud: o3d.geometry.PointCloud, pose: np.ndarray, K: np.ndarray, width: int, height: int) -> np.ndarray:
#     """Renders a point cloud into a 2D image plane.

#     Parameters:
#         cloud: Point cloud to be rendered.
#         pose: Camera extrinsic parameters for the image plane.
#         K: Camera intrinsic parameters for the image plane.
#         width: Desired width of the rendered image.
#         height: Desired height of the rendered image.

#     Returns:
#         The rendered image for the point cloud at the specified camera viewpoint.
#     """
#     #   cmap = plt.get_cmap("hot_r")
#     #   colors = cmap(dists)[:, :3]
#     #   ply.colors = o3d.utility.Vector3dVector(colors)

#     # set up the renderer
#     render = o3d.visualization.rendering.OffscreenRenderer(width, height)
#     mat = o3d.visualization.rendering.MaterialRecord()
#     mat.shader = 'defaultUnlit'
#     render.scene.add_geometry("cloud", cloud, mat)
#     render.scene.set_background(np.asarray([0,0,0,1])) #r,g,b,a
#     intrins = o3d.camera.PinholeCameraIntrinsic(width, height, K[0,0], K[1,1], K[0,2], K[1,2])
#     render.setup_camera(intrins, pose)

#     # render image
#     image = np.asarray(render.render_to_image())
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     depth = np.asarray(render.render_to_depth_image(z_in_view_space=True))

#     return image, depth

# def render_point_cloud_single(cloud: o3d.geometry.PointCloud, pose: np.ndarray, K: np.ndarray, width: int, height: int) -> np.ndarray:
#     """Renders a point cloud into a 2D image plane.

#     Parameters:
#         cloud: Point cloud to be rendered.
#         pose: Camera extrinsic parameters for the image plane.
#         K: Camera intrinsic parameters for the image plane.
#         width: Desired width of the rendered image.
#         height: Desired height of the rendered image.

#     Returns:
#         The rendered image for the point cloud at the specified camera viewpoint.
#     """
#     #   cmap = plt.get_cmap("hot_r")
#     #   colors = cmap(dists)[:, :3]
#     #   ply.colors = o3d.utility.Vector3dVector(colors)

#     # set up the renderer
#     render = o3d.visualization.rendering.OffscreenRenderer(width, height)
#     mat = o3d.visualization.rendering.MaterialRecord()
#     mat.shader = 'defaultUnlit'
#     render.scene.add_geometry("cloud", cloud, mat)
#     render.scene.set_background(np.asarray([0,0,0,1])) #r,g,b,a
#     intrins = o3d.camera.PinholeCameraIntrinsic(width, height, K[0,0], K[1,1], K[0,2], K[1,2])
#     render.setup_camera(intrins, pose)

#     # render image
#     image = np.asarray(render.render_to_image())
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     depth = np.asarray(render.render_to_depth_image(z_in_view_space=True))

#     return image, depth

# def render_point_cloud(render, intrins, pose):
#     """Renders a point cloud into a 2D image plane.

#     Parameters:

#     Returns:
#     """
#     render.setup_camera(intrins, pose)

#     # render image
#     image = np.asarray(render.render_to_image())
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     depth = np.asarray(render.render_to_depth_image(z_in_view_space=True))

#     return image, depth


def reproject(
    src_depth: NDArray[Any] | Tensor,
    src_K: NDArray[Any] | Tensor,
    src_P: NDArray[Any] | Tensor,
    tgt_depth: NDArray[Any] | Tensor,
    tgt_K: NDArray[Any] | Tensor,
    tgt_P: NDArray[Any] | Tensor,
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]] | Tuple[Tensor, Tensor, Tensor]:
    if isinstance(src_depth, Tensor):
        assert isinstance(src_K, Tensor)
        assert isinstance(src_P, Tensor)
        assert isinstance(tgt_depth, Tensor)
        assert isinstance(tgt_K, Tensor)
        assert isinstance(tgt_P, Tensor)
        return _reproject_torch(src_depth, src_K, src_P, tgt_depth, tgt_K, tgt_P)
    elif isinstance(src_depth, np.ndarray):
        assert isinstance(src_K, np.ndarray)
        assert isinstance(src_P, np.ndarray)
        assert isinstance(tgt_depth, np.ndarray)
        assert isinstance(tgt_K, np.ndarray)
        assert isinstance(tgt_P, np.ndarray)
        return _reproject_numpy(src_depth, src_K, src_P, tgt_depth, tgt_K, tgt_P)
    else:
        raise Exception(f"Unknown data type '{type(src_depth)}'")


def _reproject_torch(
    src_depth: Tensor,
    src_K: Tensor,
    src_P: Tensor,
    tgt_depth: Tensor,
    tgt_K: Tensor,
    tgt_P: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Computes the re-projection depth values and pixel indices between two depth maps.

    This function takes as input two depth maps: 'src_depth' and 'tgt_depth'. The source
    depth map is first projected into the target camera plane using the source depth
    values and the camera parameters for both views. Using the projected pixel
    coordinates in the target view, the target depths are then re-projected back into
    the source camera plane (again with the camera parameters for both views). The
    information prouced from this process is often used to compute errors in
    re-projection between two depth maps, or similar operations.

    Parameters:
        src_depth: Source depth map to be projected.
        src_K: Intrinsic camera parameters for the source depth map viewpoint.
        src_P: Extrinsic camera parameters for the source depth map viewpoint.
        tgt_depth: Target depth map used for re-projection.
        tgt_K: Intrinsic camera parameters for the target depth map viewpoint.
        tgt_P: Extrinsic camera parameters for the target depth map viewpoint.

    Returns:
        depth_reprojected: The re-projected depth values for the source depth map.
        coords_reprojected: The re-projection coordinates for the source view.
        coords_tgt: The projected coordinates for the target view.
    """
    batch_size, c, height, width = src_depth.shape

    # back-project ref depths to 3D
    x_src, y_src = torch.meshgrid(
        torch.arange(0, width), torch.arange(0, height), indexing="xy"
    )
    x_src = x_src.reshape(-1).unsqueeze(0).repeat(batch_size, 1).to(src_depth)
    y_src = y_src.reshape(-1).unsqueeze(0).repeat(batch_size, 1).to(src_depth)
    homog = torch.stack((x_src, y_src, torch.ones_like(x_src)), dim=1)
    xyz_src = torch.matmul(
        torch.linalg.inv(src_K), homog * src_depth.reshape(batch_size, 1, -1)
    )

    # transform 3D points from ref to src coords
    homog_3d = torch.concatenate((xyz_src, torch.ones_like(x_src).unsqueeze(1)), dim=1)
    xyz_tgt = torch.matmul(torch.matmul(tgt_P, torch.linalg.inv(src_P)), homog_3d)[
        :, :3
    ]

    # project src 3D points into pixel coords
    K_xyz_tgt = torch.matmul(tgt_K, xyz_tgt)
    xy_tgt = K_xyz_tgt[:, :2] / K_xyz_tgt[:, 2:3]
    x_tgt = xy_tgt[:, 0].reshape(batch_size, height, width).to(torch.float32)
    y_tgt = xy_tgt[:, 1].reshape(batch_size, height, width).to(torch.float32)
    coords_tgt = torch.stack((x_tgt, y_tgt), dim=-1)  # B x H x W x 2

    # sample the depth values from the src map at each pixel coord
    x_normalized = ((x_tgt / (width - 1)) * 2) - 1
    y_normalized = ((y_tgt / (height - 1)) * 2) - 1
    grid = torch.stack((x_normalized, y_normalized), dim=-1)  # B x H x W x 2
    sampled_depth_tgt = F.grid_sample(
        tgt_depth, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    # back-project src depths to 3D
    homog = torch.concatenate((xy_tgt, torch.ones_like(x_src).unsqueeze(1)), dim=1)
    xyz_tgt = torch.matmul(
        torch.linalg.inv(tgt_K), homog * sampled_depth_tgt.reshape(batch_size, 1, -1)
    )

    # transform 3D points from src to ref coords
    homog_3d = torch.concatenate((xyz_tgt, torch.ones_like(x_src).unsqueeze(1)), dim=1)
    xyz_reprojected = torch.matmul(
        torch.matmul(src_P, torch.linalg.inv(tgt_P)), homog_3d
    )[:, :3]

    # extract reprojected depth values
    depth_reprojected = (
        xyz_reprojected[:, 2].reshape(batch_size, height, width).to(torch.float32)
    )

    # project ref 3D points into pixel coords
    K_xyz_reprojected = torch.matmul(src_K, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:, :2] / (K_xyz_reprojected[:, 2:3] + 1e-7)
    x_reprojected = (
        xy_reprojected[:, 0].reshape(batch_size, height, width).to(torch.float32)
    )
    y_reprojected = (
        xy_reprojected[:, 1].reshape(batch_size, height, width).to(torch.float32)
    )

    coords_reprojected = torch.stack(
        (x_reprojected, y_reprojected), dim=-1
    )  # B x H x W x 2

    return (depth_reprojected, coords_reprojected, coords_tgt)


def _reproject_numpy(
    src_depth: NDArray[Any],
    src_K: NDArray[Any],
    src_P: NDArray[Any],
    tgt_depth: NDArray[Any],
    tgt_K: NDArray[Any],
    tgt_P: NDArray[Any],
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Computes the re-projection depth values and pixel indices between two depth maps.

    This function takes as input two depth maps: 'src_depth' and 'tgt_depth'. The source
    depth map is first projected into the target camera plane using the source depth
    values and the camera parameters for both views. Using the projected pixel
    coordinates in the target view, the target depths are then re-projected back into
    the source camera plane (again with the camera parameters for both views). The
    information prouced from this process is often used to compute errors in
    re-projection between two depth maps, or similar operations.

    Parameters:
        src_depth: Source depth map to be projected.
        src_K: Intrinsic camera parameters for the source depth map viewpoint.
        src_P: Extrinsic camera parameters for the source depth map viewpoint.
        tgt_depth: Target depth map used for re-projection.
        tgt_K: Intrinsic camera parameters for the target depth map viewpoint.
        tgt_P: Extrinsic camera parameters for the target depth map viewpoint.

    Returns:
        depth_reprojected: The re-projected depth values for the source depth map.
        coords_reprojected: The re-projection coordinates for the source view.
        coords_tgt: The projected coordinates for the target view.
    """
    height, width = src_depth.shape

    # back-project ref depths to 3D
    x_src, y_src = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_src, y_src = x_src.reshape([-1]), y_src.reshape([-1])
    xyz_src = np.matmul(
        np.linalg.inv(src_K),
        np.vstack((x_src, y_src, np.ones_like(x_src))) * src_depth.reshape([-1]),
    )

    # transform 3D points from ref to src coords
    xyz_tgt = np.matmul(
        np.matmul(tgt_P, np.linalg.inv(src_P)),
        np.vstack((xyz_src, np.ones_like(x_src))),
    )[:3]

    # project src 3D points into pixel coords
    K_xyz_tgt = np.matmul(tgt_K, xyz_tgt)
    xy_tgt = K_xyz_tgt[:2] / K_xyz_tgt[2:3]
    x_tgt = xy_tgt[0].reshape([height, width]).astype(np.float32)
    y_tgt = xy_tgt[1].reshape([height, width]).astype(np.float32)

    # sample the depth values from the src map at each pixel coord
    sampled_depth_tgt = cv2.remap(
        tgt_depth, x_tgt, y_tgt, interpolation=cv2.INTER_LINEAR
    )

    # back-project src depths to 3D
    xyz_tgt = np.matmul(
        np.linalg.inv(tgt_K),
        np.vstack((xy_tgt, np.ones_like(x_src))) * sampled_depth_tgt.reshape([-1]),
    )

    # transform 3D points from src to ref coords
    xyz_reprojected = np.matmul(
        np.matmul(src_P, np.linalg.inv(tgt_P)),
        np.vstack((xyz_tgt, np.ones_like(x_src))),
    )[:3]

    # extract reprojected depth values
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)

    # project ref 3D points into pixel coords
    K_xyz_reprojected = np.matmul(src_K, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / (K_xyz_reprojected[2:3] + 1e-7)
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    coords_reprojected = np.dstack((x_reprojected, y_reprojected))
    coords_tgt = np.dstack((x_tgt, y_tgt))

    return (depth_reprojected, coords_reprojected, coords_tgt)


def sample_volume(volume, z_vals, coords, H, W, near_depth, far_depth, inv_depth):
    """
    Parameters:

    Returns:
    """
    N, M = z_vals.shape
    batch_size, c, _, _, _ = volume.shape

    z_vals = z_vals.reshape(N, M, 1)  # N x M x 1
    if inv_depth:
        z_vals = 1 / z_vals
        near_depth = 1 / near_depth
        far_depth = 1 / far_depth
    coords = coords.reshape(N, 1, 2).repeat(1, M, 1)  # N x M x 2
    x_coords = coords[:, :, 1:2]
    y_coords = coords[:, :, 0:1]
    points = torch.cat([x_coords, y_coords, z_vals], dim=-1)  # N x M x 3
    points = torch.reshape(points, [-1, 3])  # N*M x 3

    # define coordinates bounds
    min_coord = torch.tensor([0, 0, near_depth]).to(points)
    max_coord = torch.tensor([W - 1, H - 1, far_depth]).to(points)
    min_coord = min_coord.reshape(1, 3).repeat(N * M, 1)
    max_coord = max_coord.reshape(1, 3).repeat(N * M, 1)

    # normalize points
    norm_points = (points - min_coord) / (max_coord - min_coord)
    norm_points = norm_points.unsqueeze(0).repeat(batch_size, 1, 1)
    norm_points = (norm_points * 2) - 1

    # Note: The shape of the input volume is 5D: Batch x Channels x Depth x Height x Width.
    #       The input coordinates must be in [x, y, z] format where x->width, y->height, z->depth.
    #       These coordinates must be normalized between [-1, 1].
    features = F.grid_sample(
        volume,
        norm_points.view(batch_size, N * M, 1, 1, 3),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    features = torch.movedim(features.reshape(c, N * M), 0, 1)  # N*M x c

    return features


def soft_hypothesis(data, target_hypo, focal_length, min_hypo, max_hypo, M, delta_in=1):
    """
    Parameters:

    Returns:
    """
    B, _, D, H, W = target_hypo.shape
    rand_match_offset = torch.rand(B, 1, M, H, W).to(target_hypo)
    near, far = z_planes_from_disp(
        target_hypo, data["baseline"], focal_length, delta=delta_in
    )
    target_range = torch.abs(far - near).repeat(1, 1, M, 1, 1)

    target_samples = (rand_match_offset * target_range) + near
    mask = torch.where(target_hypo <= 0, 0.0, 1.0).repeat(1, 1, M, 1, 1)
    matching_hypos = torch.clip(target_samples, min_hypo, max_hypo) * mask

    return matching_hypos


def soft_hypothesis_numpy(
    data, target_hypo, focal_length, min_hypo, max_hypo, M=1, delta_in=1
):
    """
    Parameters:

    Returns:
    """
    B, _, D, H, W = target_hypo.shape
    rand_match_offset = torch.rand(B, 1, M, H, W).to(target_hypo)
    rand_near_offset = torch.rand(B, 1, M, H, W).to(target_hypo)
    rand_far_offset = torch.rand(B, 1, M, H, W).to(target_hypo)

    near, far = z_planes_from_disp(
        target_hypo, data["baseline"], focal_length, delta=delta_in
    )
    target_range = torch.abs(far - near).repeat(1, 1, M, 1, 1)
    near_range = torch.abs(near - min_hypo).repeat(1, 1, M, 1, 1)
    far_range = torch.abs(max_hypo - far).repeat(1, 1, M, 1, 1)

    target_samples = (rand_match_offset * target_range) + near
    near_samples = (rand_near_offset * near_range) + min_hypo
    far_samples = (rand_far_offset * far_range) + far
    samples = torch.cat([target_samples, near_samples, far_samples], dim=1)
    samples = samples.reshape(B, -1, H, W).unsqueeze(1)  # [B, 1, M*3, H, W]

    mask = torch.where(target_hypo <= 0, 0.0, 1.0).repeat(1, 1, M * 3, 1, 1)
    hypos = torch.clip(samples, min_hypo, max_hypo) * mask

    return hypos


# def resolution_based_hypothesis(data, target_hypo, level, focal_length, min_hypo, max_hypo, delta_in=1):
#     """
#     Parameters:

#     Returns:
#     """
#     B, _, _, H, W = target_hypo.shape
#     rand_match_offset = torch.rand(B,1,level,H,W).to(target_hypo)
#     rand_near_offset = torch.rand(B,1,level,H,W).to(target_hypo)
#     rand_far_offset = torch.rand(B,1,level,H,W).to(target_hypo)

#     near, far = z_planes_from_disp(target_hypo, data["baseline"], focal_length, delta=delta_in)
#     target_range = torch.abs(far - near).repeat(1,1,level,1,1)
#     near_range = torch.abs(near - min_hypo).repeat(1,1,level,1,1)
#     far_range = torch.abs(max_hypo - far).repeat(1,1,level,1,1)

#     target_samples = (rand_match_offset * target_range) + near
#     near_samples = (rand_near_offset * near_range) + min_hypo
#     far_samples = (rand_far_offset * far_range) + far
#     samples = torch.cat([target_samples,near_samples,far_samples], dim=1)
#     samples = samples.reshape(B,-1,H,W).unsqueeze(1) # [B, 1, M*3, H, W]

#     mask = torch.where(target_hypo <= 0, 0.0, 1.0).repeat(1,1,level*3,1,1)
#     hypos = torch.clip(samples, min_hypo, max_hypo) * mask

#     return hypos


def visibility(depths, K, Ps, vis_th, levels=4):
    """
    Parameters:

    Returns:
    """
    _, views, _, H, W = depths.shape

    K_pyr = intrinsic_pyramid(K, levels)

    vis_maps = []
    vis_masks = []
    for l in range(levels):
        resized_depths = tvf.resize(
            depths[:, :, 0], [int(H / (2**l)), int(W / (2**l))]
        ).unsqueeze(2)
        # views = resized_depths.shape[1]
        vis_map = torch.where(resized_depths[:, 0] > 0.0, 1, 0)

        for i in range(1, views):
            mask = geometric_consistency_mask(
                resized_depths[:, 0],
                K_pyr[:, l],
                Ps[:, 0],
                resized_depths[:, i],
                K_pyr[:, l],
                Ps[:, i],
                pixel_th=0.5,
            )
            assert isinstance(mask, Tensor)
            vis_map += mask.unsqueeze(1)
        vis_map = vis_map.to(torch.float32)

        vis_maps.append(vis_map)
        vis_masks.append(torch.where(vis_map >= vis_th, 1, 0))
    return vis_maps, vis_masks


def visibility_numpy(depths, reference_index, K, Ps, pix_th=0.50, vis_th=None):
    """
    Parameters:

    Returns:
    """
    views = depths.shape[0]
    vis_map = np.where(depths[reference_index] > 0.0, 1, 0)

    for i in range(views):
        if i == reference_index:
            continue
        mask = geometric_consistency_mask(
            depths[reference_index], K, Ps[reference_index], depths[i], K, Ps[i], pixel_th=pix_th
        )
        vis_map += mask
    vis_map = vis_map.astype(np.float32)

    if vis_th != None:
        vis_map = np.where(vis_map >= vis_th, 1, 0)

    return vis_map

# def visibility_torch(depths, reference_index, K, Ps, vis_th=None):
#     """
#     Parameters:

#     Returns:
#     """
#     views = depths.shape[0]
#     vis_map = torch.where(depths[reference_index] > 0.0, 1, 0)

#     for i in range(views):
#         if i == reference_index:
#             continue
#         mask = geometric_consistency_mask(
#             depths[reference_index], K, Ps[reference_index], depths[i], K, Ps[i], pixel_th=0.50
#         )
#         vis_map += mask
#     vis_map = vis_map.to(torch.float32)

#     if vis_th != None:
#         vis_map = torch.where(vis_map >= vis_th, 1, 0)

#     return vis_map


def visibility_mask(
    src_depth: np.ndarray,
    src_cam: np.ndarray,
    depth_files: List[str],
    cam_files: List[str],
    src_ind: int = -1,
    pixel_th: float = 0.1,
) -> np.ndarray:
    """Computes a visibility mask between a provided source depth map and list of target depth maps.

    Parameters:
        src_depth: Depth map for the source view.
        src_cam: Camera parameters for the source depth map viewpoint.
        depth_files: List of target depth maps.
        cam_files: List of corresponding target camera parameters for each targte depth map viewpoint.
        src_ind: Index into 'depth_files' corresponding to the source depth map (if included in the list).
        pixel_th: Pixel re-projection threshold to determine matching depth estimates.

    Returns:
        The visibility mask for the source view.
    """
    vis_map = np.not_equal(src_depth, 0.0).astype(np.double)

    for i in range(len(depth_files)):
        if i == src_ind:
            continue

        # get files
        sdf = depth_files[i]
        scf = cam_files[i]

        # load data
        tgt_depth = read_pfm(sdf)
        tgt_cam = read_single_cam_sfm(scf)

        mask = geometric_consistency_mask(
            src_depth,
            src_cam[1],
            src_cam[0],
            tgt_depth,
            tgt_cam[1],
            tgt_cam[0],
            pixel_th,
        )
        vis_map += mask

    return vis_map.astype(np.float32)


def uniform_hypothesis(
    cfg,
    device,
    batch_size,
    depth_min,
    depth_max,
    img_height,
    img_width,
    planes,
    inv_depth=False,
    bin_format=False,
):
    """
    Parameters:

    Returns:
    """
    depth_range = depth_max - depth_min

    hypotheses = torch.zeros((batch_size, planes), device=device)
    for b in range(0, batch_size):
        if bin_format:
            spacing = depth_range / planes
            start_depth = depth_min + (spacing / 2)
            end_depth = depth_min + (spacing / 2) + ((planes - 1) * spacing)
        else:
            start_depth = depth_min
            end_depth = depth_max
        if inv_depth:
            hypotheses[b] = 1 / (
                torch.linspace(
                    1 / start_depth, 1 / end_depth, steps=planes, device=device
                )
            )
        else:
            hypotheses[b] = torch.linspace(
                start_depth, end_depth, steps=planes, device=device
            )
    hypotheses = (
        hypotheses.unsqueeze(2).unsqueeze(3).repeat(1, 1, img_height, img_width)
    )

    # Make coordinate for depth hypothesis, to be used by sparse convolution.
    depth_hypo_coords = torch.zeros((batch_size, planes), device=device)
    for b in range(0, batch_size):
        depth_hypo_coords[b] = torch.linspace(
            0, planes - 1, steps=planes, device=device
        )
    depth_hypo_coords = (
        depth_hypo_coords.unsqueeze(2).unsqueeze(3).repeat(1, 1, img_height, img_width)
    )

    # Calculate hypothesis interval
    hypo_intervals = hypotheses[:, 1:] - hypotheses[:, :-1]
    hypo_intervals = torch.cat(
        (hypo_intervals, hypo_intervals[:, -1].unsqueeze(1)), dim=1
    )

    return (
        hypotheses.unsqueeze(1),
        depth_hypo_coords.unsqueeze(1),
        hypo_intervals.unsqueeze(1),
    )


def z_planes_from_disp(
    Z: Tensor, b: Tensor, f: Tensor, delta: float
) -> Tuple[Tensor, Tensor]:
    """Computes the near and far Z planes corresponding to 'delta' disparity steps between two cameras.

    Parameters:
        Z: Z buffer storing D depth plane hypotheses [B x C x D x H x W]. (shape resembles a typical PSV).
        b: The baseline between cameras [B].
        f: The focal length of camera [B].
        delta: The disparity delta for the near and far planes.

    Returns:
        The tuple of near and far Z planes corresponding to 'delta' disparity steps.
    """
    B, C, D, H, W = Z.shape
    b = b.reshape(B, 1, 1, 1, 1).repeat(1, C, D, H, W)
    f = f.reshape(B, 1, 1, 1, 1).repeat(1, C, D, H, W)

    near = Z * (b * f / ((b * f) + (delta * Z)))
    far = Z * (b * f / ((b * f) - (delta * Z)))

    return (near, far)
