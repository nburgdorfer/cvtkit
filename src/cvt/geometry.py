# cvt/geometry.py

"""Module including geometric routines.

This module contains the following functions:

- `essential_from_features(src_image_file, tgt_image_file, K)` - Computes the essential matrix between two images using image features.
- `fundamental_from_KP(K, P_src, P_tgt)` - Computes the fundamental matrix between two images using camera parameters.
- `fundamental_from_features(src_image_file, tgt_image_file)` - Computes the fundamental matrix between two images using image features.
- `geometric_consistency_mask(src_depth, src_cam, tgt_depth, tgt_cam, pixel_th)` - Computes the geometric consistency mask between a source and target depth map.
- `homography(src_image_file, tgt_image_file)` - Computes a homography transformation between two images using image features.
- `match_features(src_image, tgt_image, max_features)` - Computer matching ORB features between a pair of images.
- `point_cloud_from_depth(depth, cam, color)` - Creates a point cloud from a single depth map.
- `points_from_depth(depth, cam)` - Creates a point array from a single depth map.
- `project_depth_map(depth, cam, mask)` - Projects a depth map into a list of 3D points.
- `project_renderer(renderer, K, P, width, height)` - Projects the scene in an Open3D Offscreen Renderer to the 2D image plane.
- `render_custom_values(points, values, image_shape, cam)` - Renders a point cloud into a 2D camera plane using custom values for each pixel.
- `render_point_cloud(cloud, cam, width, height)` - Renders a point cloud into a 2D image plane.
- `reproject(src_depth, src_cam, tgt_depth, tgt_cam)` - Computes the re-projection depth values and pixel indices between two depth maps.
- `visibility_mask(src_depth, src_cam, depth_files, cam_files, src_ind=-1, pixel_th=0.1)` - Computes a visibility mask between a provided source depth map and list of target depth maps.
"""

import numpy as np
import cv2
import open3d as o3d
from typing import Tuple, List, Optional
import torch
import torch.nn.functional as F

import rendering as rd
from io import *

def essential_from_features(src_image_file: str, tgt_image_file: str, K: np.ndarray) -> np.ndarray:
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

def fundamental_from_KP(K: np.ndarray, P_src: np.ndarray, P_tgt: np.ndarray) -> np.ndarray:
    """Computes the fundamental matrix between two images using camera parameters.

    Parameters:
        K: Intrinsics matrix of the two cameras (assumed to be constant between views).
        P_src: Extrinsics matrix for the source view.
        P_tgt: Extrinsics matrix for the target view.

    Returns:
        The fundamental matrix betweent the two cameras.
    """
    R1 = P_src[0:3,0:3]
    t1 = P_src[0:3,3]
    R2 = P_tgt[0:3,0:3]
    t2 = P_tgt[0:3,3]

    t1aug = np.array([t1[0], t1[1], t1[2], 1])
    epi2 = np.matmul(P_tgt,t1aug)
    epi2 = np.matmul(K,epi2[0:3])

    R = np.matmul(R2,np.transpose(R1))
    t= t2- np.matmul(R,t1)
    K1inv = np.linalg.inv(K)
    K2invT = np.transpose(K1inv)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.matmul(K2invT,np.matmul(tx,np.matmul(R,K1inv)))
    F = F/np.amax(F)

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
    F, mask = cv2.findFundamentalMat(src_points,tgt_points,cv2.FM_8POINT)

    return F

def geometric_consistency_mask(src_depth: np.ndarray, src_cam: np.ndarray, tgt_depth: np.ndarray, tgt_cam: np.ndarray, pixel_th: float) -> np.ndarray:
    """Computes the geometric consistency mask between a source and target depth map.

    Parameters:
        src_depth: Depth map for the source view.
        src_cam: Camera parameters for the source depth map viewpoint.
        tgt_depth: Depth map for the target view.
        tgt_cam: Camera parameters for the target depth map viewpoint.
        pixel_th: Pixel re-projection threshold to determine matching depth estimates.

    Returns:
        The binary consistency mask encoding depth consensus between source and target depth maps.
    """
    height, width = src_depth.shape
    x_src, y_src = np.meshgrid(np.arange(0, width), np.arange(0, height))

    depth_reprojected, coords_reprojected, coords_tgt, projection_map = reproject(src_depth, src_cam, tgt_depth, tgt_cam)

    dist = np.sqrt((coords_reprojected[:,:,0] - x_src) ** 2 + (coords_reprojected[:,:,1] - y_src) ** 2)
    mask = np.where(dist < pixel_th, 1, 0)

    return mask

def geometric_consistency_error(src_depth: np.ndarray, src_cam: np.ndarray, tgt_depth: np.ndarray, tgt_cam: np.ndarray) -> np.ndarray:
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

    depth_reprojected, coords_reprojected, coords_tgt, projection_map = reproject(src_depth, src_cam, tgt_depth, tgt_cam)

    dist = np.sqrt((coords_reprojected[:,:,0] - x_src) ** 2 + (coords_reprojected[:,:,1] - y_src) ** 2)

    return dist, projection_map

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

def match_features(src_image: np.ndarray, tgt_image: np.ndarray, max_features: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Computer matching ORB features between a pair of images.

    Args:
        src_image: The source image to compute and match features.
        tgt_image: The target image to compute and match features.
        max_features: The maximum number of features to retain.

    Returns:
        src_points: The set of matched point coordinates for the source image.
        tgt_points: The set of matched point coordinates for the target image.
    """
    src_image = cv2.cvtColor(src_image,cv2.COLOR_BGR2GRAY)
    tgt_image = cv2.cvtColor(tgt_image, cv2.COLOR_BGR2GRAY)
      
    orb = cv2.ORB_create(max_features)
      
    src_keypoints, src_descriptors = orb.detectAndCompute(src_image,None)
    tgt_keypoints, tgt_descriptors = orb.detectAndCompute(tgt_image,None)

    matcher = cv2.BFMatcher(crossCheck=True)
    matches = list(matcher.match(src_descriptors, tgt_descriptors) )
    matches.sort(key = lambda x:x.distance)

    src_points = []
    tgt_points = []
    for i in range(8):
        m = matches[i]

        src_points.append(src_keypoints[m.queryIdx].pt)
        tgt_points.append(tgt_keypoints[m.trainIdx].pt)
    src_points  = np.asarray(src_points)
    tgt_points = np.asarray(tgt_points)

    return (src_points, tgt_points)

def point_cloud_from_depth(depth: np.ndarray, cam: np.ndarray, color: np.ndarray) -> o3d.geometry.PointCloud:
    """Creates a point cloud from a single depth map.

    Parameters:
        depth: Depth map to project to 3D.
        cam: Camera parameters for the given depth map viewpoint.
        color: Color [R,G,B] used for all points in the generated point cloud.
    """
    cloud = o3d.geometry.PointCloud()

    # extract camera params
    height, width = depth.shape
    fx = cam[1,0,0]
    fy = cam[1,1,1]
    cx = cam[1,0,2]
    cy = cam[1,1,2]
    intrins = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    extrins = cam[0]

    # convert deth to o3d.geometry.Image
    depth_map = o3d.geometry.Image(depth)

    # project depth map to 3D
    cloud = cloud.create_from_depth_image(depth_map, intrins, extrins, depth_scale=1.0, depth_trunc=1000)

    # color point cloud
    colors = o3d.utility.Vector3dVector(np.full((len(cloud.points), 3), color))
    cloud.colors = colors
    
    return cloud

def points_from_depth(depth: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Creates a point array from a single depth map.

    Parameters:
        depth: Depth map to project to 3D.
        cam: Camera parameters for the given depth map viewpoint.

    Returns:
        An array of 3D points corresponding to the input depth map.
    """
    # project depth map to point cloud
    height, width = depth.shape
    x = np.linspace(0,width-1,num=width)
    y = np.linspace(0,height-1,num=height)
    x,y = np.meshgrid(x,y, indexing="xy")
    x = x.flatten()
    y = y.flatten()
    depth = depth.flatten()
    xyz_cam = np.matmul(np.linalg.inv(cam[1,:3,:3]), np.vstack((x, y, np.ones_like(x))) * depth)
    xyz_world = np.matmul(np.linalg.inv(cam[0,:4,:4]), np.vstack((xyz_cam, np.ones_like(x))))[:3]
    points = xyz_world.transpose((1, 0))
    return points


def project_depth_map(depth: torch.Tensor, cam: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Projects a depth map into a list of 3D points

    Parameters:
        depth: Input depth map to project.
        cam: Camera parameters for input depth map.

    Returns:
        A float Tensor of 3D points corresponding to the projected depth values.
    """
    if (depth.shape[1] == 1):
        depth = depth.squeeze(1)

    batch_size, height, width = depth.shape
    cam_shape = cam.shape

    # get camera extrinsics and intrinsics
    P = cam[:,0,:,:]
    K = cam[:,1,:,:]
    K[:,3,:] = torch.tensor([0,0,0,1])

    # construct back-projection from invers matrices
    # separate into rotation and translation components
    bwd_projection = torch.matmul(torch.inverse(P), torch.inverse(K)).to(torch.float32)
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

    #TODO: make sure index select is differentiable
    #       (there is a backward function but need to find the code..)
    if (mask != None):
        world_coords = torch.index_select(world_coords, dim=2, index=non_zero_inds)
        world_coords = torch.movedim(world_coords, 1, 2)

    # reshape 3D coordinates back into 2D map: [B, H, W, 3]
    #   coords_map = world_coords.reshape(batch_size, height, width, 3)

    return world_coords

def project_renderer(renderer: o3d.visualization.rendering.OffscreenRenderer, K: np.ndarray, P: np.ndarray, width: float, height: float) -> np.ndarray:
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
    intrins = o3d.camera.PinholeCameraIntrinsic(width, height, K[0,0], K[1,1], K[0,2], K[1,2])
    renderer.setup_camera(intrins, P)

    # render image
    image = np.asarray(renderer.render_to_image())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def render_custom_values(points: np.ndarray, values: np.ndarray, image_shape: Tuple[int,int], cam: np.ndarray) -> np.ndarray:
    """Renders a point cloud into a 2D camera plane using custom values for each pixel.

    Parameters:
        points: List of 3D points to be rendered.
        values: List of values to be written in the rendered image.
        image_shape: Desired shape (height,width) of the rendered image.
        cam: Camera parameters for the image viewpoint.

    Returns:
        The rendered image for the list of points using the sepcified corresponding values.
    """
    points = points.tolist()
    values = list(values.astype(float))
    cam = cam.flatten().tolist()

    rendered_img = rd.render(list(image_shape), points, values, cam)

    return rendered_img

def render_into_ref(depths: np.ndarray, confs: np.ndarray, cams: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Renders all target depth and confidence maps into the reference view (assumed to be at index 0).

    Parameters:
        depths:
        confs:
        cams:

    Returns:
        rendered_depths:
        rendered_confs:
    """
    shape = depths.shape
    views = shape[0]
    rcam = cams[0].flatten().tolist()

    rendered_depths = [depths[0]]
    rendered_confs = [confs[0]]

    for v in range(1,views):
        tcam = cams[v].flatten().tolist()
        depth_map = depths[v].flatten().tolist()
        conf_map = confs[v].flatten().tolist()

        rendered_map = np.array(rd.render_to_ref(list([shape[1],shape[2]]),depth_map,conf_map,rcam,tcam))
        rendered_depth = rendered_map[:(shape[1]*shape[2])].reshape((shape[1],shape[2]))
        rendered_conf = rendered_map[(shape[1]*shape[2]):].reshape((shape[1],shape[2]))

        rendered_depths.append(rendered_depth)
        rendered_confs.append(rendered_conf)

    rendered_depths = np.asarray(rendered_depths)
    rendered_confs = np.asarray(rendered_confs)

    return rendered_depths, rendered_confs

def render_point_cloud(cloud: o3d.geometry.PointCloud, cam: np.ndarray, width: int, height: int) -> np.ndarray:
    """Renders a point cloud into a 2D image plane.

    Parameters:
        cloud: Point cloud to be rendered.
        cam: Camera parameters for the image plane.
        width: Desired width of the rendered image.
        height: Desired height of the rendered image.

    Returns:
        The rendered image for the point cloud at the specified camera viewpoint.
    """
    #   cmap = plt.get_cmap("hot_r")
    #   colors = cmap(dists)[:, :3]
    #   ply.colors = o3d.utility.Vector3dVector(colors)

    # set up the renderer
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    render.scene.add_geometry("cloud", cloud, mat)
    render.scene.set_background(np.asarray([0,0,0,1])) #r,g,b,a
    intrins = o3d.camera.PinholeCameraIntrinsic(width, height, cam[1,0,0], cam[1,1,1], cam[1,0,2], cam[1,1,2])
    render.setup_camera(intrins, cam[0])

    # render image
    image = np.asarray(render.render_to_image())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def reproject(src_depth: np.ndarray, src_cam: np.ndarray, tgt_depth: np.ndarray, tgt_cam: np.ndarray) -> Tuple[ np.ndarray,
                                                                                                                np.ndarray,
                                                                                                                np.ndarray]:
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
        src_cam: Camera parameters for the source depth map viewpoint.
        tgt_depth: Target depth map used for re-projection.
        tgt_cam: Camera parameters for the target depth map viewpoint.

    Returns:
        depth_reprojected: The re-projected depth values for the source depth map.
        coords_reprojected: The re-projection coordinates for the source view.
        coords_tgt: The projected coordinates for the target view.
    """
    height, width = src_depth.shape

    # back-project ref depths to 3D
    x_src, y_src = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_src, y_src = x_src.reshape([-1]), y_src.reshape([-1])
    xyz_src = np.matmul(np.linalg.inv(src_cam[1,:3,:3]),
                        np.vstack((x_src, y_src, np.ones_like(x_src))) * src_depth.reshape([-1]))

    # transform 3D points from ref to src coords
    xyz_tgt = np.matmul(np.matmul(tgt_cam[0], np.linalg.inv(src_cam[0])),
                        np.vstack((xyz_src, np.ones_like(x_src))))[:3]

    # project src 3D points into pixel coords
    K_xyz_tgt = np.matmul(tgt_cam[1,:3,:3], xyz_tgt)
    xy_tgt = K_xyz_tgt[:2] / K_xyz_tgt[2:3]
    x_tgt = xy_tgt[0].reshape([height, width]).astype(np.float32)
    y_tgt = xy_tgt[1].reshape([height, width]).astype(np.float32)

    # sample the depth values from the src map at each pixel coord
    sampled_depth_tgt = cv2.remap(tgt_depth, x_tgt, y_tgt, interpolation=cv2.INTER_LINEAR)
    projection_map = np.where(sampled_depth_tgt==0,0,1) # 0 where pixel does not project into src image plane

    # back-project src depths to 3D
    xyz_tgt = np.matmul(np.linalg.inv(tgt_cam[1,:3,:3]),
                        np.vstack((xy_tgt, np.ones_like(x_src))) * sampled_depth_tgt.reshape([-1]))

    # transform 3D points from src to ref coords
    xyz_reprojected = np.matmul(np.matmul(src_cam[0], np.linalg.inv(tgt_cam[0])),
                                np.vstack((xyz_tgt, np.ones_like(x_src))))[:3]

    # extract reprojected depth values
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)

    # project ref 3D points into pixel coords
    K_xyz_reprojected = np.matmul(src_cam[1,:3,:3], xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / (K_xyz_reprojected[2:3] + 1e-7)
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    coords_reprojected = np.dstack((x_reprojected, y_reprojected))
    coords_tgt = np.dstack((x_tgt, y_tgt))

    return depth_reprojected, coords_reprojected, coords_tgt, projection_map

def visibility_mask(src_depth: np.ndarray, src_cam: np.ndarray, depth_files: List[str], cam_files: List[str], src_ind: int = -1, pixel_th: float = 0.1) -> np.ndarray:
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
    height, width = src_depth.shape
    vis_map = np.not_equal(src_depth, 0.0).astype(np.double)

    for i in range(len(depth_files)):
        if (i==src_ind):
            continue

        # get files
        sdf = depth_files[i]
        scf = cam_files[i]

        # load data
        tgt_depth = read_pfm(sdf)
        tgt_cam = read_single_cam_sfm(scf,'r')

        mask = geometric_consistency_mask(src_depth, src_cam, tgt_depth, tgt_cam, pixel_th)
        vis_map += mask

    return vis_map.astype(np.float32)

def warp_to_tgt(tgt_depth: torch.Tensor, tgt_conf: torch.Tensor, ref_cam: torch.Tensor, tgt_cam: torch.Tensor, depth_planes: torch.Tensor, depth_vol: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs a homography warping
    Parameters:
        tgt_depth: 
        tgt_conf: 
        ref_cam:
        tgt_cam:
        depth_planes:
        depth_vol:
    
    Returns:
        depth_diff:
        warped_conf:
    """
    batch_size, views, height, width = tgt_depth.shape
    # grab intrinsics and extrinsics from reference view
    P_ref = ref_cam[:,0,:,:]
    K_ref = ref_cam[:,1,:,:]
    K_ref[:,3,:] = torch.tensor([0,0,0,1])

    # get intrinsics and extrinsics from target view
    P_tgt = tgt_cam[:,0,:,:]
    K_tgt = tgt_cam[:,1,:,:]
    K_tgt[:,3,:] = torch.tensor([0,0,0,1])

    R_tgt = P_tgt[:,:3,:3]
    t_tgt = P_tgt[:,:3,3:4]
    C_tgt = torch.matmul(-R_tgt.transpose(1,2), t_tgt)
    z_tgt = R_tgt[:,2:3,:3].reshape(batch_size,1,1,1,1,3).repeat(1,depth_planes, height,width,1,1)
    
    with torch.no_grad():
        # shape camera center vector
        C_tgt = C_tgt.reshape(batch_size,1,1,1,3).repeat(1, depth_planes, height, width, 1)

        bwd_proj = torch.matmul(torch.inverse(P_ref), torch.inverse(K_ref)).to(torch.float32)
        fwd_proj = torch.matmul(K_tgt, P_tgt).to(torch.float32)

        bwd_rot = bwd_proj[:,:3,:3]
        bwd_trans = bwd_proj[:,:3,3:4]

        proj = torch.matmul(fwd_proj, bwd_proj)
        rot = proj[:,:3,:3]
        trans = proj[:,:3,3:4]

        y, x = torch.meshgrid([torch.arange(0, height,dtype=torch.float32,device=tgt_depth.device),
                                     torch.arange(0, width, dtype=torch.float32, device=tgt_depth.device)], indexing='ij')
        y, x = y.contiguous(), x.contiguous()
        y, x = y.reshape(height*width), x.reshape(height*width)
        homog = torch.stack((x,y,torch.ones_like(x)))
        homog = torch.unsqueeze(homog, 0).repeat(batch_size,1,1)

        # get world coords
        world_coords = torch.matmul(bwd_rot, homog)
        world_coords = world_coords.unsqueeze(2).repeat(1,1,depth_planes,1)
        depth_vol = depth_vol.reshape(batch_size,1,depth_planes,-1)
        world_coords = world_coords * depth_vol
        world_coords = world_coords + bwd_trans.reshape(batch_size,3,1,1)
        world_coords = torch.movedim(world_coords, 1, 3)
        world_coords = world_coords.reshape(batch_size, depth_planes, height, width,3)

        # get pixel projection
        rot_coords = torch.matmul(rot, homog)
        rot_coords = rot_coords.unsqueeze(2).repeat(1,1,depth_planes,1)
        proj_3d = rot_coords * depth_vol
        proj_3d = proj_3d + trans.reshape(batch_size,3,1,1)
        proj_2d = proj_3d[:,:2,:,:] / proj_3d[:,2:3,:,:]

        proj_x = proj_2d[:,0,:,:] / ((width-1)/2) - 1
        proj_y = proj_2d[:,1,:,:] / ((height-1)/2) - 1
        proj_2d = torch.stack((proj_x, proj_y), dim=3)
        grid = proj_2d


    proj_depth = torch.sub(world_coords, C_tgt).unsqueeze(-1)
    proj_depth = torch.matmul(z_tgt, proj_depth).reshape(batch_size,depth_planes,height,width)

    warped_depth = F.grid_sample(tgt_depth, grid.reshape(batch_size, depth_planes*height, width, 2), mode='bilinear', padding_mode="zeros", align_corners=False)
    warped_depth = warped_depth.reshape(batch_size, depth_planes, height, width)

    warped_conf = F.grid_sample(tgt_conf, grid.reshape(batch_size, depth_planes*height, width, 2), mode='bilinear', padding_mode="zeros", align_corners=False)
    warped_conf = warped_conf.reshape(batch_size, depth_planes, height, width)

    depth_diff = torch.sub(proj_depth, warped_depth)

    return depth_diff, warped_conf
