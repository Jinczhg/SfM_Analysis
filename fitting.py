import math

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from evo.tools import file_interface
import scipy.stats as stats

class SfM_Analysis:
    def __init__(self):
        self.points = None
        self.pcd_view = None

    def read_point_cloud(self, path):
        pcd = o3d.io.read_point_cloud(path)
        out_arr = np.asarray(pcd.points)
        out_arr = out_arr[(out_arr[:, 0] > -5) & (out_arr[:, 0] < 10)]  # [-192.548, 254.316]
        out_arr = out_arr[abs(out_arr[:, 1]) < 2]  # [-248.162, 16.9245]
        out_arr = out_arr[(out_arr[:, 2] > 0) & (out_arr[:, 2] < 30)]  # [-104.918, 2406.66]
        # print("output array from input list: ", out_arr)
        point_cloud = np.vstack((out_arr[:, 0], out_arr[:, 1], out_arr[:, 2])).transpose()
        pcd_view = o3d.geometry.PointCloud()
        pcd_view.points = o3d.utility.Vector3dVector(point_cloud)
        # pcd_view = pcd_view.voxel_down_sample(voxel_size=0.1)
        pcd_view.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=20))
        self.points = point_cloud
        self.pcd_view = pcd_view

    def visualize_point_cloud(self, show_normals):
        # o3d.visualization.draw_geometries([self.pcd_view])
        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        viewer.add_geometry(self.pcd_view)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        opt.point_show_normal = show_normals
        viewer.run()

    def point_clustering(self):
        pcd_view = self.pcd_view
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                pcd_view.cluster_dbscan(eps=0.15, min_points=10, print_progress=True))
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd_view.colors = o3d.utility.Vector3dVector(colors[:, :3])
        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        viewer.add_geometry(pcd_view)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        viewer.run()

    def plane_segmentation(self):
        pcd_view = self.pcd_view
        plane_model, inliers = pcd_view.segment_plane(distance_threshold=0.01,
                                                      ransac_n=3,
                                                      num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        inlier_cloud = pcd_view.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd_view.select_by_index(inliers, invert=True)
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                          zoom=0.8,
                                          front=[-0.4999, -0.1659, -0.8499],
                                          lookat=[2.1813, 2.0619, 2.0999],
                                          up=[0.1204, -0.9852, 0.1215])

    def extract_points_by_normals(self, part="road", angel=85):  # angel in degree
        pcd_view = self.pcd_view
        points = np.asarray(pcd_view.points)
        normals = np.asarray(pcd_view.normals)
        angels = np.zeros(len(normals))
        for i in range(len(normals)):
            if part == "road":
                angels[i] = abs(normals[i, 1]) / (
                    np.sqrt(normals[i, 0] ** 2 + normals[i, 2] ** 2 + np.finfo(float).eps))  # (0, -1, 0) is the road
            # elif part == "left_view":
            #     angels[i] = -normals[i, 1] / (np.sqrt(normals[i, 0] ** 2 + normals[i, 2] ** 2 + np.finfo(float).eps))
            # else:   # right_view
            #     angels[i] = -normals[i, 1] / (np.sqrt(normals[i, 0] ** 2 + normals[i, 2] ** 2 + np.finfo(float).eps))
        vert_idx = np.where(angels > math.tan(angel * math.pi / 180))
        vert_points = points[vert_idx]

        # sorted_vert_points = vert_points
        sorted_vert_points = vert_points[vert_points[:, 1].argsort()[::-1]]
        sorted_vert_points = sorted_vert_points[0:int(np.floor(0.8 * len(sorted_vert_points))), :]
        # view part
        part_view = o3d.geometry.PointCloud()
        part_view.points = o3d.utility.Vector3dVector(sorted_vert_points)
        part_viewer = o3d.visualization.Visualizer()
        part_viewer.create_window()
        part_viewer.add_geometry(part_view)
        opt = part_viewer.get_render_option()
        opt.show_coordinate_frame = True
        part_viewer.run()
        return sorted_vert_points


def align_point_clouds(point_cloud_1, point_cloud_2, traj_1, traj_2, transform_12, vis=False):
    pc_1 = np.vstack((point_cloud_1[:, 0], point_cloud_1[:, 1], point_cloud_1[:, 2], np.ones(len(point_cloud_1))))
    point_cloud_1 = np.matmul(transform_12, pc_1).T
    traj_1 = np.vstack((traj_1[:, 0], traj_1[:, 1], traj_1[:, 2], np.ones(len(traj_1))))
    traj_1 = np.matmul(transform_12, traj_1).T
    if vis:
        pcd_view_1 = o3d.geometry.PointCloud()
        pcd_view_1.points = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector(point_cloud_1))
        pcd_view_1.paint_uniform_color([1.0, 0, 0])
        pcd_view_2 = o3d.geometry.PointCloud()
        pcd_view_2.points = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector(point_cloud_2))
        pcd_view_2.paint_uniform_color([0, 1.0, 0])
        pcd_view_traj_1 = o3d.geometry.PointCloud()
        pcd_view_traj_1.points = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector(traj_1))
        pcd_view_traj_1.paint_uniform_color([0, 0, 1.0])
        pcd_view_traj_2 = o3d.geometry.PointCloud()
        pcd_view_traj_2.points = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector(traj_2))
        pcd_view_traj_2.paint_uniform_color([1.0, 0, 1.0])

        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        viewer.add_geometry(pcd_view_1)
        viewer.add_geometry(pcd_view_2)
        viewer.add_geometry(pcd_view_traj_1)
        viewer.add_geometry(pcd_view_traj_2)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        viewer.run()
    return point_cloud_1, point_cloud_2, traj_1, traj_2


def fit_plane_LTSQ(XYZ):
    (rows, cols) = XYZ.shape
    G = np.ones((rows, 3))
    G[:, 0] = XYZ[:, 0]  # X
    G[:, 1] = XYZ[:, 1]  # Y
    Z = XYZ[:, 2]
    (a, b, c), res, rank, s = np.linalg.lstsq(G, Z, rcond=None)
    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return c, normal


def shortest_distance(x, y, z, a, b, c, d):
    d = (a * x + b * y + c * z + d)
    e = (np.sqrt(a * a + b * b + c * c))
    return d / e


def vis_pc(point_cloud):
    pcd_view = o3d.geometry.PointCloud()
    pcd_view.points = o3d.utility.Vector3dVector(point_cloud)
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(pcd_view)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    viewer.run()


pcd_path = ['./first_turn/pcl_ir.pcd', './first_turn/pcl_rgb.pcd']
traj_path = ["./first_turn/result_ir.txt", "./first_turn/result_rgb.txt"]
pc_list = []
traj_list = []
# straight
# transform = np.array([[0.99954171, -0.02426103, 0.01810422, 0.01280828], [0.02432169, 0.99969926, -0.00313785, 0.01528642],
#                       [-0.01802265, 0.00357674, 0.99983118, -0.45082403]])
# first turn
transform = np.array([[0.99932336, -0.02171992, 0.02968283, -0.07670261], [0.02179248, 0.99976026, -0.00212329, 0.00109111],
                      [-0.02962959, 0.00276871, 0.99955711, -0.12763116]])

for i in range(len(pcd_path)):
    sfma = SfM_Analysis()
    sfma.read_point_cloud(pcd_path[i])
    # sfma.visualize_point_cloud(show_normals=False)
    point_cloud = sfma.points
    trajectory = file_interface.read_tum_trajectory_file(traj_path[i])
    trajectory_pc = trajectory.positions_xyz
    pc_list.append(point_cloud)
    traj_list.append(trajectory_pc)

    if len(pc_list) == 2:
        pc_list[0], pc_list[1], traj_list[0], traj_list[1] = align_point_clouds(pc_list[0], pc_list[1], traj_list[0], traj_list[1],
                                                                                transform)

road_viewer = o3d.visualization.Visualizer()
road_viewer.create_window()
for j in range(len(pc_list)):
    # the plane of the trajectory
    # z = (-normal[0] * x - normal[1] * y - d) * 1. / normal[2]
    c_tj, normal_tj = fit_plane_LTSQ(traj_list[j])
    point = np.array([0.0, 0.0, c_tj])
    d_tj = -point.dot(normal_tj)
    data = pc_list[j]
    traj = traj_list[j]

    # filter the "road" from the data
    # Method 1: by normal
    # data_road = sfma.extract_points_by_normals("road", 85)
    # Method 2: by trajectory
    max_street_depth = max(traj[:, 2])
    traj_pt1 = traj[traj[:, 2] < max_street_depth - 1]  # straight road
    traj_pt2 = traj[traj[:, 2] >= max_street_depth - 1]  # first turn
    left_pt1 = min(traj_pt1[:, 0]) - 0.1
    right_pt1 = max(traj_pt1[:, 0]) + 0.1
    left_pt2 = min(traj_pt2[:, 2]) - 0.5
    right_pt2 = max(traj_pt2[:, 2]) + 0.5

    data_pt1 = data[data[:, 0] < 0]  # straight road
    data_pt2 = data[data[:, 0] >= 0]  # first turn
    data_pt1 = data_pt1[(data_pt1[:, 0] > left_pt1) & (data_pt1[:, 0] < right_pt1)]
    data_pt2 = data_pt2[(data_pt2[:, 2] > left_pt2) & (data_pt2[:, 2] < right_pt2)]

    data = np.vstack([data_pt1, data_pt2])
    road_idx = []
    for k in range(len(data)):
        y = (-normal_tj[0] * data[k, 0] - normal_tj[2] * data[k, 2] - d_tj) * 1. / normal_tj[1]
        if data[k, 1] > y + 0.08 and (data[k, 1] < y + 0.5):
            road_idx.append(k)
    data_road = data[road_idx]

    # vis_pc(data_road)
    print("Total number of points is", len(data_road))

    c, normal = fit_plane_LTSQ(data_road)

    # plot fitted plane
    maxx = np.max(data_road[:, 0])
    maxy = np.max(data_road[:, 1])
    minx = np.min(data_road[:, 0])
    miny = np.min(data_road[:, 1])

    point = np.array([0.0, 0.0, c])
    d = -point.dot(normal)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # plot original points
    ax.scatter(data_road[:, 0], data_road[:, 1], data_road[:, 2], s=5)

    # compute needed points for plane plotting
    xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    dist = np.zeros(len(data_road))
    for t in range(len(data_road)):
        x = data_road[t, 0]
        y = data_road[t, 1]
        z = data_road[t, 2]
        dist[t] = shortest_distance(x, y, z, normal[0], normal[1], normal[2], d)
    avg_dist = np.average(abs(dist))
    print("Fitting Error Mean is", avg_dist)
    std_dist = np.std(abs(dist))
    print("Fitting Error Std is", std_dist)

    # plot plane
    ax.plot_surface(xx, yy, zz, alpha=0.2)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    # plt.plot(dist)
    # plt.show()
    _, x, _ = plt.hist(dist, bins='auto', density=True)  # arguments are passed to np.histogram
    density = stats.gaussian_kde(dist)
    plt.plot(x, density(x))
    plt.title("Histogram with 'auto' bins")
    plt.show()
