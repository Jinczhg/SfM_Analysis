import math

import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt


class SfM_Analysis:
    def __init__(self, path):
        self.path = path
        self.points = None
        self.pcd_view = None

    def read_point_cloud(self):
        pcd = o3d.io.read_point_cloud(self.path)
        out_arr = np.asarray(pcd.points)
        out_arr = out_arr[(out_arr[:, 0] > -2) & (out_arr[:, 0] < 10)]  # [-192.548, 254.316]
        out_arr = out_arr[abs(out_arr[:, 1]) < 1]  # [-248.162, 16.9245]
        out_arr = out_arr[(out_arr[:, 2] > 0) & (out_arr[:, 2] < 25)]  # [-104.918, 2406.66]
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
        sorted_vert_points = sorted_vert_points[0:int(np.floor(0.8*len(sorted_vert_points))), :]
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


def fitPlaneLTSQ(XYZ):
    (rows, cols) = XYZ.shape
    G = np.ones((rows, 3))
    G[:, 0] = XYZ[:, 0]  # X
    G[:, 1] = XYZ[:, 1]  # Y
    Z = XYZ[:, 2]
    (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)
    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return c, normal


def shortest_distance(x, y, z, a, b, c, d):
    d = abs((a * x + b * y + c * z + d))
    e = (np.sqrt(a * a + b * b + c * c))
    return d / e


path = ['./pcl_data_ir.pcd', './pcl_data_rgb.pcd']
for i in range(2):
    sfma = SfM_Analysis(path[i])
    sfma.read_point_cloud()
    sfma.visualize_point_cloud(show_normals=False)
    # road
    data = sfma.extract_points_by_normals("road", 85)
    print("Total number of points is", len(data))
    c, normal = fitPlaneLTSQ(data)

    # plot fitted plane
    maxx = np.max(data[:, 0])
    maxy = np.max(data[:, 1])
    minx = np.min(data[:, 0])
    miny = np.min(data[:, 1])

    point = np.array([0.0, 0.0, c])
    d = -point.dot(normal)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # plot original points
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=5)

    # compute needed points for plane plotting
    xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    dist = np.zeros(len(data))
    for j in range(len(data)):
        x = data[j, 0]
        y = data[j, 1]
        z = data[j, 2]
        dist[j] = shortest_distance(x, y, z, normal[0], normal[1], normal[2], d)
    avg_dist = np.average(dist)
    print("Fitting Error Mean is", avg_dist)
    std_dist = np.std(dist)
    print("Fitting Error Std is", std_dist)

    # plot plane
    ax.plot_surface(xx, yy, zz, alpha=0.2)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    plt.plot(dist)
    plt.show()
