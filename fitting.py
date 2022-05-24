import math

import numpy as np
import matplotlib.pyplot as plt
from evo.tools import file_interface
import scipy.stats as stats
import scipy

from utils import *


class SfM_Analysis:
    def __init__(self):
        self.points = None
        self.pcd_view = None

    def read_point_cloud(self, path):
        pcd = o3d.io.read_point_cloud(path)
        out_arr = np.asarray(pcd.points)
        out_arr = out_arr[(out_arr[:, 0] > -5) & (out_arr[:, 0] < 10)]  # [-192.548, 254.316]
        out_arr = out_arr[(out_arr[:, 1] > -3) & (out_arr[:, 1] < 0.25)]  # [-248.162, 16.9245]
        out_arr = out_arr[(out_arr[:, 2] > 0) & (out_arr[:, 2] < 30)]  # [-104.918, 2406.66]
        # print("output array from input list: ", out_arr)
        point_cloud = np.vstack((out_arr[:, 0], out_arr[:, 1], out_arr[:, 2])).transpose()
        pcd_view = o3d.geometry.PointCloud()
        pcd_view.points = o3d.utility.Vector3dVector(point_cloud)
        # pcd_view = pcd_view.voxel_down_sample(voxel_size=0.1)
        self.points = point_cloud
        self.pcd_view = pcd_view

    def visualize_point_cloud(self, show_normals):
        # o3d.visualization.draw_geometries([self.pcd_view])
        viewer = o3d.visualization.Visualizer()
        viewer.create_window(width=640, height=480)
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
        viewer.create_window(width=640, height=480)
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

    def extract_points_by_normals(self, part="road", angel=np.pi / 2, vis=False):  # angel in degree
        pcd_view = self.pcd_view
        pcd_view.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=20))
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
        vert_idx = np.where(angels > math.tan(angel))
        vert_points = points[vert_idx]

        sorted_vert_points = vert_points[vert_points[:, 2].argsort()[::-1]]  # sort by z-value
        # sorted_vert_points = sorted_vert_points[0:int(np.floor(0.8 * len(sorted_vert_points))), :]
        # view part
        if vis:
            part_view = o3d.geometry.PointCloud()
            part_view.points = o3d.utility.Vector3dVector(sorted_vert_points)
            part_viewer = o3d.visualization.Visualizer()
            part_viewer.create_window(width=640, height=480)
            part_viewer.add_geometry(part_view)
            opt = part_viewer.get_render_option()
            opt.show_coordinate_frame = True
            part_viewer.run()
        return sorted_vert_points


if __name__ == "__main__":
    PATH = 0
    if PATH == 0:
        # straight
        pcd_path = ['./straight/pcl_ir.pcd', './straight/pcl_rgb.pcd']
        traj_path = ["./straight/result_ir.txt", "./straight/result_rgb.txt"]
        transform = np.array([[0.99954171, -0.02426103, 0.01810422, 0.01280828], [0.02432169, 0.99969926, -0.00313785, 0.01528642],
                              [-0.01802265, 0.00357674, 0.99983118, -0.45082403]])
    else:
        # first turn
        pcd_path = ['./first_turn/pcl_ir.pcd', './first_turn/pcl_rgb.pcd']
        traj_path = ["./first_turn/result_ir.txt", "./first_turn/result_rgb.txt"]
        transform = np.array([[0.99932336, -0.02171992, 0.02968283, -0.07670261], [0.02179248, 0.99976026, -0.00212329, 0.00109111],
                              [-0.02962959, 0.00276871, 0.99955711, -0.12763116]])

    pc_list = []
    traj_list = []
    sfm_a_list = []
    for i in range(len(pcd_path)):
        sfm_a = SfM_Analysis()
        sfm_a.read_point_cloud(pcd_path[i])
        # sfm_a.visualize_point_cloud(show_normals=False)
        point_cloud = sfm_a.points
        trajectory = file_interface.read_tum_trajectory_file(traj_path[i])
        trajectory_pc = trajectory.positions_xyz
        pc_list.append(point_cloud)
        traj_list.append(trajectory_pc)
        sfm_a_list.append(sfm_a)
        if len(pc_list) == 2:
            pc_list[0], pc_list[1], traj_list[0], traj_list[1] = align_point_clouds(pc_list[0], pc_list[1], traj_list[0], traj_list[1],
                                                                                    transform, plot=False)

    # trim the traj_list
    trajectory = np.vstack([traj_list[0], traj_list[1]])
    trajectory = trajectory[trajectory[:, 2].argsort()]  # descending order
    trajectory = trajectory[int(0.15 * len(trajectory)): int(0.85 * len(trajectory)), :]  # only using part of the trajectory

    # find a common "road plane" for both point clouds by calculating the cross product of trajectory vector and road vector
    traj_vec = trajectory[-1, :] - trajectory[0, :]  # only using part of the trajectory

    # find sample points on the road to determine a vector across the road
    ir_rgb_points = np.vstack([pc_list[0], pc_list[1]])
    # beginning of the trajectory
    begin_1 = int(np.floor(0.1 * len(trajectory)))
    begin_2 = int(np.ceil(0.2 * len(trajectory)))
    left_road_points_begin, right_road_points_begin = road_points_from_traj(ir_rgb_points, trajectory, begin_1, begin_2)
    # middle of the trajectory
    mid_1 = int(np.floor(0.5 * len(trajectory)))
    mid_2 = int(np.ceil(0.6 * len(trajectory)))
    left_road_points_mid, right_road_points_mid = road_points_from_traj(ir_rgb_points, trajectory, mid_1, mid_2)
    # end of the trajectory
    end_1 = int(np.floor(0.9 * len(trajectory)))
    end_2 = len(trajectory) - 1
    left_road_points_end, right_road_points_end = road_points_from_traj(ir_rgb_points, trajectory, end_1, end_2)

    road_points = np.vstack(
        [left_road_points_begin, right_road_points_begin, left_road_points_mid, right_road_points_mid, left_road_points_end,
         right_road_points_end])

    road_vec = np.mean(
        np.vstack([right_road_points_begin, right_road_points_mid, right_road_points_end]), axis=0) - np.mean(
        np.vstack([left_road_points_begin, left_road_points_mid, left_road_points_end]), axis=0)
    # unnecessary to calculate the dot product?
    traj_vec_norm = np.sqrt(sum(traj_vec * traj_vec))
    proj_of_road_on_traj = (np.dot(road_vec, traj_vec) / traj_vec_norm * traj_vec_norm) * traj_vec
    road_plane_normal = np.cross(traj_vec, road_vec - proj_of_road_on_traj)
    road_plane_normal = road_plane_normal / np.linalg.norm(road_plane_normal)  # unit vector
    road_plane_d = -0.35

    # Debugging: visualize and verify the sampled points on the road
    Debugging = False
    if Debugging:
        vis_pc(np.vstack([road_points, trajectory]))
        plot_fitted_plane(np.vstack([road_points, trajectory]), road_plane_normal, road_plane_d)

        # point-to-plane distance
        dist = np.zeros(len(road_points))
        for t in range(len(road_points)):
            x = road_points[t, 0]
            y = road_points[t, 1]
            z = road_points[t, 2]
            dist[t] = signed_shortest_distance(x, y, z, road_plane_normal, road_plane_d)
        fig = plt.figure()
        _, x, _ = plt.hist(dist, bins='auto', density=True)  # arguments are passed to np.histogram
        plt.title("Total error is " + str(sum(abs(dist))))

    for j in range(len(pc_list)):
        data = pc_list[j]
        traj = trajectory  # traj_list[j]

        # calculate the entropy
        # rotation = scipy.spatial.transform.Rotation.align_vectors(normal_tj, np.array([0,0,1]))
        hist = np.histogramdd(pc_list[j], bins=256)[0]
        hist /= hist.sum()
        hist = hist.flatten()
        hist = hist[hist.nonzero()]
        entropy = -0.5 * np.sum(hist * np.log2(hist))
        print("Entropy is = ", entropy)

        # Find the point cloud of the road
        Method = 0
        normal = np.zeros(3)
        d = 0
        data_road = []
        if Method == 0:
            # Method 0: from point normal
            dot_product = np.dot(np.asarray([0, 1, 0]), road_plane_normal)
            normal_to_vert_angle = np.arccos(dot_product)
            angle = np.pi / 2 - normal_to_vert_angle
            data_road = sfm_a_list[j].extract_points_by_normals("road", angle, vis=False)
            data_road = data_road[data_road[:, 1] > -1]
            # plt.plot(data_road[:, 1])
            pred = np.polyfit(range(len(data_road)), data_road[:, 1], 1)
            # plt.plot(range(len(data_road)), data_road[:, 1], 'o')  # create scatter plot
            # plt.plot(range(len(data_road)), pred[0]*range(len(data_road))+pred[1])  # add line of best fit
            dist = np.zeros(len(data_road))
            for t in range(len(data_road)):
                dist[t] = abs(data_road[t, 1] - (pred[0]*t+pred[1]))
            std_dist = np.std(dist)
            inliers = dist < 3*std_dist
            data_road = data_road[inliers]
            vis_pc(data_road)
            c, normal = fit_plane_LTSQ(data_road)
            point = np.array([0.0, 0.0, c])
            d = -point.dot(normal)
            plot_fitted_plane(data_road, normal, d)
        else:
            # Find road points respectively from the straight road and the turn based on the distance to the trajectory
            # max_street_depth = max(traj[:, 2])
            # traj_pt1 = traj[traj[:, 2] < max_street_depth - 3]  # straight
            # traj_pt2 = traj[traj[:, 2] >= max_street_depth - 3]  # first turn
            #
            # # filter the road points
            # data_pt1 = data[(data[:, 0] < 0) & (data[:, 2] < max_street_depth - 3)]  # straight
            # data_pt2 = data[data[:, 0] >= 0]  # first turn
            # data_pt1 = data_pt1[(data_pt1[:, 0] > min(traj_pt1[:, 0]) - 0.1) & (data_pt1[:, 0] < max(traj_pt1[:, 0]) + 0.1)]  # straight
            # data_pt2 = data_pt2[(data_pt2[:, 2] > min(traj_pt2[:, 2]) - 0.5) & (data_pt2[:, 2] < max(traj_pt2[:, 2]) + 0.5)]  # first turn
            #
            # # data = np.vstack([data_pt1, data_pt2])  # full points
            # data = data_pt1  # only the straight road
            # traj = traj_pt1

            for percent in np.arange(0, 0.9, 0.1):
                idx_1 = int(np.floor(percent * len(trajectory)))
                idx_2 = int(np.ceil((percent + 0.1) * len(trajectory)))
                condition_0 = (data[:, 2] > trajectory[idx_1, 2]) & (data[:, 2] < trajectory[idx_2, 2])
                condition_1 = (data[:, 1] - np.median(trajectory[idx_1:idx_2, 1]) > 0) & (
                        data[:, 1] - np.median(trajectory[idx_1:idx_2 - 1, 1]) < 0.1)
                condition_3 = (np.min(trajectory[idx_1:idx_2, 0]) - data[:, 0] < 0.1) & (
                        data[:, 0] - np.max(trajectory[idx_1:idx_2, 0]) < 0.1)
                filtered_road_points = data[condition_0 & condition_1 & condition_3]
                data_road.append(filtered_road_points)

            data_road = np.vstack(data_road)
            data_road = data_road[data_road[:, 2].argsort()]

            if Method == 1:
                # Method 1: from the trajectory plane
                c, normal = fit_plane_LTSQ(traj)
                point = np.array([0.0, 0.0, c])
                d_tj = -point.dot(normal)
                d = d_tj - 0.3  # road plane that is parallel to the trajectory plane
                Debugging = True
                if Debugging:
                    plot_fitted_plane(data_road, normal, d)
                    # point-to-plane distance
                    dist = np.zeros(len(data_road))
                    for t in range(len(data_road)):
                        x = data_road[t, 0]
                        y = data_road[t, 1]
                        z = data_road[t, 2]
                        dist[t] = signed_shortest_distance(x, y, z, normal, d)
                    fig = plt.figure()
                    _, x, _ = plt.hist(dist, bins='auto', density=True)  # arguments are passed to np.histogram
                    plt.title("Total error is " + str(sum(abs(dist))))
            elif Method == 2:
                # Method 2: use the common road plane of the two point clouds
                normal = road_plane_normal
                d = road_plane_d
            else:
                exit("Program stopped due to an invalid Method Number.")

        vis_pc(data_road)
        print("Total number of points is", len(data_road))
        error_analysis(data_road, normal, d)
