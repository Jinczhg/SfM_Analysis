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
        pcd_view.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))
        points = np.asarray(pcd_view.points)
        normals = np.asarray(pcd_view.normals)
        angels = np.zeros(len(normals))
        for i in range(len(normals)):
            if part == "road":
                angels[i] = abs(normals[i, 1]) / (
                    np.sqrt(normals[i, 0] * normals[i, 0] + normals[i, 2] * normals[i, 2] + np.finfo(float).eps))  # (0, 1, 0) is the road
            # elif part == "left_view":
            #     angels[i] = -normals[i, 1] / (np.sqrt(normals[i, 0] * normals[i, 0] + normals[i, 2] * normals[i, 2] + np.finfo(float).eps))
            # else:   # right_view
            #     angels[i] = -normals[i, 1] / (np.sqrt(normals[i, 0] * normals[i, 0] + normals[i, 2] * normals[i, 2] + np.finfo(float).eps))
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
    IR_COR_to_RGB = 0
    HEATING_OPT = 1
    if PATH == 0:
        if IR_COR_to_RGB == 0 and HEATING_OPT == 0:
            # straight
            pcd_path = ['./straight/new_data/pcl_ir.pcd', './straight/new_data/pcl_rgb.pcd']
            traj_path = ["./straight/new_data/result_ir.txt", "./straight/new_data/result_rgb.txt"]
            # ir_no_correction to RGB
            transform = np.array([[0.99952298, 0.02137483, 0.02229172, -0.00054438], [-0.02131991, 0.99976906, -0.00269836, -0.00581446],
                                  [-0.02234425, 0.00222182, 0.99974787, 0.10498027]])
        else:
            if IR_COR_to_RGB == 1 and HEATING_OPT == 0:
                # ONLY CORRECT THE PIXEL VALUES BASED ON THE COOLING MODEL WHEN LOADING THE IMAGES
                pcd_path = ['./straight/new_data/pcl_ir_cor.pcd', './straight/new_data/pcl_rgb.pcd']
                traj_path = ["./straight/new_data/result_ir_cor.txt", "./straight/new_data/result_rgb.txt"]
                transform = np.array(
                    [[0.99974812, -0.00695419, 0.02133866, 0.00219466], [0.00698818, 0.99997443, -0.00151916, -0.00095347],
                     [-0.02132755, 0.00166789, 0.99977115, -0.05974309]])
            elif IR_COR_to_RGB == 0 and HEATING_OPT == 1:
                # ONLY OPTIMIZING THE HEATING CONSTANT (exp(-exposure/4.5))
                pcd_path = ['./straight/new_data/pcl_ir_opt.pcd', './straight/new_data/pcl_rgb.pcd']
                traj_path = ["./straight/new_data/result_ir_opt.txt", "./straight/new_data/result_rgb.txt"]
                transform = np.array(
                    [[0.9996208,   0.01628359,  0.02220589, -0.00157262], [-0.01622359,  0.99986424, - 0.00287948, - 0.00561096],
                     [-0.02224976, 0.00251813, 0.99974927, 0.12812562]])

            else:
                # IR_CORRECTION AND OPTIMIZATION
                pcd_path = ['./straight/new_data/pcl_ir_cor_opt.pcd', './straight/new_data/pcl_rgb.pcd']
                traj_path = ["./straight/new_data/result_ir_cor_opt.txt", './straight/new_data/result_rgb.txt']
                transform = np.array(
                    [[0.99975597, - 0.0053927, 0.0214225, -0.0006128], [0.00544018, 0.99998287, - 0.00215851, - 0.00437901],
                     [-0.0214105, 0.00227453, 0.99976818, 0.08771676]])

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
        print("(%d): %d points in the data" % (i, len(point_cloud)))
        ir_rgb_trajectory = file_interface.read_tum_trajectory_file(traj_path[i])
        trajectory_pc = ir_rgb_trajectory.positions_xyz
        pc_list.append(point_cloud)
        traj_list.append(trajectory_pc)
        sfm_a_list.append(sfm_a)
        if len(pc_list) == 2:
            pc_list[0], pc_list[1], traj_list[0], traj_list[1] = align_point_clouds(pc_list[0], pc_list[1], traj_list[0], traj_list[1],
                                                                                    transform, plot=False)

    # trim the traj_list and sample the point cloud
    traj_list[0] = traj_list[0][int(0.15 * len(traj_list[0])): int(0.85 * len(traj_list[0])), :]
    traj_list[1] = traj_list[1][int(0.15 * len(traj_list[1])): int(0.85 * len(traj_list[1])), :]

    # combine IR and RGB
    # ir_rgb_trajectory = np.vstack([traj_list[0], traj_list[1]])
    # ir_rgb_trajectory = ir_rgb_trajectory[ir_rgb_trajectory[:, 2].argsort()]  # descending order
    # ir_rgb_points = np.vstack([pc_list[0], pc_list[1]])

    # find a common "road plane" for both point clouds by calculating the cross product of trajectory vector and road vector
    # find sample points on the road to determine a vector across the road

    # If the road plane is defined by all the points (ir and rgb), then whichever has the dominated amount of the points will
    #  have more influence on the plane definition. As a result, when we use this plane to evaluate the points, the dominant one will have
    #  smaller errors. Ideally, the road plane should be defined independently. The alternative solution here is to sample same amount of
    #  points from both point clouds.
    left_road_points_lst = [[], []]
    right_road_points_lst = [[], []]
    for ii in range(len(pc_list)):
        for percent in np.arange(0, 0.95, 0.05):
            idx_1 = int(np.floor(percent * len(traj_list[ii])))
            idx_2 = int(np.ceil((percent + 0.05) * len(traj_list[ii])))
            left_road_points, right_road_points = road_points_from_traj(pc_list[ii], traj_list[ii], idx_1, idx_2)
            left_road_points_lst[ii].append(left_road_points)
            right_road_points_lst[ii].append(right_road_points)
        left_road_points_lst[ii] = np.vstack(left_road_points_lst[ii])
        right_road_points_lst[ii] = np.vstack(right_road_points_lst[ii])
        print("(%d): %d road points on the left, %d on the right" % (ii, len(left_road_points_lst[ii]), len(right_road_points_lst[ii])))

    np.random.seed(10)
    sample_amount = int(0.8 * min(len(left_road_points_lst[0]), len(left_road_points_lst[1])))
    random_idx_ir = np.random.uniform(0, len(left_road_points_lst[0]), sample_amount).astype(int)  # sample by uniform distribution
    random_idx_rgb = np.random.uniform(0, len(left_road_points_lst[1]), sample_amount).astype(int)
    left_road_points_lst[0] = left_road_points_lst[0][random_idx_ir]
    left_road_points_lst[1] = left_road_points_lst[1][random_idx_rgb]
    left_road_points = np.vstack(left_road_points_lst)
    print("On the left: %d sampled road points are from each point cloud" % sample_amount)

    sample_amount = int(0.8 * min(len(right_road_points_lst[0]), len(right_road_points_lst[1])))
    random_idx_ir = np.random.uniform(0, len(right_road_points_lst[0]), sample_amount).astype(int)  # sample by uniform distribution
    random_idx_rgb = np.random.uniform(0, len(right_road_points_lst[1]), sample_amount).astype(int)
    right_road_points_lst[0] = right_road_points_lst[0][random_idx_ir]
    right_road_points_lst[1] = right_road_points_lst[1][random_idx_rgb]
    right_road_points = np.vstack(right_road_points_lst)
    print("On the right: %d sampled road points are from each point cloud" % sample_amount)

    road_points = np.vstack([left_road_points, right_road_points])

    road_vec = np.mean(left_road_points, axis=0) - np.mean(right_road_points, axis=0)
    road_vec = road_vec / np.linalg.norm(road_vec)
    traj_vec = 0.5 * (((traj_list[0][-1, :] - traj_list[0][0, :]) / np.linalg.norm((traj_list[0][-1, :] - traj_list[0][0, :]))) +
                      ((traj_list[1][-1, :] - traj_list[1][0, :]) / np.linalg.norm((traj_list[1][-1, :] - traj_list[1][0, :]))))
    # unnecessary to calculate the dot product?
    proj_of_road_on_traj = (np.dot(road_vec, traj_vec) / np.dot(traj_vec, traj_vec)) * traj_vec
    Debugging = False
    if Debugging:
        origin = [0, 0, 0]
        X, Y, Z = zip(origin, origin, origin)
        U, V, W = zip(traj_vec, road_vec - proj_of_road_on_traj, road_vec)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.quiver(X, Y, Z, U, V, W, color=['r', 'g', 'b'])
        plt.show()
    # road_plane_normal = np.cross(traj_vec, road_vec - proj_of_road_on_traj)
    road_plane_normal = np.cross(traj_vec, [1, 0, 0])
    road_plane_normal = abs(road_plane_normal) / np.linalg.norm(road_plane_normal)  # unit vector. abs() makes sure that the normal points up.
    road_plane_d = -0.09  # this number needs to be tweaked slightly every time we change the random seed
    road_points_mean = np.mean(road_points, axis=0)
    # road_plane_d = -np.dot(road_points_mean, road_plane_normal)
    print("Road plane is: %f*x + %f*y + %f*z + %f = 0" % (road_plane_normal[0], road_plane_normal[1], road_plane_normal[2], road_plane_d))

    # Debugging: visualize and verify the sampled points on the road
    Debugging = True
    if Debugging:
        vis_pc(np.vstack([road_points, traj_list[0]]))
        plot_plane(np.vstack([road_points, traj_list[0]]), road_plane_normal, road_plane_d)

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

    # sample_amount = int(0.8 * min(len(pc_list[0]), len(pc_list[1])))
    # random_idx_ir = np.random.uniform(0, len(pc_list[0]), sample_amount).astype(int)  # sample by uniform distribution
    # random_idx_rgb = np.random.uniform(0, len(pc_list[1]), sample_amount).astype(int)
    # pc_list[0] = pc_list[0][random_idx_ir]
    # pc_list[1] = pc_list[1][random_idx_rgb]

    for j in range(len(pc_list)):
        data = pc_list[j]
        traj = traj_list[j]  # ir_rgb_trajectory

        # calculate the entropy
        # rotation = scipy.spatial.transform.Rotation.align_vectors(normal_tj, np.array([0,0,1]))
        hist = np.histogramdd(pc_list[j], bins=256)[0]
        hist /= hist.sum()
        hist = hist.flatten()
        hist = hist[hist.nonzero()]
        entropy = -0.5 * np.sum(hist * np.log2(hist))
        print("Entropy is = ", entropy)

        # Find the point cloud of the road
        Method = 2
        normal = np.zeros(3)
        d = 0
        data_road = []
        if Method == 0:
            # Method 0: from point normal
            dot_product = np.dot(np.asarray([0, 1, 0]), road_plane_normal)  # Y-up coordinate system
            normal_to_vert_angle = np.arccos(dot_product)
            tol = 5 / 180 * np.pi
            angle = np.pi / 2 - normal_to_vert_angle - tol
            data_road = sfm_a_list[j].extract_points_by_normals("road", angle, vis=False)
            data_road = data_road[data_road[:, 1] > np.mean(traj[:, 1])]  # road is higher in y-axis in current coordinate system
            # plt.plot(data_road[:, 1])
            pred = np.polyfit(range(len(data_road)), data_road[:, 1], 1)
            # plt.plot(range(len(data_road)), data_road[:, 1], 'o')  # create scatter plot
            # plt.plot(range(len(data_road)), pred[0]*range(len(data_road))+pred[1])  # add line of best fit
            dist = np.zeros(len(data_road))
            for t in range(len(data_road)):
                dist[t] = abs(data_road[t, 1] - (pred[0] * t + pred[1]))
            std_dist = np.std(dist)
            inliers = dist < 3 * std_dist
            data_road = data_road[inliers]
            # vis_pc(data_road)
            c, normal = fit_plane_LTSQ(data_road)
            point = np.array([0.0, 0.0, c])
            d = -point.dot(normal)
            # plot_plane(data_road, normal, d)
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

            for percent in np.arange(0, 0.95, 0.05):
                idx_1 = int(np.floor(percent * len(traj)))
                idx_2 = int(np.ceil((percent + 0.05) * len(traj)))
                condition_0 = (data[:, 2] > traj[idx_1, 2]) & (data[:, 2] < traj[idx_2, 2])
                condition_1 = (data[:, 1] - np.mean(traj[idx_1:idx_2, 1]) > 0.05) & (
                        data[:, 1] - np.mean(traj[idx_1:idx_2, 1]) < 0.15)  # road is higher in y-axis in current coordinate system
                condition_3 = (np.min(traj[idx_1:idx_2, 0]) - data[:, 0] < 0.2) | (
                        data[:, 0] - np.max(traj[idx_1:idx_2, 0]) < 0.1)  # driving on the right of the road. Road points are closer on the right.
                filtered_road_points = data[condition_0 & condition_1 & condition_3]
                data_road.append(filtered_road_points)

            data_road = np.vstack(data_road)
            data_road = data_road[data_road[:, 2].argsort()]

            if Method == 1:
                # Method 1: fit a plane to road points
                c, normal = fit_plane_LTSQ(data_road)
                point = np.array([0.0, 0.0, c])
                d = -point.dot(normal)
            elif Method == 2:
                # Method 2: use the common road plane of the two point clouds
                # Method 2.1: manually define the road plane
                # y_up = np.asarray([0, 1, 0])  # Y-up coordinate system
                # proj_of_y_on_traj = (np.dot(y_up, traj_vec)/np.dot(traj_vec, traj_vec))*traj_vec
                # normal = y_up - proj_of_y_on_traj
                # normal = abs(normal) / np.linalg.norm(normal)
                # Method 2.2: define the road plane from the equally sampled points to avoid bias from the amount-dominated point cloud
                normal = road_plane_normal
                d = road_plane_d
            elif Method == 3:  # TODO: Trajectory fitting results is very uncertain here. Method 1 is not currently incorrect. Need to fix.
                # Method 3: from the trajectory plane
                c, normal = fit_plane_LTSQ(traj)
                point = np.array([0.0, 0.0, c])
                d_tj = -point.dot(normal)
                d = d_tj + 0.1  # road plane that is parallel to the trajectory plane
                Debugging = True
                if Debugging:
                    plot_plane(data_road, normal, d)
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
            else:
                exit("Program stopped due to an invalid Method Number.")

        # vis_pc(data_road)
        plot_plane(data_road, normal, d)
        print("%d points are being used for analysis" % len(data_road))
        error_analysis(data_road, normal, d)
