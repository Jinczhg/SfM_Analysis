import math

import numpy as np
import matplotlib.pyplot as plt
from evo.tools import file_interface
import scipy.stats as stats
import scipy
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d

from utils import *


class SfM_Analysis:
    def __init__(self):
        self.points = None
        self.pcd_view = None

    def read_point_cloud(self, path):
        pcd = o3d.io.read_point_cloud(path)
        out_arr = np.asarray(pcd.points)
        out_arr = out_arr[(out_arr[:, 0] > -5) & (out_arr[:, 0] < 5)]  # [-192.548, 254.316]
        out_arr = out_arr[(out_arr[:, 1] > -5) & (out_arr[:, 1] < 5)]  # [-248.162, 16.9245]
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
    pcd_path = ['./straight/dso_results/ir/pcl_42892.pcd', './straight/dso_results/ir_cor/pcl_65780.pcd',
                './straight/dso_results/rgb/pcl_55940.pcd']
    traj_path = ["./straight/dso_results/ir/traj_ir.txt", "./straight/dso_results/ir_cor/traj_ir_cor.txt",
                 "./straight/dso_results/rgb/traj_rgb.txt"]
    transform02 = np.array([[0.99980232, -0.01178885, 0.01601074, 0.00921175], [0.01205841, 0.99978539, -0.01684532, 0.03240032],
                            [-0.01580871, 0.01703505, 0.99972991, -0.48311585]])
    # transform02 = np.array([[0.99955841, -0.02495745,  0.01612775,  0.00745329], [0.02522304,  0.999546, -0.01647981, 0.03480672],
    #                         [-0.01570913,  0.01687932,  0.99973412, -0.50827898]])
    transform12 = np.array([[0.99968172, -0.02025884, 0.01503444, 0.012658], [0.02057199, 0.99956833, -0.02097511, 0.04522296],
                            [-0.01460302, 0.02127772, 0.99966695, -0.50948741]])
    # transform12 = np.array([[0.99962437, -0.02137063,  0.01715844, -0.00072078], [0.02178987,  0.99945913, -0.02463007, 0.05058136],
    #                         [-0.0166228,   0.0249947,   0.99954937, -0.35614861]])

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
        # if len(pc_list) == 2:
        #     pc_list[0], pc_list[1], traj_list[0], traj_list[1] = align_point_clouds(pc_list[0], pc_list[1], traj_list[0], traj_list[1],
        #                                                                             transform, plot=False)
        if len(pc_list) == 3:
            pc_list[0], pc_list[2], traj_list[0], traj_list[2] = align_point_clouds(pc_list[0], pc_list[2], traj_list[0], traj_list[2],
                                                                                    transform02, plot=False)
            pc_list[1], pc_list[2], traj_list[1], traj_list[2] = align_point_clouds(pc_list[1], pc_list[2], traj_list[1], traj_list[2],
                                                                                    transform12, plot=False)

    linepts = [[], [], []]
    for j in range(len(pcd_path)):
        # trim the traj_list and sample the point cloud
        traj_list[j] = traj_list[j][int(0.15 * len(traj_list[j])): int(0.85 * len(traj_list[j])), :]
        # fit a 3D line to the trajectory
        # Calculate the mean of the points, i.e. the 'center' of the cloud
        datamean = traj_list[j].mean(axis=0)
        uu, dd, vv = np.linalg.svd(traj_list[j] - datamean)
        # vv[0] contains the first principal component, i.e. the direction vector
        # Now generate some points along this best fit line, for plotting.
        linepts[j] = vv[0] * np.mgrid[-7:7:2j][:, np.newaxis]
        # shift by the mean to get the line in the right place
        linepts[j] += datamean

        dist = point_to_line_distance_3D(linepts[j][0,:], linepts[j][1,:], traj_list[j])
        MSE = np.square(dist).mean()
        RMSE = math.sqrt(MSE)
        print("Fitting RMSE is", RMSE)
        std_dist = np.std(abs(dist))
        print("Fitting Std is", std_dist)

    # 3D plot
    Plotting = False
    if Plotting:
        ax = m3d.Axes3D(plt.figure())
        ax.scatter3D(*traj_list[j].T)
        ax.plot3D(*linepts[0].T)
        ax.plot3D(*linepts[1].T)
        ax.plot3D(*linepts[2].T)
        plt.show()
        # 2D plot
        # fig = plt.figure()
        # ax = plt.axes()
        # linepts2d = [linepts[0][:, 0], linepts[0][:, 2]]
        # linepts2d = np.asarray(linepts2d)
        # ax.plot(linepts2d[0,:], linepts2d[1,:], c='red')
        # linepts2d = [linepts[1][:, 0], linepts[1][:, 2]]
        # linepts2d = np.asarray(linepts2d)
        # ax.plot(linepts2d[0,:], linepts2d[1,:], c='green')
        # linepts2d = [linepts[2][:, 0], linepts[2][:, 2]]
        # linepts2d = np.asarray(linepts2d)
        # ax.plot(linepts2d[0,:], linepts2d[1,:], c='blue')

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
    left_road_points_lst = [[], [], []]
    right_road_points_lst = [[], [], []]
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

    np.random.seed(100)

    sample_amount = int(0.8 * min(len(left_road_points_lst[0]), len(left_road_points_lst[1]), len(left_road_points_lst[2])))
    random_idx_ir = np.random.uniform(0, len(left_road_points_lst[0]), sample_amount).astype(int)  # sample by uniform distribution
    random_idx_ir_cor = np.random.uniform(0, len(left_road_points_lst[1]), sample_amount).astype(int)
    random_idx_rgb = np.random.uniform(0, len(left_road_points_lst[2]), sample_amount).astype(int)
    left_road_points_lst[0] = left_road_points_lst[0][random_idx_ir]  # left_road_points_lst[0][random_idx_ir]
    left_road_points_lst[1] = left_road_points_lst[1][random_idx_ir_cor]
    left_road_points_lst[2] = left_road_points_lst[2][random_idx_rgb]
    left_road_points = np.vstack(left_road_points_lst)
    print("On the left: %d sampled road points are from every point cloud" % sample_amount)

    sample_amount = int(0.8 * min(len(right_road_points_lst[0]), len(right_road_points_lst[1]), len(right_road_points_lst[2])))
    random_idx_ir = np.random.uniform(0, len(right_road_points_lst[0]), sample_amount).astype(int)  # sample by uniform distribution
    random_idx_ir_cor = np.random.uniform(0, len(right_road_points_lst[1]), sample_amount).astype(int)
    random_idx_rgb = np.random.uniform(0, len(right_road_points_lst[2]), sample_amount).astype(int)
    right_road_points_lst[0] = right_road_points_lst[0][random_idx_ir]  # right_road_points_lst[0][random_idx_ir]
    right_road_points_lst[1] = right_road_points_lst[1][random_idx_ir_cor]
    right_road_points_lst[2] = right_road_points_lst[2][random_idx_rgb]
    right_road_points = np.vstack(right_road_points_lst)
    print("On the right: %d sampled road points are from every point cloud" % sample_amount)

    road_points = np.vstack([left_road_points, right_road_points])

    road_vec = np.mean(left_road_points, axis=0) - np.mean(right_road_points, axis=0)
    road_vec = road_vec / np.linalg.norm(road_vec)
    traj_vec = 0.5 * (((traj_list[1][-1, :] - traj_list[1][0, :]) / np.linalg.norm((traj_list[1][-1, :] - traj_list[1][0, :]))) +
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
    road_plane_normal = np.cross(traj_vec, road_vec - proj_of_road_on_traj)
    # road_plane_normal = np.cross(traj_vec, [1, 0, 0])
    road_plane_normal = abs(road_plane_normal) / np.linalg.norm(road_plane_normal)  # unit vector. abs() makes sure that the normal
    # points up.
    # road_plane_d = -0.1075  # this number can be tweaked by looking at the error distribution shown later
    road_points_mean = np.mean(road_points, axis=0)
    road_plane_d = -np.dot(road_points_mean, road_plane_normal)
    print("Road plane is: %f*x + %f*y + %f*z + %f = 0" % (road_plane_normal[0], road_plane_normal[1], road_plane_normal[2], road_plane_d))

    # Debugging: visualize and verify the sampled points on the road
    Debugging = False
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
        plt.show()

    # fc = np.asarray([[11,37,181], [111,19,190], [62,149,38]]) / 255
    # lb = ["IR", "IR+cor", "RGB"]
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
                condition_1 = (data[:, 1] - np.mean(traj[idx_1:idx_2, 1]) > 0.035) & (
                        data[:, 1] - np.mean(traj[idx_1:idx_2, 1]) < 0.045)  # road is higher in y-axis in current coordinate system
                condition_3 = (np.min(traj[idx_1:idx_2, 0]) - data[:, 0] < 0.35) | (
                        data[:, 0] - np.max(traj[idx_1:idx_2, 0]) < 0.1)  # driving on the right of the road. Road points are closer on
                # the right.
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
        # plot_plane(data_road, normal, d)
        print("%d points are being used for analysis" % len(data_road))
        error_analysis(data_road, normal, d, True)
