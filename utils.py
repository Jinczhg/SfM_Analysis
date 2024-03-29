import math

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import mlab
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.stats import norm
import matplotlib.mlab as mlab


def align_point_clouds(moving_point_cloud, ref_point_cloud, moving_traj, ref_traj, transform_12, plot=False):
    pc_1 = np.vstack((moving_point_cloud[:, 0], moving_point_cloud[:, 1], moving_point_cloud[:, 2], np.ones(len(moving_point_cloud))))
    moving_point_cloud = np.matmul(transform_12, pc_1).T
    moving_traj = np.vstack((moving_traj[:, 0], moving_traj[:, 1], moving_traj[:, 2], np.ones(len(moving_traj))))
    moving_traj = np.matmul(transform_12, moving_traj).T
    if plot:
        pcd_view_1 = o3d.geometry.PointCloud()
        pcd_view_1.points = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector(moving_point_cloud))
        pcd_view_1.paint_uniform_color([1.0, 0, 0])
        pcd_view_2 = o3d.geometry.PointCloud()
        pcd_view_2.points = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector(ref_point_cloud))
        pcd_view_2.paint_uniform_color([0, 1.0, 0])
        pcd_view_moving_traj = o3d.geometry.PointCloud()
        pcd_view_moving_traj.points = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector(moving_traj))
        pcd_view_moving_traj.paint_uniform_color([0, 0, 1.0])
        pcd_view_ref_traj = o3d.geometry.PointCloud()
        pcd_view_ref_traj.points = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector(ref_traj))
        pcd_view_ref_traj.paint_uniform_color([1.0, 0, 1.0])

        viewer = o3d.visualization.Visualizer()
        viewer.create_window(width=640, height=480)
        viewer.add_geometry(pcd_view_1)
        viewer.add_geometry(pcd_view_2)
        viewer.add_geometry(pcd_view_moving_traj)
        viewer.add_geometry(pcd_view_ref_traj)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        viewer.run()
    return moving_point_cloud, ref_point_cloud, moving_traj, ref_traj


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


def signed_shortest_distance(x, y, z, normal, d):
    d = (normal[0] * x + normal[1] * y + normal[2] * z + d)
    e = (np.sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]))
    return d / e


def point_to_line_distance_3D(linePtsStart, linePtsEnd, pts):

    x = linePtsEnd - linePtsStart
    dist = np.linalg.norm(np.outer(np.dot(pts-linePtsEnd, x)/np.dot(x, x), x)+linePtsEnd-pts, axis=1)
    return dist


def road_points_from_traj(point_cloud, trajectory, ind_1, ind_2):
    condition_0 = (point_cloud[:, 2] > trajectory[ind_1, 2]) & (point_cloud[:, 2] < trajectory[ind_2, 2])
    condition_1 = (point_cloud[:, 1] - np.mean(trajectory[ind_1:ind_2, 1]) > 0.035) & (
            point_cloud[:, 1] - np.mean(trajectory[ind_1:ind_2, 1]) < 0.045)
    condition_left_road = (np.min(trajectory[ind_1:ind_2, 0]) - point_cloud[:, 0] > 0.35) & (
            np.min(trajectory[ind_1:ind_2, 0]) - point_cloud[:, 0] < 0.45)
    condition_right_road = (point_cloud[:, 0] - np.max(trajectory[ind_1:ind_2, 0]) > 0.1) & (
            point_cloud[:, 0] - np.max(trajectory[ind_1:ind_2, 0]) < 0.2)  # driving on the right of the road. Road points are closer on
    # the right.
    left_road_points = point_cloud[condition_0 & condition_1 & condition_left_road]
    right_road_points = point_cloud[condition_0 & condition_1 & condition_right_road]
    return left_road_points, right_road_points


def vis_pc(point_cloud):
    pcd_view = o3d.geometry.PointCloud()
    pcd_view.points = o3d.utility.Vector3dVector(point_cloud)
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(width=640, height=480)
    viewer.add_geometry(pcd_view)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    viewer.run()
    viewer.destroy_window()
    viewer.close()


def plot_plane(point_cloud, normal, d):
    if point_cloud.shape[1] == 3:
        maxx = np.max(point_cloud[:, 0])
        maxy = np.max(point_cloud[:, 1])
        minx = np.min(point_cloud[:, 0])
        miny = np.min(point_cloud[:, 1])

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # plot original points
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=5)

        # compute needed points for plane plotting
        xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
        zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

        ax.plot_surface(xx, yy, zz, alpha=0.2)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
    else:
        exit("Input point cloud is not a Nx3 array")


def error_analysis(data, normal, d, plot=False):
    # point-to-plane error analysis
    dist = np.zeros(len(data))
    for t in range(len(data)):
        x = data[t, 0]
        y = data[t, 1]
        z = data[t, 2]
        dist[t] = signed_shortest_distance(x, y, z, normal, d)

    #avg_dist = np.average(abs(dist))
    MSE = np.square(dist).mean()
    RMSE = math.sqrt(MSE)
    print("Fitting RMSE is", RMSE)
    std_dist = np.std(abs(dist))
    print("Fitting Std is", std_dist)

    if(plot):
        plt.figure()
        n, bins, patches = plt.hist(dist, bins='auto', density=True)  # arguments are passed to np.histogram
        # density = stats.gaussian_kde(dist)
        # plt.plot(bins, density(bins))
        (mu, sigma) = norm.fit(dist)
        # print("mu = " + str(mu) + "; sigma = " + str(sigma))
        y = norm.pdf(bins, mu, sigma)
        plt.plot(bins, y, 'r--', linewidth=2)
        plt.xlabel('Point Distance from the Road Plane', fontdict={'fontsize': 15})
        plt.ylabel('Probability Density', fontdict={'fontsize': 15})
        plt.ylim(0, 40)
        plt.xlim(-0.04, 0.04)
        plt.title(r'$ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma), fontdict={'fontsize': 12})
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()