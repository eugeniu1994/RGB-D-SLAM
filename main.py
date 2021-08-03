import cv2
from os import path
import numpy as np
from PIL import Image
from numbers import Number
import open3d as o3d
from viewer import Viewer
from optimization import *
import g2o

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

debug = False


class Frame(object):
    def __init__(self, rgb_file, depth_file, orb, descriptor=None):
        super(Frame, self).__init__()
        self.rgb = self.read_image(rgb_file)
        self.depth = self.read_image(depth_file)
        self.kps, self.des = self.ComputeKPointsAndDescriptors(self.rgb, orb, descriptor)

    def read_image(self, filename):
        if 'depth' in filename:
            return Image.open(filename)
            img = cv2.imread(filename)
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            return cv2.imread(filename)

    def ComputeKPointsAndDescriptors(self, rgb, orb, descriptor = None):
        if descriptor is None:
            kps, des = orb.detectAndCompute(rgb, None)
        else:
            kps = orb.detect(rgb)
            kps, des = descriptor.compute(rgb, kps)

        self.outimg1 = cv2.drawKeypoints(rgb, keypoints=kps, outImage=None)
        return kps, des

class Isometry3d(object):
    """3d rigid transform."""

    def __init__(self, R, t):
        self.R = R
        self.t = t

    def matrix(self):
        m = np.identity(4)
        m[:3, :3] = self.R
        m[:3, 3] = self.t
        return m

    def inverse(self):
        return Isometry3d(self.R.T, -self.R.T @ self.t)

    def __mul__(self, T1):
        R = self.R @ T1.R
        t = self.R @ T1.t + self.t
        return Isometry3d(R, t)

class rgbd_slam(object):
    def __init__(self, show_match=True, use_GFTT = True):
        self.RGBFileNamePath = 'data/traj3n_frei_png/rgb'
        self.DepthFileNamePath = 'data/traj3n_frei_png/depth'
        self.K = np.matrix([[481.20, 0, 319.5], [0, 480.0, 239.5], [0, 0, 1]])  # camera intrinsic parameters
        self.scale = 5000  # for depth
        self.fx, self.fy, self.cx, self.cy = 481.20, 480.0, 319.5, 239.5
        self.D = np.array([0, 0, 0, 0], dtype=np.float32)  # no distortion

        nfeatures = None
        #nfeatures = 1024
        self.detector = cv2.ORB_create(nfeatures = nfeatures)
        FLAN_INDEX_KDTREE = 0
        index_params, search_params = dict(algorithm=FLAN_INDEX_KDTREE, trees=5), dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # create BFMatcher object
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.use_GFTT = use_GFTT
        if use_GFTT:
            self.detector = cv2.GFTTDetector_create(maxCorners=1000, minDistance=15.0,
                                                            qualityLevel=0.001, useHarrisDetector=False)
            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)
            self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            self.descriptor_extractor = None


        self.rk = g2o.RobustKernelDCS()
        self.graph_optimizer = PoseGraphOptimization()
        self.width, self.height = 640, 480

        self.pose = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]).astype(float)
        self.colour = []
        self.pts_obj = []
        self.pts_obj_prev = []
        self.show_match = show_match

        self.lk_params = dict(winSize  = (21, 21),
				#maxLevel = 3,
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.sift = cv2.SIFT_create()

        self.poses = []# Queue()
        self.points3D = []# Queue()
        self.points2D = []
        self.features_ = []

        self.BA_images = []

    def point_cloud(self, depth):
        # return a dense point cloud map
        return []
        points, colour = [], []
        for v in range(depth.size[1]):
            for u in range(depth.size[0]):
                # color = rgb.getpixel((u, v))
                color = self.curr_frame.rgb[v, u]
                Z = depth.getpixel((u, v))
                if Z == 0: continue
                Z = Z / self.scale
                X = (u - self.cx) * Z / self.fx
                Y = (v - self.cy) * Z / self.fy
                # points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
                points.append([X, Y, Z])
                colour.append([color])
        points = np.asarray(points)
        inrange = np.where((points[:, -1] > 0.01) & (points[:, -1] < 4.))

        self.colour = np.array(colour)[inrange[0]]
        skip = 150
        return points[inrange[0]][::skip]
        # return points

    def point_cloud_(self, depth):
        rows, cols = depth.size
        # depth = depth.convert('L')

        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        valid = (depth > 0.01) & (depth < 4.)
        depth = np.asarray(depth) / float(self.scale)
        # z = np.where(valid, depth / float(self.scale), 0)
        z = np.where(valid, depth, 0)
        x = np.where(valid, z * (c - self.cx) / self.fx, 0)
        y = np.where(valid, z * (r - self.cy) / self.fy, 0)
        cloud = np.dstack((x, y, z)).reshape(-1, 3)
        # inrange = np.where((z > 0.01))
        return cloud  # [inrange[0]]

    def point_cloud_(self, depth):
        rows, cols = depth.shape
        pointcloud, colors = [], []
        for m in range(0, rows):
            for n in range(0, cols):
                d = depth[m][n]
                if d == 0:
                    pass
                else:
                    point = self.point2dTo3d(n, m, d)
                    pointcloud.append(point)
        # b = rgb[m][n][0]
        # g = rgb[m][n][1]
        # r = rgb[m][n][2]
        # color = (r << 16) | (g << 8) | b
        # colors.append(int(color))
        return pointcloud  # , colors

    def point2dTo3d(self, n, m, d):
        z = float(d) / self.scale
        x = (n - self.cx) * z / self.fx
        y = (m - self.cy) * z / self.fy
        point = np.array([x, y, z], dtype=np.float32)
        return point

    def normofTransform(self, rvec, tvec):
        return abs(min(cv2.norm(rvec), np.pi * 2 - cv2.norm(rvec))) + abs(cv2.norm(tvec))

    def transformMatrix(self, rvec, tvec):
        r, t = np.matrix(rvec), np.matrix(tvec)
        R, _ = cv2.Rodrigues(r)
        Rt = np.hstack((R, t))
        T = np.vstack((Rt, np.matrix([0, 0, 0, 1])))
        return T

    def get_color(self, img, pt):
        x = int(np.clip(pt[0], 0, self.width - 1))
        y = int(np.clip(pt[1], 0, self.height - 1))
        color = img[y, x]
        if isinstance(color, Number):
            color = np.array([color, color, color])
        return color[::-1] / 255.

    def solvePnP(self, frame1, frame2, use_BF = False):
        kp1, kp2, des1, des2, depth = frame1.kps, frame2.kps, frame1.des, frame2.des, frame1.depth
        goodMatches = []
        if use_BF:
            # Match descriptors.
            matches = self.bf.match(des1, des2)
            # Sort them in the order of their distance.
            goodMatches = sorted(matches, key=lambda x: x.distance)#[:30]
        else:
            matches = self.matcher.knnMatch(np.asarray(des1, dtype=np.float32), np.asarray(des2, dtype=np.float32), k=2)
            for m1, m2 in matches:
                if m1.distance < 0.7 * m2.distance:
                    goodMatches.append(m1)
        if debug:
            print("good matches = {}".format(np.shape(goodMatches)))

        self.pts_obj, self.pts_img2, self.pts_img1 = [], [], []
        self.colour = []
        self.features = []
        for i in range(0, len(goodMatches)):
            p = kp1[goodMatches[i].queryIdx].pt
            # d = depth[int(p[1])][int(p[0])]
            d = depth.getpixel((int(p[0]), int(p[1])))
            if d == 0:
                pass
            else:
                p2 = kp2[goodMatches[i].trainIdx].pt
                #dif = abs(cv2.norm(p) - cv2.norm(p2))
                #if dif > .1:
                    #print('dif -> {}'.format(dif))
                    #pass
                self.pts_img2.append(p2)
                self.pts_img1.append(p)
                pd = self.point2dTo3d(p[0], p[1], d)
                self.pts_obj.append(pd)
                # c = frame1.rgb[int(p[1])][int(p[0])]
                self.colour.append(self.get_color(img=frame1.rgb, pt=p))
                self.features.append(des1[goodMatches[i].queryIdx])

        self.pts_obj, self.pts_img2 = np.array(self.pts_obj), np.array(self.pts_img2)
        self.pts_img1 = np.array(self.pts_img1)
        if debug:
            print('pts_obj -> {}, pts_img->{}'.format(np.shape(self.pts_obj), np.shape(self.pts_img2)))
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(self.pts_obj, self.pts_img2, self.K, self.D, useExtrinsicGuess=False)

        #retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            #self.pts_obj, pts_img,self.K, self.D, None, None,False, 50, 2.0, 0.9, None)

        #retval, rvec, tvec = cv2.solvePnP(self.pts_obj, pts_img, self.K, self.D, useExtrinsicGuess=False)

        T = self.transformMatrix(rvec, tvec)
        if self.show_match and inliers is not None:
            self.match_features = cv2.drawMatches(frame1.rgb, kp1, frame2.rgb, kp2,[goodMatches[i] for i in inliers.ravel()], None, flags=2)
            #cv2.imshow('prev -> current features', cv2.resize(self.match_features, None, fx=.6, fy=.6))
            #cv2.waitKey(1)
        if inliers is None:
            inliers = []

        return rvec, tvec, inliers, T

    def Front_End(self, RGBFileName, DepthFileName, lastFrame, MIN_INLIERS=5):
        self.currentFrame = Frame(rgb_file=RGBFileName, depth_file=DepthFileName, orb=self.detector, descriptor=self.descriptor_extractor)
        rvec, tvec, inliers, T = self.solvePnP(frame1=lastFrame, frame2=self.currentFrame)
        r, t = T[:3, :3], np.asarray(T[:3, -1]).squeeze()
        T = Isometry3d(R=r, t=t).inverse().matrix()
        if debug:
            print('inliers -> {}'.format(np.shape(inliers)))

        if len(inliers) < MIN_INLIERS:
            print('pass len(pnp.inliers) < MIN_INLIERS -> {}'.format(len(inliers)))
            pass
        else:
            pointCloud = self.point_cloud(self.currentFrame.depth)
            # p = slamBase.transformPointCloud(p0, pnp.T)
            # pointCloud = slamBase.addPointCloud(pointCloud.to_list(), p)
            R, _ = cv2.Rodrigues(tvec)
            return pointCloud, lastFrame, self.currentFrame, T, R, tvec, inliers

    def lost_tracking(self, T):
        move = self.normofTransform(rvec=T[:3, :3], tvec=T[:3, -1])
        # print('move -> {}'.format(move))
        if move > 2.5:
            print('===================LOST TRACKING======================')
            return True
        return False

    def Relocalize(self, use_o3d=False, T=None):
        print('===================RELOCALIZE======================')
        if not use_o3d:
            kp1, kp2, des1, des2, depth = self.lastFrame.kps, self.currentFrame.kps, self.lastFrame.des, self.currentFrame.des, self.lastFrame.depth
            kp1, des1 = self.sift.detectAndCompute(self.lastFrame.rgb, None)
            kp2, des2 = self.sift.detectAndCompute(self.currentFrame.rgb, None)

            prevPts = np.array([p.pt for p in kp1], dtype=np.float32)
            nextPts = np.array([p.pt for p in kp2], dtype=np.float32)
            kp2, st, err = cv2.calcOpticalFlowPyrLK(prevImg=self.lastFrame.rgb, nextImg=self.currentFrame.rgb,
                                                    prevPts = prevPts, nextPts=nextPts,
                                                    **self.lk_params)  # shape: [k,2] [k,1] [k,1]
            st = st.reshape(st.shape[0])
            kp1, kp2 = prevPts[st == 1], kp2[st == 1]
            self.pts_obj, pts_img = [], []
            self.colour = []
            for i in range(0, len(kp1)):
                p = kp1[i]
                d = depth.getpixel((int(p[0]), int(p[1])))
                if d == 0:
                    pass
                else:
                    p2 = kp2[i]
                    pts_img.append(p2)
                    pd = self.point2dTo3d(p[0], p[1], d)
                    self.pts_obj.append(pd)
                    self.colour.append(self.get_color(img=self.lastFrame.rgb, pt=p))
            self.pts_obj, pts_img = np.array(self.pts_obj), np.array(pts_img)
            if debug:
                print('pts_obj -> {}, pts_img->{}'.format(np.shape(self.pts_obj), np.shape(pts_img)))
            #retval, rvec, tvec, inliers = cv2.solvePnPRansac(self.pts_obj, pts_img, self.K, self.D,useExtrinsicGuess=False)
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
             self.pts_obj, pts_img,self.K, self.D, None, None,False, 50, 2.0, 0.9, None)

            T = self.transformMatrix(rvec, tvec)

            #rvec, tvec, inliers, T = self.solvePnP(self.currentFrame, self.lastFrame)
            r, t = T[:3, :3], np.asarray(T[:3, -1]).squeeze()
            T = Isometry3d(R=r, t=t).inverse().matrix()
            #T = Isometry3d(R=r, t=t).matrix()
        else:
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(self.pts_obj_prev)
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(self.pts_obj)
            self.threshold = 0.0001  # Distance threshold
            # Initial transformation matrix, generally provided by coarse registration
            self.trans_init = o3d.np.asarray([[1.0, 0.0, 0.0, 0.0],
                                              [0.0, 1.0, 0.0, 0.0],
                                              [0.0, 0.0, 1.0, 0],
                                              [0.0, 0.0, 0.0,1.0]]) if T is None else T
            self.ICP_max_iter = 50
            icp_p2p = o3d.registration.registration_icp(source, target, self.threshold, self.trans_init,
                                                        o3d.registration.TransformationEstimationPointToPoint(),
                                                        # Execute point-to-point ICP algorithm
                                                        o3d.registration.ICPConvergenceCriteria(
                                                            max_iteration=self.ICP_max_iter))
            T = icp_p2p.transformation
            print(T)

        return T

    def BA(self):
        #print('BA')
        #points_3d = np.array(self.points3D, dtype=object)          #3D feature points points
        #camera_poses = np.array(self.poses, dtype=object)         #camera poses

        #print('points_3d->{}, points_2d->{}, camera_params->{}'.format(len(self.points3D),len(self.points2D), len(self.poses)))
        #print('self.features_ -> {}'.format(np.shape(self.features_)))
        #have to discard the last value (dont have the feature for it)
        '''
         N = 10 - n_cameras
         M = n_points -> unique 3D points
         camera_params: N x (rvec, tvec, fx, d1, d2) = N x 10
         points_3d: M x 3 -> with duplicates removed
         points_2d:  n_observations x 2 -> also stacked
         camera_ind: n_observations x 1 : ind of the 3D point that can be seen from this camera
                points (from 0 to n_points - 1) -> maybe 
         camera_ind: (n_observations,)
                 indices of cameras (from 0 to n_cameras - 1) involved in each observation.
            
        -compute the duplicates based on 2D features and eliminate the 3D points of the duplicates
        
        use the features to match across the 10 measurements -> and optimize ith BA
        
        '''

        #create views
        #create matches
        # reconstruct

        '''
        ----take images from BA_images
        ----create views
        -crate matches
        '''

        self.poses = []# Queue()
        self.points3D = [] # Queue()
        self.points2D = []
        self.features_ = []
        self.BA_images = []

    def doJob(self):
        START_INDEX, END_INDEX = 1, 1240
        #START_INDEX = 499
        viewer = Viewer()
        rgb_path = path.join(self.RGBFileNamePath, str(START_INDEX) + '.png')
        depth_path = path.join(self.DepthFileNamePath, str(START_INDEX) + '.png')
        self.lastFrame = Frame(rgb_file=rgb_path, depth_file=depth_path, orb=self.detector, descriptor=self.descriptor_extractor)
        self.last_cloud = self.point_cloud(self.lastFrame.depth)
        viewer.update_image(self.lastFrame.outimg1)
        # back-end------------------------------------------------------
        id = 0  # first node of the posegraph
        self.graph_optimizer.add_vertex(id, self.pose, is_fixed=True)

        #cv2.imshow('name ',self.lastFrame.outimg1)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        for i in range(START_INDEX, END_INDEX):
            rgb_path = path.join(self.RGBFileNamePath, str(i + 1) + '.png')
            depth_path = path.join(self.DepthFileNamePath, str(i + 1) + '.png')
            self.pointCloud, _, self.currentFrame, T, R, t, _ = self.Front_End(rgb_path, depth_path, self.lastFrame)

            a = self.lost_tracking(T)
            if a:
                print('lost tracking at image {}',format(i))
                T = np.eye(4)
                #continue

            self.lastFrame = self.currentFrame
            self.last_cloud = self.pointCloud


            if a is False:
                self.pose = T @ self.pose  # update the current pose

                # Back-end -> posegraph optimization
                id += 1
                self.graph_optimizer.add_vertex(id=id, pose=self.pose)
                self.graph_optimizer.add_edge(vertices=[id - 1, id], measurement=T, robust_kernel=self.rk)
                self.graph_optimizer.optimize()

                nodes_optimized = [i.estimate().matrix() for i in self.graph_optimizer.vertices().values()]
                optimized_pose = np.asarray(nodes_optimized[0]).squeeze()
                R, t = optimized_pose[:3, :3], np.asarray(optimized_pose[:3, -1]).reshape(3, 1)
                Rt = np.hstack((R, t))
                self.pose = np.vstack((Rt, np.matrix([0, 0, 0, 1])))

            self.BA_images.append((cv2.imread(rgb_path), i, self.pose))
            if i % 5 == 0:
                self.BA()

            ones = np.ones(len(self.pts_obj)).reshape(-1, 1)
            self.pts_obj = np.hstack((self.pts_obj, ones))
            self.pts_obj = self.pts_obj @ self.pose.T
            self.pts_obj = self.pts_obj[:, :3] / np.asarray(self.pts_obj[:, -1]).reshape(-1, 1)
            self.pts_obj_prev = self.pts_obj

            viewer.update_pose(pose=g2o.Isometry3d(self.pose), cloud=self.pts_obj, colour=np.array(self.colour))
            #viewer.update_image(self.currentFrame.outimg1)
            viewer.update_image(cv2.resize(self.match_features, None, fx=.6, fy=.6))

            self.poses.append(self.poses)
            self.points3D.append(self.pts_obj)
            self.points2D.append(self.pts_img2)  #self.points2D.append(self.pts_img1)
            self.features_.append(self.features)
            # add tracking and relocalizing with ICP (NOT WORKING YET)
            # add loop closure detection
            # add BA for better mapping in a separate thread
            # add threads, make it faster

        viewer.stop()

        '''fig2 = plt.figure(figsize=(8, 8))
        fig2.suptitle('Estimate trajectory', fontsize=12)
        ax2 = plt.axes(projection='3d')
        ax2.set_xlabel('X(m)')
        ax2.set_ylabel('Y(m)')
        ax2.set_zlabel('Z(m)')
        ax2.grid()
        print('prev----------------------')
        print(self.pts_obj_prev)
        print('current----------------------')
        print(self.pts_obj)
        ax2.scatter(*self.pts_obj_prev.T, marker='o', c='b', alpha=1, s=10, label='prev_pts') #shows translation
        ax2.scatter(*self.pts_obj.T, marker='*', c='r', alpha=1, s=8, label='curr_pts') #shows translation
        ax2.legend()
        fig2.canvas.draw_idle()
        plt.pause(0.001)
        cv2.imshow('lastFrame', self.lastFrame.outimg1)
        cv2.imshow('curr_frame', self.curr_frame.outimg1)
        cv2.imshow('match_features', self.match_features)
        plt.show()'''
        # compare with the ground truth

if __name__ == '__main__':
    obj = rgbd_slam(show_match=True, use_GFTT = True)
    obj.doJob()
