from typing import List, Optional
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import time
from .configs import load_map_config
from .obs_gpdf_vec import GassianProcessDistanceField as GPDF

from skimage import measure

from sklearn.cluster import DBSCAN


class Env_Approx:
    def __init__(self, shared_lock, margin:float=0.0, rho:float=10, original_env:str = None, radius:float=1.0) -> None:
        self.margin = margin
        self.rho = rho
        self.rho_uni = rho
        self.original_env = original_env
        self.radius = radius

        self.num_dyn_circle = 0


        self.gpdf_set:List[GPDF] = []
        self.gpdf_env:List[GPDF] = []
        self.gpdf_mmp:List[GPDF] = []
        self.offset = []
        self.xc_ref = []

        self.lock = shared_lock
        self.precomp_h_robot = None
        self.precomp_grad_shifted = None

        self.monitor_time = True

        empty_gpdf = GPDF()
        self.gpdf_set.append(empty_gpdf)
        self.gpdf_env.append(empty_gpdf)

        self.offset.append(0)
        self.xc_ref.append([0.0, 0.0])

        self.xc = np.empty((0,2))
        self.dxc = np.empty((0,2))

        self.resolution = 0.1
        self._inv_res = 1.0 / self.resolution

        self.dynamic_obstacle_threshold = 0.8
        self._human_strip_radius = 0.8
        self.obs_offset = -0.1

        map_cfg = load_map_config(self.original_env)
        self.x_min = map_cfg['x_min']
        self.x_max = map_cfg['x_max']
        self.y_min = map_cfg['y_min']
        self.y_max = map_cfg['y_max']
        self.other_num_samples = map_cfg['other_num_samples']

        self.x_cells = int((self.x_max - self.x_min) / self.resolution) + 1
        self.y_cells = int((self.y_max - self.y_min) / self.resolution) + 1

        self.update_mmp_gpdf([])

        self.other_obs_pt_contour_single = None
        self.other_obs_pt_contours = None
        self.other_obs_centroids = []

        self._dbscan_eps_default = 2
        self._dbscan_eps_confined = 2
        self.db_cluster = DBSCAN(eps=self._dbscan_eps_default, min_samples=3)

        self.human_centroids = None

        self.lidar_track_human_center = []
        self.lidar_track_human_vel = []

        self.pic_numbers = 0

    def set_confined(self, is_confined: bool):
        self.db_cluster.eps = self._dbscan_eps_confined if is_confined else self._dbscan_eps_default

    @property
    def mmp(self) -> bool:
        return len(self.gpdf_mmp) > 0

    @property
    def num_dyn_mmp(self) -> int:
        if self.mmp:
            return len(self.gpdf_mmp)
        else:
            return 0

    def update_mmp_gpdf(self, coords_list: List[np.ndarray]):
        self.gpdf_mmp = [GPDF(coords) for coords in coords_list]
        self.gpdf_set = self.gpdf_mmp + self.gpdf_env

        self.xc = 100*np.ones((self.num_dyn_mmp,2))
        self.dxc = 0*np.ones((self.num_dyn_mmp,2))

    def snapshot(self) -> SimpleNamespace:
        with self.lock:
            return SimpleNamespace(
                gpdf_set=self.gpdf_set,
                gpdf_env=self.gpdf_env,
                xc_ref=self.xc_ref,
                offset=self.offset,
                num_dyn_circle=self.num_dyn_circle,
                precomp_h_robot=self.precomp_h_robot,
                precomp_grad_shifted=self.precomp_grad_shifted,
                xc=self.xc,
                dxc=self.dxc,
                other_obs_pt_contour_single=self.other_obs_pt_contour_single,
            )

    def h_grad_set(self, x, t=0, snap: Optional[SimpleNamespace] = None):
        if snap is None:
            snap = self.snapshot()
        gpdf_set = snap.gpdf_set
        xc_ref = snap.xc_ref
        offset = snap.offset

        gpdf_num = len(gpdf_set)
        dis_set = np.empty((0, x.shape[0]))
        grad_set = np.empty((0, 2))
        for k in range(gpdf_num):
            dis, grad  = gpdf_set[k].dis_normal_func(x)
            dis = np.array(dis.reshape(1, -1))
            if k==0:
                dis_set = dis
                grad_set = grad.reshape(1, 2)
            else:
                dis_set = np.concatenate((dis_set, dis),axis=0)
                grad_set = np.concatenate((grad_set, grad.reshape(1, 2)), axis=0)

        for n in range(len(xc_ref)):
            dis_set[-(n+1)] = dis_set[-(n+1)] + offset[-(n+1)]

        if self.mmp:
            gradt = np.array(self.h_grad_t(grad_set, snap.dxc)).reshape(-1,1)
            return dis_set-1, grad_set, gradt, np.zeros(grad_set.shape), np.zeros(grad_set.shape)
        else:
            dis, grad = self.h_gradc(x, t=t, xc=snap.xc, dxc=snap.dxc, num_dyn_circle=snap.num_dyn_circle)
            grad = np.concatenate((grad, grad_set),axis=0)
            dis = np.concatenate((dis - 0.2, dis_set), axis=0)
            xc_ref_arr = np.asarray(xc_ref).reshape(-1, 2)
            xc = np.concatenate((snap.xc, xc_ref_arr),axis=0)
            dxc = np.concatenate((snap.dxc, np.zeros_like(xc_ref_arr)),axis=0)
            gradt = np.array(self.h_grad_t(grad,dxc)).reshape(-1,1)

            return dis-1, grad, gradt, xc, dxc

    def h_grad_vector(self, x, obstacle_idx=-1, dynamic_obstacle=False):

        if dynamic_obstacle:
            gpdf = self.gpdf_mmp
        else:
            gpdf = self.gpdf_env

        if obstacle_idx < 0:
            dis_set = np.zeros((len(gpdf),len(x)))
            grad_set = np.zeros((len(gpdf),len(x),2))
            for k in range(len(gpdf)):
                dis, grad  = gpdf[k].dis_normal_func(x)
                dis = dis.reshape(1, -1)
                if k==0:
                    dis_set = np.array(dis)
                    grad_set = grad.T[None,:,:]
                else:
                    dis_set = np.concatenate((dis_set,dis),axis=0)
                    grad_set = np.concatenate((grad_set, grad.T[None, :,:]), axis=0)
        else:
            dis_set, grad_set  = gpdf[obstacle_idx].dis_normal_func(x)
            if self.mmp:
                return np.array(dis_set)-1, np.array(grad_set.T)
            dis_set = dis_set.reshape(1,-1)
            grad_set = grad_set.T[None,:,:]

        grad_num = np.sum(np.exp(-self.rho*dis_set.reshape(dis_set.shape[0],dis_set.shape[1],1))*grad_set, axis=0)
        grad_den = np.sum(np.exp(-self.rho*dis_set),axis=0)
        dis_uni = -1/self.rho*np.log(grad_den)-1
        grad_uni = grad_num/grad_den.reshape(-1,1)
        return dis_uni, grad_uni

    def h_gradc(self, x, t=0, xc=None, dxc=None, num_dyn_circle=None):
        if xc is None:
            xc = self.xc
        if dxc is None:
            dxc = self.dxc
        if num_dyn_circle is None:
            num_dyn_circle = self.num_dyn_circle

        if x.shape[0] != 1:
            diff = (-xc[:num_dyn_circle, np.newaxis] + x).reshape(-1, xc[:num_dyn_circle].shape[1])
            den = np.linalg.norm(diff, axis=1)
            dis = den.reshape(xc[:num_dyn_circle].shape[0],x.shape[0])-self.radius-self.margin+1
            grad = (diff/den.reshape(-1,1)).reshape(xc[:num_dyn_circle].shape[0],x.shape[0],2)
            return dis, grad

        xc = xc.reshape(num_dyn_circle, 2)
        dxc = dxc.reshape(num_dyn_circle, 2)

        den = np.linalg.norm(x-(xc[:num_dyn_circle]+dxc[:num_dyn_circle]*t),axis=1).reshape(xc[:num_dyn_circle].shape[0],1)
        dis = den - self.radius - self.margin + 1
        grad = (x-xc[:num_dyn_circle])/den
        return dis, grad

    def h_grad_uni(self, x, idx):
        h_set = np.zeros((len(idx), len(x)))
        grad_set = np.zeros((len(idx), len(x), 2))
        for k, id in enumerate(idx):
            h_set[k], grad_set[k] = self.gpdf_set[id].dis_normal_func(x)
            h_set[k] = h_set[k] + self.offset[id]
        grad_num = np.sum(np.exp(-self.rho_uni * h_set[:, :, None]) * grad_set, axis=0)
        grad_den = np.sum(np.exp(-self.rho_uni * h_set), axis=0, keepdims=True)
        dis_uni = -1 / self.rho_uni * np.log(grad_den) - 1
        grad_uni = grad_num / grad_den
        if np.linalg.norm(grad_uni) < 0.5:
            h_min_idx = np.argmin(h_set)
            return dis_uni, grad_set[h_min_idx]
        return dis_uni, grad_uni / np.linalg.norm(grad_uni, axis=1)

    def h_grad_t(self, grad, total_dxc):

        if len(grad.shape)>2:
            return -np.sum(grad @ total_dxc.reshape(total_dxc.shape[0], total_dxc.shape[1],1), axis=2)
        else:
            return -np.sum(grad * total_dxc, axis=1)

    def hes_c(self, x, t=0):
        hes = np.zeros((self.xc.shape[0],2,2))
        diff = x-(self.xc+self.dxc*t)
        dist = np.linalg.norm(diff,axis=1).reshape(self.xc.shape[0],1)
        dist32 = dist**3
        hes[:,0,0] = (1/dist-diff[:,0,None]**2/dist32).squeeze()
        hes[:,1,1] = (1/dist-diff[:,1,None]**2/dist32).squeeze()
        hes[:,0,1] = (-diff[:,0,None]*diff[:,1,None]/dist32).squeeze()
        hes[:,1,0] = hes[:,0,1]
        return hes

    def update_from_msg(self, msg) -> None:
        gpdf_env = []
        for gp in msg.models:
            gpdf = GPDF()
            coords = np.array([[p.x, p.y] for p in gp.points], dtype=float)
            gpdf.pc_coords = coords
            gpdf.gpdf_model = np.array(gp.weights, dtype=float).reshape(-1, 1)
            if gp.dis_field:
                gpdf.dis_field = np.array(gp.dis_field, dtype=np.float32).reshape(gp.grid_rows, gp.grid_cols)
                gpdf.grid_meta = {
                    'xmin': gp.grid_xmin,
                    'ymin': gp.grid_ymin,
                    'dx': gp.grid_dx,
                    'dy': gp.grid_dy,
                }
            if gp.has_centroid:
                gpdf.centroid = np.array([gp.centroid.x, gp.centroid.y], dtype=float)
            gpdf_env.append(gpdf)

        xc  = np.array([[p.x, p.y] for p in msg.xc],  dtype=float) if msg.xc  else np.empty((0, 2))
        dxc = np.array([[p.x, p.y] for p in msg.dxc], dtype=float) if msg.dxc else np.empty((0, 2))
        obs_pts  = np.array([[p.x, p.y] for p in msg.obs_contour_pts], dtype=float) if msg.obs_contour_pts else None

        if gpdf_env:
            precomp_h = np.array([gp.h_robot for gp in msg.models], dtype=float).reshape(-1, 1)
            precomp_grad = np.array(
                [[gp.grad_h_shifted_x, gp.grad_h_shifted_y] for gp in msg.models], dtype=float
            )
        else:
            precomp_h = None
            precomp_grad = None

        with self.lock:
            self.gpdf_env = gpdf_env
            self.gpdf_set = self.gpdf_mmp + gpdf_env
            self.offset = list(msg.offsets)
            self.xc_ref = [[0.0, 0.0]] * len(gpdf_env)
            self.num_dyn_circle = int(msg.num_dyn_circle)
            self.xc = xc
            self.dxc = dxc
            self.other_obs_pt_contour_single = obs_pts
            self.precomp_h_robot = precomp_h
            self.precomp_grad_shifted = precomp_grad

    def update(self, occupied_points: np.ndarray, human_centroids_vel, debug=False):
        self.human_centroids = human_centroids_vel

        if debug and occupied_points is not None and len(occupied_points) > 0:
            plt.figure(figsize=(6, 5))
            plt.scatter(occupied_points[:, 0], occupied_points[:, 1], s=3)
            plt.savefig(f"2D_lidardebug_{self.pic_numbers}.png", dpi=100, bbox_inches='tight')
            self.pic_numbers += 1
            plt.close()

        time_update = time.time()
        self._cluster_and_contour(occupied_points)
        cluster_time = (time.time() - time_update) if self.monitor_time else 0.0

        start_time = time.time()
        gpdf_set, gpdf_env, xc_ref, offset, xc, dxc, num_dyn = self._gpdf_calculation()
        gpdf_time = (time.time() - start_time) if self.monitor_time else 0.0

        with self.lock:
            self.gpdf_set = gpdf_set
            self.gpdf_env = gpdf_env
            self.xc_ref = xc_ref
            self.offset = offset
            self.xc = xc
            self.dxc = dxc
            self.num_dyn_circle = num_dyn

        return cluster_time, gpdf_time

    @staticmethod
    def _points_preprocess(cluster_points, n_samples):
        cluster_points = np.unique(cluster_points, axis = 0)

        if len(cluster_points) == n_samples:
            return cluster_points

        if len(cluster_points) > n_samples:
            return Env_Approx._random_sampling(cluster_points, n_samples)

        if len(cluster_points) < n_samples:
            return Env_Approx._jittering(cluster_points, n_samples)

    @staticmethod
    def _random_sampling(points, n_samples):
        selected = points[np.random.choice(points.shape[0], size=n_samples, replace=False)]
        return selected

    @staticmethod
    def _jittering(cluster_points, n_samples, jitter_scale = 0.01, rng = None):
        cp = np.asarray(cluster_points, dtype=np.float32, order="C")
        N, D = cp.shape if cp.ndim == 2 else (len(cp), 1)
        if N == 0:
            raise ValueError("cluster_points must be non-empty")

        m = n_samples - N

        idx = np.arange(m) % N
        repeated = cp[idx].copy()

        if rng is None:
            rng = np.random.default_rng()
        jitter = rng.normal(loc=0.0, scale=jitter_scale, size=repeated.shape).astype(cp.dtype, copy=False)

        repeated += jitter

        out = np.empty((n_samples, D), dtype=cp.dtype)
        out[:N] = cp
        out[N:] = repeated
        return out

    def _gpdf_calculation(self):
        gpdf_set:List[GPDF] = []
        gpdf_env:List[GPDF] = []
        xc_ref = []
        offset = []

        if self.human_centroids is not None and len(self.human_centroids) >= 0:
            self._collect_centroid_vel()

        xc_inupdate = []
        dxc_inupdate = []
        num_dyn_circle_inupdate = 0
        claimed_humans: set = set()

        human_centers = np.array(self.lidar_track_human_center) if self.lidar_track_human_center else None
        human_vels   = np.array(self.lidar_track_human_vel)    if self.lidar_track_human_vel    else None

        for i, points in enumerate(self.other_obs_pt_contours):
            cluster_points = np.array(points)

            matched_indices = []
            if human_centers is not None:
                for hidx, hc in enumerate(human_centers):
                    if np.min(np.linalg.norm(cluster_points - hc, axis=1)) < self.dynamic_obstacle_threshold:
                        matched_indices.append(hidx)

            for hidx in matched_indices:
                if hidx not in claimed_humans:
                    xc_inupdate.append(human_centers[hidx])
                    dxc_inupdate.append(human_vels[hidx])
                    num_dyn_circle_inupdate += 1
                    claimed_humans.add(hidx)

            if matched_indices:
                keep = np.ones(len(cluster_points), dtype=bool)
                for hidx in matched_indices:
                    keep &= np.linalg.norm(cluster_points - human_centers[hidx], axis=1) > self._human_strip_radius
                cluster_points = cluster_points[keep]
                if len(cluster_points) < 15:
                    continue
                cluster_points = Env_Approx._points_preprocess(cluster_points, self.other_num_samples)

            static_gpdf = GPDF(cluster_points)
            static_gpdf.centroid = (
                self.other_obs_centroids[i] if i < len(self.other_obs_centroids) else cluster_points.mean(axis=0)
            )
            gpdf_env.append(static_gpdf)
            gpdf_set.append(static_gpdf)
            offset.append(self.obs_offset)
            xc_ref.append([0, 0])

        pts_human_filter = np.ones(len(self.other_obs_pt_contour_single), dtype=bool)
        if claimed_humans:
            for hidx in claimed_humans:
                pts_human_filter &= np.linalg.norm(self.other_obs_pt_contour_single - human_centers[hidx], axis=1) > self._human_strip_radius
        self.other_obs_pt_contour_single = self.other_obs_pt_contour_single[pts_human_filter]

        return gpdf_set, gpdf_env, xc_ref, offset, np.asarray(xc_inupdate), np.asarray(dxc_inupdate), num_dyn_circle_inupdate

    def _points_to_grid_indices(self, points):
        if points is None:
            return np.empty((0, 2), dtype=np.int64)

        pts = np.asarray(points)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.int64)

        x = pts[:, 0]
        y = pts[:, 1]

        i = np.floor((x - self.x_min) * self._inv_res).astype(np.int64, copy=False)
        j = np.floor((y - self.y_min) * self._inv_res).astype(np.int64, copy=False)

        np.clip(i, 0, self.x_cells - 1, out=i)
        np.clip(j, 0, self.y_cells - 1, out=j)

        return np.stack((i, j), axis=1)

    def _grid_indices_to_points_np(self, i, j):
        i = np.asarray(i, dtype = np.float32)
        j = np.asarray(j, dtype = np.float32)
        x = self.x_min + (i + 0.5) * self.resolution
        y = self.y_min + (j + 0.5) * self.resolution

        pts = np.empty((i.size, 2), dtype=np.float32)
        pts[:, 0] = x
        pts[:, 1] = y
        return pts

    def _cluster_and_contour(self, occupied_points: np.ndarray) -> None:
        if occupied_points is None or len(occupied_points) == 0:
            self.other_obs_pt_contours = []
            self.other_obs_pt_contour_single = np.empty((0, 2))
            return

        keep_indices = self._points_to_grid_indices(occupied_points)
        labels = self.db_cluster.fit_predict(keep_indices)
        self.other_obs_pt_contours = []

        for label in set(labels):
            if label == -1:
                continue
            cluster_indices = keep_indices[labels == label]
            cluster_mask = np.zeros((self.x_cells, self.y_cells), dtype=bool)
            cluster_mask[cluster_indices[:, 0], cluster_indices[:, 1]] = True

            contours = measure.find_contours(cluster_mask.astype(float), level=0.5)
            if len(contours) == 0:
                cluster_xy = self._grid_indices_to_points_np(cluster_indices[:, 0], cluster_indices[:, 1])
            else:
                contour_int = np.rint(np.vstack(contours)).astype(int)
                np.clip(contour_int[:, 0], 0, self.x_cells - 1, out=contour_int[:, 0])
                np.clip(contour_int[:, 1], 0, self.y_cells - 1, out=contour_int[:, 1])
                cluster_xy = self._grid_indices_to_points_np(contour_int[:, 0], contour_int[:, 1])

            self.other_obs_pt_contours.append(cluster_xy)

        if len(self.other_obs_pt_contours) == 0:
            self.other_obs_pt_contour_single = np.empty((0, 2))
            return

        self.other_obs_pt_contour_single = np.vstack(self.other_obs_pt_contours)

        contour_list = []
        centroids = []
        for cluster_pts in self.other_obs_pt_contours:
            if len(cluster_pts) == 0:
                continue
            clean_pts = Env_Approx._points_preprocess(cluster_pts, self.other_num_samples)
            centroids.append(clean_pts.astype(float).mean(axis=0))
            contour_list.append(clean_pts)
        self.other_obs_pt_contours = contour_list
        self.other_obs_centroids = centroids


    def _collect_centroid_vel(self):
        self.lidar_track_human_center.clear()
        self.lidar_track_human_vel.clear()
        for dynamic_centroid in self.human_centroids:
            self.lidar_track_human_center.append(np.array(dynamic_centroid["center"]))
            self.lidar_track_human_vel.append(np.array(dynamic_centroid["velocity"]))
