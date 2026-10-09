#!/usr/bin/env python3
import numpy as np
from scipy.optimize import linear_sum_assignment

import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer

from geometry_msgs.msg import Point, Vector3
from obstacle_detector.msg import Obstacles, CircleObstacle


class ObstacleTrackListNode(Node):
    def __init__(self):
        super().__init__('obstacle_track_list_node')

        self.declare_parameter('input_topic', '/obstacles')
        self.declare_parameter('raw_input_topic', '/raw_obstacles')
        self.declare_parameter('active_output_topic', '/obstacles_tracked')
        self.declare_parameter('track_list_topic', '/obstacles_track_list')
        self.declare_parameter('match_max_distance', 0.5)
        self.declare_parameter('sync_slop', 0.1)
        self.declare_parameter('max_chances', 5)
        self.declare_parameter('merge_distance', 0.5)

        input_topic = self.get_parameter('input_topic').value
        raw_input_topic = self.get_parameter('raw_input_topic').value
        active_output_topic = self.get_parameter('active_output_topic').value
        track_list_topic = self.get_parameter('track_list_topic').value
        self.match_max_distance = float(self.get_parameter('match_max_distance').value)
        sync_slop = float(self.get_parameter('sync_slop').value)
        self.max_chances = int(self.get_parameter('max_chances').value)
        self.merge_distance = float(self.get_parameter('merge_distance').value)

        self.tracks = []

        self.obs_sub = Subscriber(self, Obstacles, input_topic)
        self.raw_obs_sub = Subscriber(self, Obstacles, raw_input_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.obs_sub, self.raw_obs_sub], queue_size=10, slop=sync_slop)
        self.sync.registerCallback(self.callback_obs)

        self.active_pub = self.create_publisher(Obstacles, active_output_topic, 1)
        self.track_list_pub = self.create_publisher(Obstacles, track_list_topic, 1)

    def callback_obs(self, msg: Obstacles, raw_msg: Obstacles):
        detections = [{
            'center': np.array([c.center.x, c.center.y], dtype=np.float64),
            'velocity': (c.velocity.x, c.velocity.y),
            'radius': c.radius,
            'true_radius': c.true_radius,
        } for c in msg.circles]
        raw_detections = [{
            'center': np.array([c.center.x, c.center.y], dtype=np.float64),
            'radius': c.radius,
            'true_radius': c.true_radius,
        } for c in raw_msg.circles]

        active_idx = [i for i, t in enumerate(self.tracks) if t['active']]

        matched_track_for_det = {}
        matched_tracks = set()
        if active_idx and detections:
            cost = np.array([
                [np.linalg.norm(self.tracks[ti]['center'] - d['center']) for d in detections]
                for ti in active_idx
            ])
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= self.match_max_distance:
                    ti = active_idx[r]
                    matched_track_for_det[c] = ti
                    matched_tracks.add(ti)

        for di, ti in matched_track_for_det.items():
            d = detections[di]
            track = self.tracks[ti]
            track['center'] = d['center']
            track['velocity'] = d['velocity']
            track['radius'] = d['radius']
            track['true_radius'] = d['true_radius']
            track['chances_left'] = self.max_chances

        unmatched_active = [ti for ti in active_idx if ti not in matched_tracks]
        rescued_tracks = set()
        if unmatched_active and raw_detections:
            raw_cost = np.array([
                [np.linalg.norm(self.tracks[ti]['center'] - d['center']) for d in raw_detections]
                for ti in unmatched_active
            ])
            row_ind, col_ind = linear_sum_assignment(raw_cost)
            for r, c in zip(row_ind, col_ind):
                if raw_cost[r, c] <= self.match_max_distance:
                    ti = unmatched_active[r]
                    d = raw_detections[c]
                    track = self.tracks[ti]
                    track['center'] = d['center']
                    track['radius'] = d['radius']
                    track['true_radius'] = d['true_radius']
                    track['chances_left'] = self.max_chances
                    rescued_tracks.add(ti)

        for ti in active_idx:
            if ti not in matched_tracks and ti not in rescued_tracks:
                track = self.tracks[ti]
                track['chances_left'] -= 1
                if track['chances_left'] <= 0:
                    track['velocity'] = (0.0, 0.0)
                    track['active'] = False

        for di, d in enumerate(detections):
            if di not in matched_track_for_det:
                self.tracks.append({
                    'center': d['center'],
                    'velocity': d['velocity'],
                    'radius': d['radius'],
                    'true_radius': d['true_radius'],
                    'active': True,
                    'chances_left': self.max_chances,
                })

        stamp = msg.header

        active_tracks = [t for t in self.tracks if t['active']]
        merged_tracks = self._merge_close_circles(active_tracks)

        active_msg = Obstacles()
        active_msg.header = stamp
        active_msg.circles = [self._to_circle(t) for t in merged_tracks]
        self.active_pub.publish(active_msg)

        track_list_msg = Obstacles()
        track_list_msg.header = stamp
        track_list_msg.circles = [self._to_circle(t) for t in self.tracks]
        self.track_list_pub.publish(track_list_msg)

    def _merge_close_circles(self, tracks):
        n = len(tracks)
        if n == 0:
            return []

        adjacency = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(tracks[i]['center'] - tracks[j]['center']) <= self.merge_distance:
                    adjacency[i].append(j)
                    adjacency[j].append(i)

        visited = [False] * n
        merged = []
        for i in range(n):
            if visited[i]:
                continue

            stack = [i]
            visited[i] = True
            component = []
            while stack:
                k = stack.pop()
                component.append(k)
                for nb in adjacency[k]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)

            if len(component) == 1:
                merged.append(tracks[component[0]])
                continue

            members = [tracks[k] for k in component]
            center = np.mean([m['center'] for m in members], axis=0)
            radius = max(
                np.linalg.norm(center - m['center']) + m['radius'] for m in members)
            true_radius = max(
                np.linalg.norm(center - m['center']) + m['true_radius'] for m in members)
            winner = max(members, key=lambda m: np.linalg.norm(m['velocity']))

            merged.append({
                'center': center,
                'velocity': winner['velocity'],
                'radius': radius,
                'true_radius': true_radius,
            })

        return merged

    @staticmethod
    def _to_circle(track):
        circle = CircleObstacle()
        circle.center = Point(x=float(track['center'][0]), y=float(track['center'][1]), z=0.0)
        circle.velocity = Vector3(x=float(track['velocity'][0]), y=float(track['velocity'][1]), z=0.0)
        circle.radius = track['radius']
        circle.true_radius = track['true_radius']
        return circle


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleTrackListNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
