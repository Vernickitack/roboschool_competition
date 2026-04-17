#!/usr/bin/env python3
"""
ROS 2 competition controller — autonomous navigation with SLAM + YOLO.

Ported from main_controller_v1.35.py to ROS2 format.
Runs as a standalone ROS 2 node subscribing to sensor topics from bridge_node
and publishing velocity commands + detected objects.
"""
import sys
import math
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist, TwistStamped
    from sensor_msgs.msg import Image, Imu, JointState
    from std_msgs.msg import Int32, String
    _HAS_ROS2 = True
except ImportError:
    _HAS_ROS2 = False
    Node = object


_NAMESPACED = "/aliengo"

_ROS2_TOPICS = {
    "cmd_vel": "/cmd_vel",
    "base_velocity": f"{_NAMESPACED}/base_velocity",
    "camera_color": f"{_NAMESPACED}/camera/color/image_raw",
    "camera_depth": f"{_NAMESPACED}/camera/depth/image_raw",
    "joint_states": f"{_NAMESPACED}/joint_states",
    "imu": f"{_NAMESPACED}/imu",
    "object_sequence": "competition/object_sequence",
    "detected_object": "competition/detected_object",
}


# =============================================================================
# Classes from main_controller_v1.35.py
# =============================================================================

class InputHandler:
    def __init__(self, control_dt: float):
        self.default_dt = float(control_dt)
        self._pose = np.zeros(3, dtype=np.float32)
        width, height = 848, 480
        fov_h = math.radians(86.0)
        fx = (width / 2.0) / math.tan(fov_h / 2.0)
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        self.intrinsics = {
            "width": width,
            "height": height,
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
        }

    @property
    def pose(self) -> tuple[float, float, float]:
        return float(self._pose[0]), float(self._pose[1]), float(self._pose[2])

    def _integrate_pose(self, vx: float, vy: float, wz: float, dt: float) -> None:
        x, y, yaw = float(self._pose[0]), float(self._pose[1]), float(self._pose[2])
        dx = (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        dy = (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        dyaw = wz * dt
        x += dx
        y += dy
        yaw += dyaw
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        self._pose[:] = (x, y, yaw)

    def get_frame(self, state: Any, camera_state: Any) -> dict:
        dt = float(getattr(state, "dt", None) if state else 0) or self.default_dt
        try:
            vx = float(state.vx)
            vy = float(state.vy)
            wz = float(state.wz)
        except Exception:
            vx = vy = wz = 0.0
        self._integrate_pose(vx, vy, wz, dt)
        
        rgb = getattr(camera_state, "rgb", None)
        depth = getattr(camera_state, "depth", None)
        
        return {
            "rgb": rgb,
            "depth": depth,
            "pose": self.pose,
            "timestamp": float(getattr(state, "sim_time_s", 0.0)),
            "intrinsics": self.intrinsics,
        }


def detect_markers(
    rgb: np.ndarray | None,
    depth: np.ndarray | None,
    intrinsics: dict | None = None,
) -> list[tuple[int, tuple[float, float, float], float]]:
    """Object detector using YOLO."""
    if rgb is None or depth is None or intrinsics is None:
        return []
    
    fx = intrinsics.get("fx")
    fy = intrinsics.get("fy")
    cx = intrinsics.get("cx")
    cy = intrinsics.get("cy")
    if fx is None or fy is None or cx is None or cy is None:
        return []

    global _yolo_model
    model_loaded = "_yolo_model" in globals() and _yolo_model is not None

    if not model_loaded:
        try:
            from ultralytics import YOLO
            script_dir = Path(__file__).resolve().parent
            possible_paths = [
                script_dir / "best.pt",
                script_dir.parent / "best.pt",
            ]
            for path in possible_paths:
                if path.exists():
                    _yolo_model = YOLO(str(path))
                    model_loaded = True
                    break
        except Exception:
            pass

    if not model_loaded:
        return []

    try:
        results = _yolo_model.predict(source=rgb, verbose=False, device='cpu')
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            return []

        detections = []
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.array(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else np.array(boxes.conf)
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else np.array(boxes.cls)

        for box, conf, cls_id in zip(xyxy, confs, clss):
            conf = float(conf)
            if conf < 0.5:
                continue
            x1, y1, x2, y2 = box[:4]
            u = int((x1 + x2) / 2.0)
            v = int((y1 + y2) / 2.0)
            if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
                continue
            d = float(depth[v, u])
            if not (d > 0.0 and math.isfinite(d)):
                continue
            du = (u - cx)
            dv = (v - cy)
            x_local = d
            y_local = (du / fx) * d
            z_local = (dv / fy) * d
            detections.append((int(cls_id), (x_local, y_local, z_local), conf))
        return detections
    except Exception:
        return []


class ScenePerception:
    def __init__(self, sampling: int = 8):
        self.sampling = max(int(sampling), 1)

    def _compute_local_points(
        self,
        depth: np.ndarray | None,
        intrinsics: dict,
        pose: tuple[float, float, float],
    ) -> list[tuple[float, float, float]]:
        if depth is None:
            return []
        fx = intrinsics.get("fx")
        cx = intrinsics.get("cx")
        if fx is None or fx == 0.0 or cx is None:
            return []
        H, W = depth.shape
        sampling = self.sampling
        points = []
        for v in range(0, H, sampling):
            row = depth[v]
            for u in range(0, W, sampling):
                d = float(row[u])
                if not (d > 0.0 and math.isfinite(d)):
                    continue
                du = (u - cx)
                y_local = (du / fx) * d
                x_local = d
                points.append((x_local, y_local, d))
        return points

    def process(self, input_data: dict) -> dict:
        rgb = input_data.get("rgb")
        depth = input_data.get("depth")
        intrinsics = input_data.get("intrinsics", {})
        pose = input_data.get("pose")
        
        markers = detect_markers(rgb, depth, intrinsics)
        rays = self._compute_local_points(depth, intrinsics, pose)
        
        return {
            "rays": rays,
            "markers": markers,
            "pose": pose,
        }


class OccupancyGridMap:
    def __init__(
        self,
        *,
        resolution: float = 0.2,
        log_odds_free: float = -0.7,
        log_odds_occ: float = 2.0,
        log_odds_min: float = -5.0,
        log_odds_max: float = 5.0,
    ) -> None:
        self.resolution = float(resolution)
        self.lo_free = float(log_odds_free)
        self.lo_occ = float(log_odds_occ)
        self.lo_min = float(log_odds_min)
        self.lo_max = float(log_odds_max)
        self.log_odds: dict[tuple[int, int], float] = {}
        self.visited: set[tuple[int, int]] = set()

    def _coord_to_index(self, x: float, y: float) -> tuple[int, int]:
        ix = int(math.floor(x / self.resolution))
        iy = int(math.floor(y / self.resolution))
        return ix, iy

    def _bresenham(self, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        cells = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return cells

    def update(self, rays: list[tuple[float, float, float]], pose: tuple[float, float, float]) -> None:
        robot_x, robot_y, robot_yaw = pose
        start_ix, start_iy = self._coord_to_index(robot_x, robot_y)
        self.visited.add((start_ix, start_iy))
        
        for (x_local, y_local, _r) in rays:
            cos_yaw = math.cos(robot_yaw)
            sin_yaw = math.sin(robot_yaw)
            world_x = robot_x + cos_yaw * x_local - sin_yaw * y_local
            world_y = robot_y + sin_yaw * x_local + cos_yaw * y_local
            end_ix, end_iy = self._coord_to_index(world_x, world_y)
            cells = self._bresenham(start_ix, start_iy, end_ix, end_iy)
            if not cells:
                continue
            for cell in cells[:-1]:
                lo = self.log_odds.get(cell, 0.0) + self.lo_free
                lo = max(self.lo_min, min(self.lo_max, lo))
                self.log_odds[cell] = lo
            final_cell = cells[-1]
            lo_final = self.log_odds.get(final_cell, 0.0) + self.lo_occ
            lo_final = max(self.lo_min, min(self.lo_max, lo_final))
            self.log_odds[final_cell] = lo_final

    def get_probability(self, ix: int, iy: int) -> float:
        lo = self.log_odds.get((ix, iy), 0.0)
        return 1.0 / (1.0 + math.exp(-lo))

    def classify(self, ix: int, iy: int) -> str:
        p = self.get_probability(ix, iy)
        if p > 0.65:
            return "occupied"
        if p < 0.35:
            return "free"
        return "unknown"


class ObjectMemoryEntry:
    def __init__(self, object_id: int, position: tuple[float, float], status: str = "discovered"):
        self.id: int = int(object_id)
        self.position: tuple[float, float] = (float(position[0]), float(position[1]))
        self.status: str = str(status)
        self._obs_count: int = 1

    def update_position(self, new_pos: tuple[float, float]) -> None:
        x_old, y_old = self.position
        x_new, y_new = new_pos
        n = self._obs_count
        x_avg = (x_old * n + x_new) / (n + 1)
        y_avg = (y_old * n + y_new) / (n + 1)
        self.position = (x_avg, y_avg)
        self._obs_count = n + 1


class ObjectMemory:
    def __init__(self) -> None:
        self.entries: dict[int, ObjectMemoryEntry] = {}
        self.active_object_id: int | None = None

    def update_with_detections(self, detections: list, pose: tuple[float, float, float]) -> None:
        robot_x, robot_y, robot_yaw = pose
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        
        for det in detections:
            obj_id, (x_l, y_l, z_l), conf = det
            world_x = robot_x + cos_yaw * float(x_l) - sin_yaw * float(y_l)
            world_y = robot_y + sin_yaw * float(x_l) + cos_yaw * float(y_l)
            
            if obj_id not in self.entries:
                self.entries[obj_id] = ObjectMemoryEntry(obj_id, (world_x, world_y), status="discovered")
            else:
                self.entries[obj_id].update_position((world_x, world_y))

    def set_active(self, obj_id: int) -> None:
        if obj_id not in self.entries:
            return
        if self.active_object_id is not None and self.active_object_id in self.entries:
            prev_entry = self.entries[self.active_object_id]
            if prev_entry.status == "active":
                prev_entry.status = "discovered"
        entry = self.entries[obj_id]
        if entry.status != "visited":
            entry.status = "active"
            self.active_object_id = obj_id

    def mark_visited(self, obj_id: int) -> None:
        entry = self.entries.get(obj_id)
        if entry is None:
            return
        entry.status = "visited"
        if self.active_object_id == obj_id:
            self.active_object_id = None

    def get_active_object(self) -> ObjectMemoryEntry | None:
        if self.active_object_id is None:
            return None
        return self.entries.get(self.active_object_id)


class MissionLogic:
    def __init__(
        self,
        object_queue: list[int],
        occupancy_map: OccupancyGridMap,
        object_memory: ObjectMemory,
        *,
        exploration_speed: float = 0.3,
        target_speed: float = 0.5,
        arrival_threshold: float = 0.5,
    ) -> None:
        self.object_queue = list(object_queue)
        self.map = occupancy_map
        self.memory = object_memory
        self.state: str = "explore"
        self.current_target_id: int | None = None
        self.exploration_speed = float(exploration_speed)
        self.target_speed = float(target_speed)
        self.arrival_threshold = float(arrival_threshold)

    def _choose_next_target(self) -> None:
        self.current_target_id = None
        self.state = "explore"

        for obj_id in self.object_queue:
            entry = self.memory.entries.get(obj_id)
            if entry is not None and entry.status == "visited":
                continue
            if entry is None:
                break
            if entry.status == "discovered":
                self.memory.set_active(obj_id)
                self.current_target_id = obj_id
                self.state = "go_to_target"
                break
            if entry.status == "active":
                self.current_target_id = obj_id
                self.state = "go_to_target"
                break
            break

    def update(self, pose: tuple[float, float, float]) -> None:
        active_entry = self.memory.get_active_object()
        if active_entry is not None and active_entry.status != "visited":
            self.current_target_id = active_entry.id
            self.state = "go_to_target"
            return
        self._choose_next_target()

    def compute_velocity(self, pose: tuple[float, float, float], sim_time: float = 0.0) -> tuple[float, float, float]:
        if self.state == "go_to_target" and self.current_target_id is not None:
            entry = self.memory.entries.get(self.current_target_id)
            if entry is None:
                return 0.0, 0.0, 0.0
            dx_world = entry.position[0] - pose[0]
            dy_world = entry.position[1] - pose[1]
            angle_to_goal = math.atan2(dy_world, dx_world)
            yaw_error = angle_to_goal - pose[2]
            yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
            distance = math.hypot(dx_world, dy_world)
            v_mag = min(self.target_speed, distance)
            vx = v_mag * math.cos(yaw_error)
            vy = v_mag * math.sin(yaw_error)
            vw = yaw_error
            vx = max(min(vx, 3.0), -3.0)
            vy = max(min(vy, 1.5), -1.5)
            vw = max(min(vw, 4.0), -4.0)
            return vx, vy, vw
        
        vx = self.exploration_speed
        vy = 0.0
        vw = 0.3 * math.sin(sim_time * 0.5)
        vx = max(min(vx, 3.0), -3.0)
        vy = max(min(vy, 1.5), -1.5)
        vw = max(min(vw, 4.0), -4.0)
        return vx, vy, vw


class AStarPlanner:
    def __init__(self, occupancy_map: OccupancyGridMap, *, margin: int = 20, diagonal: bool = True) -> None:
        self.map = occupancy_map
        self.margin = int(margin)
        self.diagonal = bool(diagonal)

    def _neighbors(self, cell: tuple[int, int], bounds: tuple[int, int, int, int]) -> list:
        (ix, iy) = cell
        neighbors = []
        directions = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
        ] if self.diagonal else [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        ]
        xmin, xmax, ymin, ymax = bounds
        for dx, dy, cost in directions:
            nx, ny = ix + dx, iy + dy
            if nx < xmin or nx > xmax or ny < ymin or ny > ymax:
                continue
            if self.map.classify(nx, ny) == "occupied":
                continue
            neighbors.append(((nx, ny), cost))
        return neighbors

    def _heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))

    def plan(self, start: tuple[int, int], goal: tuple[int, int]) -> list | None:
        if start == goal:
            return [start]
        xmin = min(start[0], goal[0]) - self.margin
        xmax = max(start[0], goal[0]) + self.margin
        ymin = min(start[1], goal[1]) - self.margin
        ymax = max(start[1], goal[1]) + self.margin
        
        import heapq
        open_heap = []
        heapq.heappush(open_heap, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self._heuristic(start, goal)}
        closed_set = set()
        
        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            closed_set.add(current)
            for neighbor, cost in self._neighbors(current, (xmin, xmax, ymin, ymax)):
                if neighbor in closed_set:
                    continue
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score_neighbor = tentative_g + self._heuristic(neighbor, goal)
                    f_score[neighbor] = f_score_neighbor
                    heapq.heappush(open_heap, (f_score_neighbor, neighbor))
        return None

    def indices_to_world(self, path: list) -> list:
        resolution = self.map.resolution
        return [((ix + 0.5) * resolution, (iy + 0.5) * resolution) for (ix, iy) in path]


class PurePursuitController:
    def __init__(self, lookahead: float = 0.8, max_speed: float = 0.6) -> None:
        self.lookahead = float(lookahead)
        self.max_speed = float(max_speed)
        self._target_index = 0

    def reset(self) -> None:
        self._target_index = 0

    def _find_target_index(self, path: list, pose: tuple) -> int:
        rx, ry, _ = pose
        for i in range(self._target_index, len(path)):
            px, py = path[i]
            dist = math.hypot(px - rx, py - ry)
            if dist >= self.lookahead:
                return i
        return len(path) - 1

    def compute_command(self, path: list, pose: tuple) -> tuple[float, float, float]:
        if not path:
            return 0.0, 0.0, 0.0
        self._target_index = self._find_target_index(path, pose)
        target = path[self._target_index]
        dx_world = target[0] - pose[0]
        dy_world = target[1] - pose[1]
        distance = math.hypot(dx_world, dy_world)
        if distance < 1e-3:
            return 0.0, 0.0, 0.0
        angle_to_goal = math.atan2(dy_world, dx_world)
        yaw_error = angle_to_goal - pose[2]
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
        v_mag = min(self.max_speed, distance)
        vx = v_mag * math.cos(yaw_error)
        vy = v_mag * math.sin(yaw_error)
        vw = yaw_error
        vx = max(min(vx, 3.0), -3.0)
        vy = max(min(vy, 1.5), -1.5)
        if vw > 4.0:
            vw = 4.0
        elif vw < -4.0:
            vw = -4.0
        return vx, vy, vw


class NavigationPlanner:
    def __init__(
        self,
        occupancy_map: OccupancyGridMap,
        *,
        lookahead: float = 0.8,
        max_speed: float = 0.6,
        margin: int = 20,
    ) -> None:
        self.map = occupancy_map
        self.planner = AStarPlanner(occupancy_map, margin=margin)
        self.controller = PurePursuitController(lookahead=lookahead, max_speed=max_speed)
        self.current_goal: tuple | None = None
        self.path_world = []
        self._last_start = None

    def _needs_replan(self, start_cell: tuple, goal_cell: tuple) -> bool:
        if not self.path_world:
            return True
        if self.current_goal != goal_cell:
            return True
        if self._last_start is None or start_cell != self._last_start:
            return True
        return False

    def compute_command(self, pose: tuple, target_pos: tuple) -> tuple[float, float, float]:
        goal_ix, goal_iy = self.map._coord_to_index(target_pos[0], target_pos[1])
        start_ix, start_iy = self.map._coord_to_index(pose[0], pose[1])
        
        if self._needs_replan((start_ix, start_iy), (goal_ix, goal_iy)):
            path_indices = self.planner.plan((start_ix, start_iy), (goal_ix, goal_iy))
            if path_indices is None:
                self.path_world = []
                self.current_goal = (goal_ix, goal_iy)
                self.controller.reset()
            else:
                self.path_world = self.planner.indices_to_world(path_indices)
                self.current_goal = (goal_ix, goal_iy)
                self._last_start = (start_ix, start_iy)
                self.controller.reset()
        
        if not self.path_world:
            dx_world = target_pos[0] - pose[0]
            dy_world = target_pos[1] - pose[1]
            angle_to_goal = math.atan2(dy_world, dx_world)
            yaw_error = angle_to_goal - pose[2]
            yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
            distance = math.hypot(dx_world, dy_world)
            v_mag = min(self.controller.max_speed, distance)
            vx = v_mag * math.cos(yaw_error)
            vy = v_mag * math.sin(yaw_error)
            vw = yaw_error
            vx = max(min(vx, 3.0), -3.0)
            vy = max(min(vy, 1.5), -1.5)
            if vw > 2.0:
                vw = 2.0
            elif vw < -2.0:
                vw = -2.0
            return vx, vy, vw
        
        return self.controller.compute_command(self.path_world, pose)


class CloseObstacleRecovery:
    def __init__(
        self,
        *,
        trigger_distance: float = 0.50,
        reverse_speed: float = -0.35,
        turn_speed: float = 1.2,
        backup_distance_m: float = 1.0,
        turn_angle_rad: float = math.pi / 2.0,
        control_dt: float = 0.02,
        cooldown_steps: int = 10,
    ) -> None:
        self.trigger_distance = float(trigger_distance)
        self.reverse_speed = float(reverse_speed)
        self.turn_speed = float(turn_speed)
        self.backup_distance_m = float(backup_distance_m)
        self.turn_angle_rad = float(turn_angle_rad)
        self.control_dt = float(max(control_dt, 1e-6))
        self.cooldown_steps = int(cooldown_steps)
        self.mode = "idle"
        self.steps_left = 0
        self.cooldown_left = 0
        self.turn_sign = 1.0
        self.backup_steps = max(1, int(math.ceil(self.backup_distance_m / (abs(self.reverse_speed) * self.control_dt))))
        self.turn_steps = max(1, int(math.ceil(self.turn_angle_rad / (abs(self.turn_speed) * self.control_dt))))

    def _choose_turn_sign(self, depth: np.ndarray) -> float:
        h, w = depth.shape
        left = depth[:, : w // 3]
        right = depth[:, 2 * w // 3:]
        left_vals = left[np.isfinite(left) & (left > 0.0)]
        right_vals = right[np.isfinite(right) & (right > 0.0)]
        left_mean = float(np.mean(left_vals)) if left_vals.size else 0.0
        right_mean = float(np.mean(right_vals)) if right_vals.size else 0.0
        return 1.0 if left_mean >= right_mean else -1.0

    def _front_min_distance(self, depth: np.ndarray) -> float:
        h, w = depth.shape
        col_margin = max(w // 6, 1)
        row_start = h // 3
        row_end = 2 * h // 3
        center = depth[row_start:row_end, col_margin : w - col_margin]
        vals = center[np.isfinite(center) & (center > 0.0)]
        if vals.size == 0:
            return float("inf")
        return float(np.min(vals))

    def compute_override(self, depth: np.ndarray | None) -> tuple[bool, tuple, str]:
        if depth is None or not isinstance(depth, np.ndarray) or depth.ndim != 2:
            if self.mode == "cooldown":
                self.cooldown_left -= 1
                if self.cooldown_left <= 0:
                    self.mode = "idle"
            return False, (0.0, 0.0, 0.0), ""

        front_min = self._front_min_distance(depth)

        if self.mode == "cooldown":
            self.cooldown_left -= 1
            if self.cooldown_left <= 0:
                self.mode = "idle"
            return False, (0.0, 0.0, 0.0), ""

        if self.mode == "idle":
            if front_min < self.trigger_distance:
                self.mode = "backup"
                self.steps_left = self.backup_steps
                self.get_logger().info(f"Obstacle ahead ({front_min:.2f}m) - backing up")
            return False, (0.0, 0.0, 0.0), ""

        if self.mode == "backup":
            vx, vy, vw = self.reverse_speed, 0.0, 0.0
            self.steps_left -= 1
            if self.steps_left <= 0:
                self.mode = "turn"
                self.turn_sign = self._choose_turn_sign(depth)
                self.steps_left = self.turn_steps
            return True, (vx, vy, vw), "Backing up"

        if self.mode == "turn":
            vx, vy, vw = 0.0, 0.0, self.turn_sign * self.turn_speed
            self.steps_left -= 1
            if self.steps_left <= 0:
                self.mode = "cooldown"
                self.cooldown_left = self.cooldown_steps
            return True, (vx, vy, vw), f"Turning {'left' if self.turn_sign > 0 else 'right'}"

        return False, (0.0, 0.0, 0.0), ""

    def get_logger(self):
        return _SimpleLogger()


class _SimpleLogger:
    def info(self, msg): print(f"[INFO] {msg}")
    def warn(self, msg): print(f"[WARN] {msg}")
    def error(self, msg): print(f"[ERROR] {msg}")


# =============================================================================
# ROS2 Navigation Controller
# =============================================================================

if _HAS_ROS2:
    class NavigationController(Node):
        def __init__(self):
            super().__init__("navigation_controller")
            self.get_logger().info("Starting ROS2 Navigation Controller...")

            self.cmd_pub = self.create_publisher(
                Twist, _ROS2_TOPICS["cmd_vel"], 10
            )
            self.detected_object_pub = self.create_publisher(
                Int32, _ROS2_TOPICS["detected_object"], 10
            )
            self.vel_sub = self.create_subscription(
                TwistStamped, _ROS2_TOPICS["base_velocity"], self._vel_cb, 10
            )
            self.joint_sub = self.create_subscription(
                JointState, _ROS2_TOPICS["joint_states"], self._joint_cb, 10
            )
            self.imu_sub = self.create_subscription(
                Imu, _ROS2_TOPICS["imu"], self._imu_cb, 10
            )
            self.rgb_sub = self.create_subscription(
                Image, _ROS2_TOPICS["camera_color"], self._rgb_cb, 10
            )
            self.depth_sub = self.create_subscription(
                Image, _ROS2_TOPICS["camera_depth"], self._depth_cb, 10
            )
            self.seq_sub = self.create_subscription(
                String, _ROS2_TOPICS["object_sequence"], self._seq_cb, 10
            )

            self.latest_vx: float = 0.0
            self.latest_vy: float = 0.0
            self.latest_wz: float = 0.0
            self.vel_stamp: Optional[float] = None
            self.latest_rgb: Optional[np.ndarray] = None
            self.latest_depth: Optional[np.ndarray] = None
            self.latest_joint_state: Dict = {"names": [], "position": [], "velocity": [], "stamp_sec": None}
            self.latest_imu: Dict = {"wx": 0.0, "wy": 0.0, "wz": 0.0, "stamp_sec": None}

            self.object_queue: List[int] = []
            self._seq_received: bool = False
            self.control_dt: float = 0.02
            self.step_count: int = 0
            self.sim_time: float = 0.0

            self._DEPTH_W, self._DEPTH_H = 848, 480
            self._DEPTH_FX = self._DEPTH_W / (2.0 * math.tan(math.radians(43.0)))
            self._DEPTH_FY = self._DEPTH_FX
            self._DEPTH_CX = self._DEPTH_W / 2.0
            self._DEPTH_CY = self._DEPTH_H / 2.0

            self.intrinsics = {
                "width": self._DEPTH_W,
                "height": self._DEPTH_H,
                "fx": self._DEPTH_FX,
                "fy": self._DEPTH_FY,
                "cx": self._DEPTH_CX,
                "cy": self._DEPTH_CY,
            }

            self.input_handler = InputHandler(self.control_dt)
            self.scene_perception = ScenePerception()
            self.occupancy_map = OccupancyGridMap()
            self.object_memory = ObjectMemory()
            self.navigation_planner = NavigationPlanner(
                self.occupancy_map,
                lookahead=0.8,
                max_speed=1.2,
                margin=20,
            )
            self.mission_logic = MissionLogic(
                object_queue=[],
                occupancy_map=self.occupancy_map,
                object_memory=self.object_memory,
                exploration_speed=1.0,
                target_speed=1.0,
                arrival_threshold=1.5,
            )
            self.wall_recovery = CloseObstacleRecovery(
                trigger_distance=0.50,
                reverse_speed=-0.35,
                turn_speed=1.2,
                backup_distance_m=1.0,
                turn_angle_rad=math.pi / 2.0,
                control_dt=self.control_dt,
                cooldown_steps=10,
            )
            self.wall_recovery.get_logger = lambda: self.get_logger()

            self.logged_object_ids: set = set()
            self.announced_object_ids: set = set()
            self.prev_state: str = "explore"
            self.prev_target_id: Optional[int] = None
            self.after_visit_countdown: int = 0
            self.spin_mode: bool = False
            self.current_spin_speed: float = 0.0

            self.max_vx_abs: float = 3.0
            self.max_vy_abs: float = 1.5
            self.max_wz_abs: float = 4.0
            self.max_planar_speed: float = 1.2

            self.confirm_target_id: Optional[int] = None
            self.confirmation_steps: int = 0
            self.confirm_required_steps: int = int(round(2.0 / max(self.control_dt, 1e-6)))
            # Confirmation zone around the target.
            self.confirm_distance_threshold: float = 1.0

            self.create_timer(self.control_dt, self._main_loop)

            self.get_logger().info(
                "Navigation controller started. Waiting for sensor data + object_sequence..."
            )

        def _vel_cb(self, msg: TwistStamped) -> None:
            self.latest_vx = float(msg.twist.linear.x)
            self.latest_vy = float(msg.twist.linear.y)
            self.latest_wz = float(msg.twist.angular.z)
            self.vel_stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

        def _joint_cb(self, msg: JointState) -> None:
            self.latest_joint_state = {
                "names": list(msg.name),
                "position": list(msg.position),
                "velocity": list(msg.velocity),
                "stamp_sec": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
            }

        def _imu_cb(self, msg: Imu) -> None:
            self.latest_imu = {
                "wx": float(msg.angular_velocity.x),
                "wy": float(msg.angular_velocity.y),
                "wz": float(msg.angular_velocity.z),
                "stamp_sec": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
            }

        def _rgb_cb(self, msg: Image) -> None:
            try:
                self.latest_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3)).copy()
            except ValueError:
                self.get_logger().warning("Failed to reshape RGB image.")

        def _depth_cb(self, msg: Image) -> None:
            try:
                self.latest_depth = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width)).copy()
            except ValueError:
                self.get_logger().warning("Failed to reshape Depth image.")

        def _seq_cb(self, msg: String) -> None:
            if self._seq_received:
                return
            try:
                raw = json.loads(msg.data)
                if isinstance(raw, list):
                    self.object_queue = [int(x) for x in raw]
                    self._seq_received = True
                    self.mission_logic.object_queue = self.object_queue
                    self.get_logger().info(f"Received object queue: {self.object_queue}")
            except Exception as exc:
                self.get_logger().error(f"Failed to parse object_sequence: {exc}")

        def _normalize_cmd(self, vx: float, vy: float, vw: float) -> tuple[float, float, float]:
            vx = max(min(float(vx), self.max_vx_abs), -self.max_vx_abs)
            vy = max(min(float(vy), self.max_vy_abs), -self.max_vy_abs)
            vw = max(min(float(vw), self.max_wz_abs), -self.max_wz_abs)

            planar_speed = math.hypot(vx, vy)
            if planar_speed > self.max_planar_speed and planar_speed > 1e-9:
                scale = self.max_planar_speed / planar_speed
                vx *= scale
                vy *= scale

            return vx, vy, vw

        def publish_detected_object(self, object_id: int) -> None:
            self._publish_detected(int(object_id))

        def _publish_cmd(self, vx: float, vy: float, vw: float) -> None:
            vx, vy, vw = self._normalize_cmd(vx, vy, vw)
            cmd = Twist()
            cmd.linear.x = float(vx)
            cmd.linear.y = float(vy)
            cmd.linear.z = 0.0
            cmd.angular.x = 0.0
            cmd.angular.y = 0.0
            cmd.angular.z = float(vw)
            self.cmd_pub.publish(cmd)

        def _publish_detected(self, obj_id: int) -> None:
            msg = Int32()
            msg.data = int(obj_id)
            self.detected_object_pub.publish(msg)

        def _main_loop(self) -> None:
            if not self._seq_received:
                self.step_count += 1
                return

            if self.latest_rgb is None or self.latest_depth is None:
                self.step_count += 1
                return

            self.sim_time += self.control_dt

            class FakeState:
                def __init__(self, vx, vy, wz, dt, sim_time):
                    self.vx = vx
                    self.vy = vy
                    self.wz = wz
                    self.dt = dt
                    self.sim_time_s = sim_time

            class FakeCameraState:
                def __init__(self, rgb, depth):
                    self.rgb = rgb
                    self.depth = depth

            fake_state = FakeState(
                self.latest_vx, self.latest_vy, self.latest_wz, self.control_dt, self.sim_time
            )
            camera_state = FakeCameraState(self.latest_rgb, self.latest_depth)

            input_data = self.input_handler.get_frame(fake_state, camera_state)
            scene_data = self.scene_perception.process(input_data)

            self.occupancy_map.update(scene_data.get("rays", []), scene_data.get("pose"))
            self.object_memory.update_with_detections(scene_data.get("markers", []), scene_data.get("pose"))

            markers = scene_data.get("markers", [])

            pending_announcements = []
            for (det_id, _loc, _conf) in markers:
                det_id = int(det_id)
                if det_id in self.object_queue and det_id not in self.announced_object_ids:
                    pending_announcements.append(det_id)
                    self.announced_object_ids.add(det_id)
            for det_id in pending_announcements:
                self.get_logger().info(f"Object detected: id={det_id}")

            def log_found_object(object_id: int) -> None:
                self.get_logger().info(f"Object fixed: id={int(object_id)}")
                self.publish_detected_object(int(object_id))

            def get_found_object_id(current_state, current_camera_data, current_object_queue):
                _ = current_state
                _ = current_camera_data
                _ = current_object_queue
                if self.confirm_target_id is None:
                    return None
                if self.confirm_target_id in self.logged_object_ids:
                    return None
                if self.confirmation_steps < self.confirm_required_steps:
                    return None
                return int(self.confirm_target_id)

            self.mission_logic.update(scene_data.get("pose"))
            current_state = self.mission_logic.state
            current_target = self.mission_logic.current_target_id

            visited_count = sum(
                1 for e in self.object_memory.entries.values() if e.status == "visited"
            )
            total_targets = len(self.object_queue)

            pose = scene_data.get("pose")
            target_visible = False
            target_distance = float("inf")
            active_entry = None
            if current_target is not None:
                active_entry = self.object_memory.entries.get(current_target)
                if active_entry is not None:
                    dx = active_entry.position[0] - pose[0]
                    dy = active_entry.position[1] - pose[1]
                    target_distance = math.hypot(dx, dy)
                target_visible = any(int(det_id) == int(current_target) for (det_id, _loc, _conf) in markers)

            if current_target != self.confirm_target_id:
                self.confirm_target_id = None
                self.confirmation_steps = 0

            confirmation_in_progress = (
                self.confirm_target_id is not None
                and self.confirm_target_id == current_target
                and self.confirmation_steps > 0
            )

            should_confirm = (
                current_state == "go_to_target"
                and current_target is not None
                and active_entry is not None
                and target_distance <= self.confirm_distance_threshold
                and (target_visible or confirmation_in_progress)
            )

            if should_confirm:
                if self.confirm_target_id != current_target:
                    self.confirm_target_id = int(current_target)
                    self.confirmation_steps = 0
                self.confirmation_steps += 1
            else:
                self.confirm_target_id = None
                self.confirmation_steps = 0

            detected_object_id = get_found_object_id(
                fake_state,
                {"image": self.latest_rgb, "depth": self.latest_depth},
                self.object_queue,
            )
            if detected_object_id is not None:
                log_found_object(detected_object_id)
                self.logged_object_ids.add(detected_object_id)
                self.object_memory.mark_visited(detected_object_id)
                self.confirm_target_id = None
                self.confirmation_steps = 0
                current_state = self.mission_logic.state = "explore"
                current_target = self.mission_logic.current_target_id = None

                visited_count = sum(
                    1 for e in self.object_memory.entries.values() if e.status == "visited"
                )

            if total_targets > 0 and visited_count >= total_targets and not self.spin_mode:
                self.spin_mode = True
                self.current_spin_speed = 0.0

            if self.spin_mode:
                self.current_spin_speed = min(4.0, self.current_spin_speed + 0.10)
                vx, vy, vw = 0.0, 0.0, self.current_spin_speed
            elif self.confirm_target_id is not None and self.confirmation_steps > 0:
                vx, vy, vw = 0.0, 0.0, 0.0
            else:
                if current_state == "go_to_target" and current_target is not None:
                    entry = self.object_memory.entries.get(current_target)
                    if entry is not None:
                        vx, vy, vw = self.navigation_planner.compute_command(
                            pose, entry.position
                        )

                        # Slow down on the final approach so the robot can enter the
                        # confirmation radius reliably instead of overshooting it.
                        dist_to_target = math.hypot(
                            entry.position[0] - pose[0],
                            entry.position[1] - pose[1],
                        )
                        if dist_to_target <= (self.confirm_distance_threshold + 0.5):
                            linear_speed = math.hypot(vx, vy)
                            max_linear_speed = min(0.35, self.max_planar_speed)
                            if linear_speed > max_linear_speed and linear_speed > 1e-6:
                                scale = max_linear_speed / linear_speed
                                vx *= scale
                                vy *= scale
                            vw = max(min(vw, 1.0), -1.0)
                    else:
                        vx, vy, vw = 0.0, 0.0, 0.0
                else:
                    vx, vy, vw = self.mission_logic.compute_velocity(
                        pose, self.sim_time
                    )

            if self.confirm_target_id is None:
                has_override, override_cmd, override_msg = self.wall_recovery.compute_override(
                    self.latest_depth
                )
                if has_override:
                    vx, vy, vw = override_cmd

            self.prev_state = current_state
            self.prev_target_id = current_target

            self._publish_cmd(vx, vy, vw)

            self.step_count += 1


def main(args=None):
    if not _HAS_ROS2:
        print("ROS2 is not installed. Install with: pip install rclpy")
        return
    rclpy.init(args=args)
    node = NavigationController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
