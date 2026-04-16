from __future__ import annotations

import math

import numpy as np

from aliengo_competition.common.run_logger import CompetitionRunLogger
from aliengo_competition.robot_interface.base import AliengoRobotInterface
from aliengo_competition.robot_interface.types import CameraState, RobotState

class InputHandler:
    """
    Блок приёма и подготовки входных данных.

    Этот класс синхронизирует данные фронтальной RGB‑камеры и датчика глубины,
    ведёт простую интеграцию собственной позы на основе измеренных скоростей
    робота и возвращает структурированный пакет данных для дальнейшей
    обработки в блоках восприятия и картографии.

    В качестве ориентира для калибровки камеры используются значения,
    предоставленные организаторами в узле публикации CameraInfo. Если в
    будущем понадобится подключение к ROS 2, этот класс можно расширить,
    чтобы подписываться на соответствующий топик, но текущая реализация
    содержит статические параметры: ширина, высота, угол поля зрения по
    горизонтали и вычисленные из них фокусные расстояния fx/fy и центры
    проекции cx/cy.
    """

    def __init__(self, control_dt: float):
        # Интервал обновления для интегрирования позы. При каждом
        # вызове get_frame() dt может заменяться на значение state.dt,
        # если оно отличается от переданного при инициализации.
        self.default_dt = float(control_dt)
        # Собственная поза робота в локальной системе координат (x, y, yaw).
        self._pose = np.zeros(3, dtype=np.float32)
        # Настройки камеры. Эти значения соответствуют узлу
        # DepthCameraInfoPublisher: ширина 848, высота 480, угол обзора 86°.
        width = 848
        height = 480
        fov_h_deg = 86.0
        fov_h = math.radians(fov_h_deg)
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
        """Возвращает текущую оценку собственной позы (x, y, yaw)."""
        return float(self._pose[0]), float(self._pose[1]), float(self._pose[2])

    def _integrate_pose(self, vx: float, vy: float, wz: float, dt: float) -> None:
        """
        Обновляет локальную оценку позы по данным об линейных и угловых
        скоростях. Модель кинематики представляет базовую платформу как
        дифференциально управляемый механизм: линейные скорости заданы в
        локальной системе робота, потому используются стандартные
        преобразования через угол yaw.

        :param vx: линейная скорость вдоль продольной оси робота (м/с)
        :param vy: линейная скорость вдоль поперечной оси робота (м/с)
        :param wz: угловая скорость вокруг вертикальной оси (рад/с)
        :param dt: шаг интегрирования (с)
        """
        # Извлекаем текущий угол ориентации
        x, y, yaw = float(self._pose[0]), float(self._pose[1]), float(self._pose[2])
        # Прямое интегрирование. Переход от скоростей в локальной системе
        # координат к изменениям в глобальной системе: учитываем текущий yaw.
        dx = (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        dy = (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        dyaw = wz * dt
        x += dx
        y += dy
        yaw += dyaw
        # Ограничиваем yaw в диапазоне [-pi, pi] для численной стабильности
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        self._pose[:] = (x, y, yaw)

    def get_frame(self, state: RobotState, camera_state: CameraState | None) -> dict:
        """
        Формирует структурированный пакет данных из текущего состояния
        робота и показаний камеры.

        :param state: текущее состояние робота (RobotState)
        :param camera_state: состояние камеры (CameraState) или None
        :return: словарь с полями rgb, depth, pose, timestamp и intrinsics
        """
        # Определяем шаг dt для интегрирования. Если state.dt корректен,
        # используем его, иначе – значение по умолчанию.
        dt = float(state.dt) if hasattr(state, "dt") and state.dt > 0 else self.default_dt
        # Интегрируем позу по измеренным скоростям
        try:
            vx = float(state.vx)
            vy = float(state.vy)
            wz = float(state.wz)
        except Exception:
            # Если не удалось получить скорости, не обновляем позу
            vx = vy = wz = 0.0
        self._integrate_pose(vx, vy, wz, dt)
        # Извлекаем данные с камеры
        rgb = None
        depth = None
        if isinstance(camera_state, CameraState):
            rgb = camera_state.rgb
            depth = camera_state.depth
        # Возвращаем пакет данных
        return {
            "rgb": rgb,
            "depth": depth,
            "pose": self.pose,
            "timestamp": float(state.sim_time_s),
            "intrinsics": self.intrinsics,
        }

def detect_markers(
    rgb: np.ndarray | None,
    depth: np.ndarray | None,
    intrinsics: dict | None = None,
) -> list[tuple[int, tuple[float, float, float], float]]:
    """
    Детектор объектов на базе Ultralytics YOLO.

    Требует установленный пакет ``ultralytics`` и файл ``best.pt``.
    Ищет веса рядом с контроллером, в текущем рабочем каталоге и в двух
    родительских директориях контроллера. Если модель не найдена или не
    загружается, возбуждается ``RuntimeError``.

    Возвращает список детекций в формате
    ``(class_id, (x_local, y_local, z_local), confidence)``.
    """
    import os
    import math as _math
    from ultralytics import YOLO

    if rgb is None or depth is None or intrinsics is None:
        return []
    fx = intrinsics.get("fx")
    fy = intrinsics.get("fy")
    cx = intrinsics.get("cx")
    cy = intrinsics.get("cy")
    if fx is None or fy is None or cx is None or cy is None:
        return []

    global _yolo_model_ultralytics
    model_loaded = False
    if "_yolo_model_ultralytics" in globals():
        model_loaded = _yolo_model_ultralytics is not None
    else:
        _yolo_model_ultralytics = None  # type: ignore

    if not model_loaded:
        script_dir = os.path.dirname(__file__)
        possible_dirs = [script_dir, os.getcwd()]
        parent = script_dir
        for _ in range(2):
            parent = os.path.dirname(parent)
            if parent and parent not in possible_dirs:
                possible_dirs.append(parent)
        possible_paths = list(dict.fromkeys([os.path.join(d, "best.pt") for d in possible_dirs]))

        load_error = None
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    _yolo_model_ultralytics = YOLO(path)  # type: ignore
                    model_loaded = True
                    break
                except Exception as e:
                    load_error = e
                    _yolo_model_ultralytics = None  # type: ignore

        if not model_loaded or _yolo_model_ultralytics is None:
            paths_str = ", ".join(possible_paths)
            raise RuntimeError(
                f"Ultralytics YOLO model 'best.pt' not found or could not be loaded. "
                f"Searched paths: [{paths_str}]."
                + (f" Last error: {load_error}" if load_error is not None else "")
            )

    try:
        results = _yolo_model_ultralytics.predict(source=rgb, verbose=False, device='cpu')  # type: ignore
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            return []

        detections: list[tuple[int, tuple[float, float, float], float]] = []
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
            if not (d > 0.0 and _math.isfinite(d)):
                continue
            du = (u - cx)
            dv = (v - cy)
            x_local = d
            y_local = (du / fx) * d
            z_local = (dv / fy) * d
            detections.append((int(cls_id), (x_local, y_local, z_local), conf))
        return detections
    except Exception as e:
        raise RuntimeError(f"Error during Ultralytics YOLO inference: {e}")

class ScenePerception:
    """
    Блок восприятия сцены.

    Принимает структурированный пакет данных от InputHandler, выделяет
    информацию о свободном и занятом пространстве на основе карты глубин
    и вызывает детектор объектов для получения списка маркеров. Результатом
    является структура, описывающая "лучи" (окончания сегментов между
    роботом и измеренными препятствиями) и обнаруженные маркеры.

    Пока карта сетки занятости не обновляется в этом блоке — эта задача
    относится к последующему блоку картографии. Вместо этого здесь
    вычисляются 2D‑координаты концов лучей в локальной системе робота,
    которые затем могут быть использованы для обновления сетки.
    """

    def __init__(self, sampling: int = 8):
        """
        :param sampling: шаг дискретизации пикселей по горизонтали/вертикали.
            При большом разрешении камеры просмотр каждого пикселя может быть
            неэффективным; параметр sampling определяет интервал между
            рассматриваемыми пикселями (например, 8 будет означать, что
            обрабатывается каждый 8‑й пиксель по оси X и Y).
        """
        self.sampling = max(int(sampling), 1)

    def _compute_local_points(
        self,
        depth: np.ndarray | None,
        intrinsics: dict,
        pose: tuple[float, float, float],
    ) -> list[tuple[float, float, float]]:
        """
        Преобразует карту глубины в набор локальных 2D‑точек (x, y, r).

        Для каждого пикселя глубины (с шагом sampling) вычисляется
        расстояние r = depth[v, u] и угловое отклонение относительно
        центральной колонки, затем координаты (x, y) в локальной системе
        камеры: x = r, y = (u - cx)/fx * r. Мы предполагаем, что камера
        ориентирована вдоль оси x робота; вертикальный угол игнорируется.
        Возвращаем список троек (x_local, y_local, r), где r используется
        для сортировки или фильтрации при обновлении карты.

        :param depth: карта глубины (H×W) или None
        :param intrinsics: словарь с параметрами "fx", "fy", "cx", "cy",
            "width", "height"
        :param pose: текущее положение робота (x, y, yaw). В этом методе
            собственная поза не используется, поскольку возвращаются
            координаты в локальной системе; преобразование в карту будет
            выполняться в другом модуле.
        :return: список (x_local, y_local, r) для измеренных точек
        """
        if depth is None:
            return []
        fx = intrinsics.get("fx")
        cx = intrinsics.get("cx")
        # Базовые проверки, чтобы избежать деления на ноль
        if fx is None or fx == 0.0 or cx is None:
            return []
        H, W = depth.shape
        sampling = self.sampling
        points: list[tuple[float, float, float]] = []
        # Итерируем по пикселям с заданным шагом
        for v in range(0, H, sampling):
            # Извлекаем строку глубин для ускорения
            row = depth[v]
            for u in range(0, W, sampling):
                d = float(row[u])
                # Отбрасываем слишком маленькие или нечисловые значения
                if not (d > 0.0 and math.isfinite(d)):
                    continue
                # Горизонтальное смещение от центра
                du = (u - cx)
                y_local = (du / fx) * d
                x_local = d
                points.append((x_local, y_local, d))
        return points

    def process(self, input_data: dict) -> dict:
        """
        Обрабатывает входные данные от InputHandler и возвращает анализ сцены.

        :param input_data: структура, содержащая ключи rgb, depth, pose,
            timestamp, intrinsics
        :return: словарь с ключами:
            "rays": список (x_local, y_local, r) для окончания лучей (препятствия);
            "markers": список результатов детекции (см. detect_markers);
            "pose": поза робота, переданная из input_data
        """
        rgb = input_data.get("rgb")
        depth = input_data.get("depth")
        intrinsics = input_data.get("intrinsics", {})
        pose = input_data.get("pose")
        # Детектируем маркеры с использованием YOLO/fallback‑модуля. Передаем
        # параметры калибровки, чтобы восстановить локальные координаты из
        # пикселей и глубины.
        markers = detect_markers(rgb, depth, intrinsics)
        # Вычисляем локальные координаты окончания лучей
        rays = self._compute_local_points(depth, intrinsics, pose)
        return {
            "rays": rays,
            "markers": markers,
            "pose": pose,
        }

class OccupancyGridMap:
    """
    Блок представления среды (картография).

    Хранит двумерную сетку занятости в логарифмическом представлении (log‑odds).
    Каждая ячейка хранит значение log(p/(1-p)), где p — вероятность занятости.
    0 соответствует вероятности 0.5 (неизвестно), положительные значения
    означают, что ячейка скорее занята, отрицательные — скорее свободна.

    Обновление выполняется на основе лучей из блока восприятия сцены: свободные
    клетки вдоль луча получают отрицательный сдвиг log‑odds, а конечная ячейка
    (где измерен препятствие) — положительный. Значения saturate в диапазоне
    [L_min, L_max].
    """

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
        # Словарь для хранения log‑odds каждой ячейки
        self.log_odds: dict[tuple[int, int], float] = {}
        # Множество посещённых ячеек (где побывал робот)
        self.visited: set[tuple[int, int]] = set()

    def _coord_to_index(self, x: float, y: float) -> tuple[int, int]:
        """Преобразует координаты в индексы сетки по resolution."""
        ix = int(math.floor(x / self.resolution))
        iy = int(math.floor(y / self.resolution))
        return ix, iy

    def _bresenham(self, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        """
        Генерирует список клеток между (x0, y0) и (x1, y1) по алгоритму
        Брезенхэма, включая оба конца.
        """
        cells: list[tuple[int, int]] = []
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
        """
        Обновляет сетку занятости на основе списка лучей и текущей позы робота.

        :param rays: список троек (x_local, y_local, r) из ScenePerception
        :param pose: текущая поза робота (x, y, yaw)
        """
        robot_x, robot_y, robot_yaw = pose
        # Индексы ячейки, где находится робот
        start_ix, start_iy = self._coord_to_index(robot_x, robot_y)
        # Отмечаем клетку, где находится робот, как посещённую
        self.visited.add((start_ix, start_iy))
        # Для каждого луча определяем конечную ячейку и обновляем log‑odds
        for (x_local, y_local, _r) in rays:
            # Преобразуем из локальных координат камеры в мировые координаты
            cos_yaw = math.cos(robot_yaw)
            sin_yaw = math.sin(robot_yaw)
            world_x = robot_x + cos_yaw * x_local - sin_yaw * y_local
            world_y = robot_y + sin_yaw * x_local + cos_yaw * y_local
            end_ix, end_iy = self._coord_to_index(world_x, world_y)
            # Получаем клетки по прямой между роботом и препятствием
            cells = self._bresenham(start_ix, start_iy, end_ix, end_iy)
            if not cells:
                continue
            # Все клетки, кроме последней, считаем свободными
            for cell in cells[:-1]:
                lo = self.log_odds.get(cell, 0.0) + self.lo_free
                # Saturate
                lo = max(self.lo_min, min(self.lo_max, lo))
                self.log_odds[cell] = lo
            # Последняя клетка – занятая
            final_cell = cells[-1]
            lo_final = self.log_odds.get(final_cell, 0.0) + self.lo_occ
            lo_final = max(self.lo_min, min(self.lo_max, lo_final))
            self.log_odds[final_cell] = lo_final

    def update_visited(self, pose: tuple[float, float, float]) -> None:
        """
        Отмечает клетку, где находится робот, как посещённую.
        Вызывается, если требуется отметить трассу робота независимо от
        обновления по лучам.
        """
        ix, iy = self._coord_to_index(pose[0], pose[1])
        self.visited.add((ix, iy))

    def get_log_odds(self, ix: int, iy: int) -> float:
        """Возвращает log‑odds для заданной ячейки (по умолчанию 0)."""
        return self.log_odds.get((ix, iy), 0.0)

    def get_probability(self, ix: int, iy: int) -> float:
        """
        Возвращает апостериорную вероятность занятости для клетки,
        преобразуя log‑odds в вероятность через логистическую функцию.
        """
        lo = self.get_log_odds(ix, iy)
        # p = 1 / (1 + exp(-lo))
        return 1.0 / (1.0 + math.exp(-lo))

    def classify(self, ix: int, iy: int) -> str:
        """
        Классифицирует клетку как 'free', 'occupied' или 'unknown' по порогам.
        Пороговые значения выбраны в соответствии с рекомендацией: p > 0.65
        — занята, p < 0.35 — свободна, иначе неизвестна.
        """
        p = self.get_probability(ix, iy)
        if p > 0.65:
            return "occupied"
        if p < 0.35:
            return "free"
        return "unknown"

class ObjectMemoryEntry:
    """
    Запись в памяти объектов.

    Хранит уникальный идентификатор маркера, оценку его позиции в мировой
    системе координат и текущий статус: 'discovered', 'active' или 'visited'.
    Позиция обновляется по мере появления новых наблюдений (усреднением).
    """

    def __init__(self, object_id: int, position: tuple[float, float], status: str = "discovered"):
        self.id: int = int(object_id)
        self.position: tuple[float, float] = (float(position[0]), float(position[1]))
        self.status: str = str(status)
        # Считаем, сколько раз объект был замечен — для усреднения координат
        self._obs_count: int = 1

    def update_position(self, new_pos: tuple[float, float]) -> None:
        """
        Обновляет оценку позиции усреднением с предыдущим значением.
        """
        x_old, y_old = self.position
        x_new, y_new = new_pos
        n = self._obs_count
        # Простое скользящее среднее
        x_avg = (x_old * n + x_new) / (n + 1)
        y_avg = (y_old * n + y_new) / (n + 1)
        self.position = (x_avg, y_avg)
        self._obs_count = n + 1

class ObjectMemory:
    """
    Блок памяти объектов.

    Хранит обнаруженные объекты и их статусы. Предоставляет методы для
    добавления/обновления обнаружений, выбора активных объектов и пометки
    посещённых объектов.
    """

    def __init__(self) -> None:
        # Словарь: id -> ObjectMemoryEntry
        self.entries: dict[int, ObjectMemoryEntry] = {}
        # ID текущего активного объекта (или None)
        self.active_object_id: int | None = None

    def update_with_detections(self, detections: list[tuple[int, tuple[float, float, float], float]], pose: tuple[float, float, float]) -> None:
        """
        Обновляет память на основании списка детекций. Для каждой детекции
        вычисляет мировые координаты объекта и добавляет его в память или
        усредняет позицию, если объект уже присутствует.

        :param detections: список троек (id, (x_l, y_l, z_l), confidence)
        :param pose: текущая поза робота (x, y, yaw)
        """
        robot_x, robot_y, robot_yaw = pose
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        for det in detections:
            obj_id, (x_l, y_l, z_l), conf = det
            # Переводим координаты из локальной системы в мировую систему
            world_x = robot_x + cos_yaw * float(x_l) - sin_yaw * float(y_l)
            world_y = robot_y + sin_yaw * float(x_l) + cos_yaw * float(y_l)
            if obj_id not in self.entries:
                # Создаём новую запись
                self.entries[obj_id] = ObjectMemoryEntry(obj_id, (world_x, world_y), status="discovered")
            else:
                entry = self.entries[obj_id]
                # Обновляем позицию усреднением
                entry.update_position((world_x, world_y))

    def set_active(self, obj_id: int) -> None:
        """
        Устанавливает указанный объект активным. Снимает статус active с
        предыдущего активного объекта. Статус активного объекта не может
        быть 'visited'.
        """
        # Если объект не существует, ничего не делаем
        if obj_id not in self.entries:
            return
        # Снимаем флаг с предыдущего активного
        if self.active_object_id is not None and self.active_object_id in self.entries:
            prev_entry = self.entries[self.active_object_id]
            if prev_entry.status == "active":
                prev_entry.status = "discovered"
        # Устанавливаем новый активный
        entry = self.entries[obj_id]
        if entry.status != "visited":
            entry.status = "active"
            self.active_object_id = obj_id

    def mark_visited(self, obj_id: int) -> None:
        """
        Помечает объект как посещённый. Если этот объект был активным, снимает
        статус active. После пометки visited объект не будет участвовать в
        дальнейшем планировании.
        """
        entry = self.entries.get(obj_id)
        if entry is None:
            return
        entry.status = "visited"
        if self.active_object_id == obj_id:
            self.active_object_id = None

    def get_active_object(self) -> ObjectMemoryEntry | None:
        """Возвращает запись текущего активного объекта, если он есть."""
        if self.active_object_id is None:
            return None
        return self.entries.get(self.active_object_id)

    def get_discovered_objects(self) -> list[ObjectMemoryEntry]:
        """Возвращает список объектов со статусом 'discovered'."""
        return [entry for entry in self.entries.values() if entry.status == "discovered"]

    def get_all_entries(self) -> list[ObjectMemoryEntry]:
        """Возвращает список всех записей об объектах."""
        return list(self.entries.values())

class MissionLogic:
    """
    Блок логики миссии (конечный автомат).

    Определяет текущий режим работы навигационной системы: исследование,
    движение к активному объекту, уточнение координат и фиксацию. В этой
    упрощённой реализации задействованы два состояния: 'explore' (поиск
    объектов) и 'go_to_target' (движение к активному объекту). Переключение
    между состояниями осуществляется на основе состояния памяти объектов и
    заданной последовательности object_queue.
    """

    def __init__(self, object_queue: list[int], occupancy_map: OccupancyGridMap, object_memory: ObjectMemory,
                 *, exploration_speed: float = 0.3, target_speed: float = 0.5, arrival_threshold: float = 0.5) -> None:
        self.object_queue = list(object_queue)
        self.map = occupancy_map
        self.memory = object_memory
        self.state: str = "explore"
        self.current_target_id: int | None = None
        self.exploration_speed = float(exploration_speed)
        self.target_speed = float(target_speed)
        self.arrival_threshold = float(arrival_threshold)

    def _choose_next_target(self) -> None:
        """
        Выбирает следующую цель в строгом порядке object_queue.

        Логика:
        - если объект уже visited, переходим к следующему;
        - если первый непройденный объект ещё не обнаружен, остаёмся в режиме explore;
        - если первый непройденный объект обнаружен, назначаем его активным;
        - объекты дальше по очереди не могут стать активной целью раньше.
        """
        self.current_target_id = None
        self.state = "explore"

        for obj_id in self.object_queue:
            entry = self.memory.entries.get(obj_id)

            if entry is not None and entry.status == "visited":
                continue

            if entry is None:
                # Первый нужный объект ещё не найден: продолжаем поиск.
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
        """
        Обновляет состояние конечного автомата. Проверяет наличие активного
        объекта и меняет состояние в зависимости от дистанции до цели и
        статуса объекта.

        :param pose: текущая поза робота (x, y, yaw)
        """
        # Если уже есть активный объект
        active_entry = self.memory.get_active_object()
        if active_entry is not None and active_entry.status != "visited":
            self.current_target_id = active_entry.id
            self.state = "go_to_target"
            # Проверяем достижение цели
            # Вычисляем расстояние от робота до объекта
            dx = active_entry.position[0] - pose[0]
            dy = active_entry.position[1] - pose[1]
            dist = math.hypot(dx, dy)
            if dist < self.arrival_threshold:
                # Цель достигнута: помечаем объект visited
                self.memory.mark_visited(active_entry.id)
                # Переходим в следующее состояние (фиксируем объект)
                self.current_target_id = None
                self.state = "explore"
            # Ничего не делаем, остаёмся в состоянии движения к цели
            return
        # Нет активного объекта: ищем следующий
        self._choose_next_target()

    def compute_velocity(self, pose: tuple[float, float, float], sim_time: float = 0.0) -> tuple[float, float, float]:
        """
        Вычисляет желаемые скорости (vx, vy, vw) в зависимости от текущего
        состояния. Возвращаемые скорости находятся в локальной системе
        координат робота.

        :param pose: текущая поза робота (x, y, yaw)
        :param sim_time: симуляционное время (используется для паттерна исследований)
        :return: (vx, vy, vw)
        """
        # Движение к активной цели
        if self.state == "go_to_target" and self.current_target_id is not None:
            entry = self.memory.entries.get(self.current_target_id)
            if entry is None:
                # Странно, цель пропала
                return 0.0, 0.0, 0.0
            # Вычисляем вектор до цели
            dx_world = entry.position[0] - pose[0]
            dy_world = entry.position[1] - pose[1]
            # Угол до цели в мировой системе координат
            angle_to_goal = math.atan2(dy_world, dx_world)
            # Ошибка по углу (в локальной системе робота)
            yaw_error = angle_to_goal - pose[2]
            yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
            # Расстояние до цели
            distance = math.hypot(dx_world, dy_world)
            # Линейная скорость пропорциональна расстоянию, но ограничивается
            v_mag = min(self.target_speed, distance)
            # Скорости в локальной системе робота: vx вперёд, vy вбок
            vx = v_mag * math.cos(yaw_error)
            vy = v_mag * math.sin(yaw_error)
            # Угловая скорость стремится уменьшить yaw_error
            vw = yaw_error
            # Ограничиваем линейную и угловую скорости согласно максимально допустимым значениям
            vx = max(min(vx, 3.0), -3.0)
            vy = max(min(vy, 1.5), -1.5)
            vw = max(min(vw, 4.0), -4.0)
            return vx, vy, vw
        # Режим исследования
        # Простая стратегия: двигаться вперёд с небольшой синусоидальной модуляцией
        vx = self.exploration_speed
        vy = 0.0
        # Колебательный разворот для сканирования пространства
        vw = 0.3 * math.sin(sim_time * 0.5)
        # Ограничиваем скорости
        vx = max(min(vx, 3.0), -3.0)
        vy = max(min(vy, 1.5), -1.5)
        vw = max(min(vw, 4.0), -4.0)
        return vx, vy, vw

class AStarPlanner:
    """
    Простая реализация алгоритма A* для поиска пути по сетке занятости.

    Планы строятся в пространстве индексов сетки OccupancyGridMap. Неизвестные клетки
    рассматриваются как свободные; занятые клетки игнорируются. Планирование
    ограничивается прямоугольной областью вокруг старта и цели с запасом
    (margin), чтобы предотвратить бесконечное расширение поиска. Поддерживаются
    8‑направленные перемещения с диагональной стоимостью sqrt(2).
    """

    def __init__(self, occupancy_map: OccupancyGridMap, *, margin: int = 20, diagonal: bool = True) -> None:
        self.map = occupancy_map
        self.margin = int(margin)
        self.diagonal = bool(diagonal)

    def _neighbors(self, cell: tuple[int, int], bounds: tuple[int, int, int, int]) -> list[tuple[tuple[int, int], float]]:
        """
        Возвращает соседние клетки и стоимость перехода.

        :param cell: текущая клетка (ix, iy)
        :param bounds: ограничивающий прямоугольник (xmin, xmax, ymin, ymax)
        :return: список (neighbor_cell, cost)
        """
        (ix, iy) = cell
        neighbors: list[tuple[tuple[int, int], float]] = []
        # 4‑ и 8‑связные соседства
        directions = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
        ] if self.diagonal else [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        ]
        xmin, xmax, ymin, ymax = bounds
        for dx, dy, cost in directions:
            nx, ny = ix + dx, iy + dy
            # Ограничиваем поиск заданными пределами
            if nx < xmin or nx > xmax or ny < ymin or ny > ymax:
                continue
            # Проверяем, что клетка не занята
            if self.map.classify(nx, ny) == "occupied":
                continue
            neighbors.append(((nx, ny), cost))
        return neighbors

    def _heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        """Эвристика Евклидово расстояние между двумя клетками."""
        return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))

    def plan(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
        """
        Выполняет поиск A* из start в goal. Возвращает список клеток от start до goal
        (включая оба), либо None, если путь не найден.
        """
        if start == goal:
            return [start]
        # Определяем поисковый прямоугольник
        xmin = min(start[0], goal[0]) - self.margin
        xmax = max(start[0], goal[0]) + self.margin
        ymin = min(start[1], goal[1]) - self.margin
        ymax = max(start[1], goal[1]) + self.margin
        # Множества и структуры для A*
        import heapq
        open_heap: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, start))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start: 0.0}
        f_score: dict[tuple[int, int], float] = {start: self._heuristic(start, goal)}
        closed_set: set[tuple[int, int]] = set()
        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                # Восстанавливаем путь
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
        # Путь не найден
        return None

    def indices_to_world(self, path: list[tuple[int, int]]) -> list[tuple[float, float]]:
        """Преобразует список индексов сетки в мировые координаты центров клеток."""
        resolution = self.map.resolution
        return [((ix + 0.5) * resolution, (iy + 0.5) * resolution) for (ix, iy) in path]

class PurePursuitController:
    """
    Локальный планировщик «pure pursuit» для следования за траекторией.

    Алгоритм выбирает ближайшую точку на пути, находящуюся на расстоянии
    не менее lookahead, и генерирует скорости, направленные к этой точке.
    """

    def __init__(self, lookahead: float = 0.8, max_speed: float = 0.6) -> None:
        self.lookahead = float(lookahead)
        self.max_speed = float(max_speed)
        # Индекс текущего вейпоинта в пути
        self._target_index: int = 0

    def reset(self) -> None:
        """Сбрасывает внутреннее состояние (например, при смене цели)."""
        self._target_index = 0

    def _find_target_index(self, path: list[tuple[float, float]], pose: tuple[float, float, float]) -> int:
        """
        Находит индекс точки пути, находящейся на расстоянии >= lookahead от робота.
        Если таких точек нет, возвращает последний индекс.
        """
        rx, ry, _ = pose
        # Ищем первую точку, на расстоянии >= lookahead
        for i in range(self._target_index, len(path)):
            px, py = path[i]
            dist = math.hypot(px - rx, py - ry)
            if dist >= self.lookahead:
                return i
        # Иначе возвращаем последний
        return len(path) - 1

    def compute_command(self, path: list[tuple[float, float]], pose: tuple[float, float, float]) -> tuple[float, float, float]:
        """
        Вычисляет (vx, vy, vw) для следования по пути. Возвращает (0,0,0), если
        путь пуст или достигнут последний вейпоинт.
        """
        if not path:
            return 0.0, 0.0, 0.0
        # Обновляем индекс цели
        self._target_index = self._find_target_index(path, pose)
        target = path[self._target_index]
        # Координаты до цели в мировой системе
        dx_world = target[0] - pose[0]
        dy_world = target[1] - pose[1]
        distance = math.hypot(dx_world, dy_world)
        if distance < 1e-3:
            # Если цель достигнута, остаёмся на месте
            return 0.0, 0.0, 0.0
        # Угол на цель
        angle_to_goal = math.atan2(dy_world, dx_world)
        yaw_error = angle_to_goal - pose[2]
        # Нормализуем угол
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
        # Линейная скорость ограничивается расстоянием и max_speed
        v_mag = min(self.max_speed, distance)
        # Скорости в локальной системе робота
        vx = v_mag * math.cos(yaw_error)
        vy = v_mag * math.sin(yaw_error)
        vw = yaw_error
        # Ограничиваем линейную и угловую скорости до заданных пределов
        vx = max(min(vx, 3.0), -3.0)
        vy = max(min(vy, 1.5), -1.5)
        # Ограничиваем вращение по модулю 2
        if vw > 4.0:
            vw = 4.0
        elif vw < -4.0:
            vw = -4.0
        return vx, vy, vw

class NavigationPlanner:
    """
    Навигационный планировщик, объединяющий A* и Pure Pursuit.

    Для каждого нового целевого объекта строит путь на основе A* по карте.
    Пока путь актуален, следование осуществляется контроллером pure pursuit.
    """

    def __init__(self, occupancy_map: OccupancyGridMap, *, lookahead: float = 0.8, max_speed: float = 0.6, margin: int = 20) -> None:
        self.map = occupancy_map
        self.planner = AStarPlanner(occupancy_map, margin=margin)
        self.controller = PurePursuitController(lookahead=lookahead, max_speed=max_speed)
        self.current_goal: tuple[float, float] | None = None
        self.path_world: list[tuple[float, float]] = []
        self._last_start: tuple[int, int] | None = None

    def _needs_replan(self, start_cell: tuple[int, int], goal_cell: tuple[int, int]) -> bool:
        """
        Определяет, требуется ли перестроение пути. Это происходит, если
        текущая цель изменилась или стартовая клетка существенно сместилась.
        """
        if not self.path_world:
            return True
        # Если цель изменилась — перестраиваем
        if self.current_goal != goal_cell:
            return True
        # Если стартовая клетка изменилась — можно перестроить для надёжности
        if self._last_start is None or start_cell != self._last_start:
            return True
        return False

    def compute_command(self, pose: tuple[float, float, float], target_pos: tuple[float, float]) -> tuple[float, float, float]:
        """
        Возвращает желаемые скорости для движения к целевой точке target_pos.
        При необходимости перестраивает путь и сбрасывает контроллер.
        """
        # Переводим целевую позицию в индекс сетки
        goal_ix, goal_iy = self.map._coord_to_index(target_pos[0], target_pos[1])
        start_ix, start_iy = self.map._coord_to_index(pose[0], pose[1])
        # Проверка на необходимость перестроения пути
        if self._needs_replan((start_ix, start_iy), (goal_ix, goal_iy)):
            path_indices = self.planner.plan((start_ix, start_iy), (goal_ix, goal_iy))
            if path_indices is None:
                # Путь не найден — fallback: направляемся напрямую
                # Сбросим путь и будем возвращать простую навигацию
                self.path_world = []
                self.current_goal = (goal_ix, goal_iy)
                self.controller.reset()
            else:
                self.path_world = self.planner.indices_to_world(path_indices)
                self.current_goal = (goal_ix, goal_iy)
                self._last_start = (start_ix, start_iy)
                self.controller.reset()
        # Если путь пуст (не найден), падаем обратно на прямое движение
        if not self.path_world:
            # Вычисляем скорости прямо к объекту, аналогично MissionLogic
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
            # Ограничиваем скорости
            vx = max(min(vx, 3.0), -3.0)
            vy = max(min(vy, 1.5), -1.5)
            if vw > 2.0:
                vw = 2.0
            elif vw < -2.0:
                vw = -2.0
            return vx, vy, vw
        # Следуем по найденному пути
        return self.controller.compute_command(self.path_world, pose)

def _unwrap_env_from_robot(robot: AliengoRobotInterface):
    env = getattr(robot, "env", None)
    while env is not None and hasattr(env, "env") and getattr(env, "env") is not env:
        env = env.env
    return env

def _infer_control_dt(robot: AliengoRobotInterface, fallback_dt: float = 0.02) -> float:
    env = _unwrap_env_from_robot(robot)
    dt = getattr(env, "dt", None) if env is not None else None
    try:
        dt_value = float(dt)
        if dt_value > 0.0:
            return dt_value
    except (TypeError, ValueError):
        pass
    return float(fallback_dt)


class CloseObstacleRecovery:
    """
    Простой детерминированный механизм обхода стены.

    Если робот подходит к препятствию ближе заданного порога, recovery
    запускает завершённый манёвр:
    1. фиксированно отходит назад примерно на 1 метр;
    2. затем выполняет поворот примерно на 90 градусов;
    3. после короткого cooldown возвращает управление основной логике.

    Манёвр не переоценивается каждый кадр, поэтому не зацикливается на
    попеременных поворотах возле одной и той же стены.
    """

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

        self.mode = "idle"  # idle / backup / turn / cooldown
        self.steps_left = 0
        self.cooldown_left = 0
        self.turn_sign = 1.0

        self.backup_steps = max(1, int(math.ceil(self.backup_distance_m / (abs(self.reverse_speed) * self.control_dt))))
        self.turn_steps = max(1, int(math.ceil(self.turn_angle_rad / (abs(self.turn_speed) * self.control_dt))))

    def _choose_turn_sign(self, depth: np.ndarray) -> float:
        """Поворачивает в сторону, где средняя глубина больше."""
        h, w = depth.shape
        left = depth[:, : w // 3]
        right = depth[:, 2 * w // 3 :]
        left_vals = left[np.isfinite(left) & (left > 0.0)]
        right_vals = right[np.isfinite(right) & (right > 0.0)]
        left_mean = float(np.mean(left_vals)) if left_vals.size else 0.0
        right_mean = float(np.mean(right_vals)) if right_vals.size else 0.0
        return 1.0 if left_mean >= right_mean else -1.0

    def _front_min_distance(self, depth: np.ndarray) -> float:
        """Минимальная валидная дистанция в центральной зоне фронтального обзора."""
        h, w = depth.shape
        col_margin = max(w // 6, 1)
        row_start = h // 3
        row_end = 2 * h // 3
        center = depth[row_start:row_end, col_margin : w - col_margin]
        vals = center[np.isfinite(center) & (center > 0.0)]
        if vals.size == 0:
            return float("inf")
        return float(np.min(vals))

    def compute_override(self, depth: np.ndarray | None) -> tuple[bool, tuple[float, float, float], str]:
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
            if front_min >= self.trigger_distance:
                return False, (0.0, 0.0, 0.0), ""
            self.turn_sign = self._choose_turn_sign(depth)
            self.mode = "backup"
            self.steps_left = self.backup_steps

        if self.mode == "backup":
            self.steps_left -= 1
            if self.steps_left <= 0:
                self.mode = "turn"
                self.steps_left = self.turn_steps
            return True, (self.reverse_speed, 0.0, 0.0), f"Близко к стене ({front_min:.2f} м), отхожу назад"

        if self.mode == "turn":
            self.steps_left -= 1
            if self.steps_left <= 0:
                self.mode = "cooldown"
                self.cooldown_left = self.cooldown_steps
            return True, (0.0, 0.0, self.turn_sign * self.turn_speed), "Выполняю поворот на 90 градусов"

        return False, (0.0, 0.0, 0.0), ""


class _CameraRenderer:

    def __init__(self, enabled: bool, depth_max_m: float):
        self.enabled = bool(enabled)
        self.depth_max_m = max(float(depth_max_m), 0.1)
        self._window_name = "Front Camera (Intel RealSense D435-like)"
        self._cv2 = None
        self._active = False
        if not self.enabled:
            return
        try:
            import cv2
        except Exception as exc:
            print(f"Отрисовка камеры отключена: не удалось импортировать cv2 ({exc})")
            self.enabled = False
            return
        self._cv2 = cv2
        self._cv2.namedWindow(self._window_name, self._cv2.WINDOW_NORMAL)
        self._active = True

    def show(self, camera: CameraState) -> None:
        if not self._active or not isinstance(camera, CameraState):
            return
        image = camera.rgb
        depth = camera.depth
        if image is None or depth is None:
            return

        rgb = np.asarray(image)
        depth_m = np.asarray(depth, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] < 3 or depth_m.ndim != 2:
            return
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = rgb[..., :3]
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=self.depth_max_m, neginf=0.0)
        depth_m = np.clip(depth_m, 0.0, self.depth_max_m)
        depth_u8 = (depth_m * (255.0 / self.depth_max_m)).astype(np.uint8)

        cv2 = self._cv2
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        depth_color = cv2.resize(depth_color, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        view = np.concatenate((rgb_bgr, depth_color), axis=1)

        cv2.putText(view, "RGB", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            view,
            f"Depth 0..{self.depth_max_m:.1f}m",
            (rgb.shape[1] + 10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self._window_name, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.close()

    def close(self) -> None:
        if not self._active or self._cv2 is None:
            return
        self._cv2.destroyWindow(self._window_name)
        self._active = False

def run(
    robot: AliengoRobotInterface,
    steps: int = 15000,
    render_camera: bool = False,
    camera_depth_max_m: float = 4.0,
    seed: int = 0,
) -> None:
    robot.reset()
    env = getattr(robot, "env", None)
    if env is None:
        raise ValueError("Интерфейс робота должен предоставлять 'env' для обязательного логирования.")

    logger = CompetitionRunLogger(env=env, seed=int(seed))
    camera_renderer = _CameraRenderer(enabled=render_camera, depth_max_m=camera_depth_max_m)
    control_dt = _infer_control_dt(robot, fallback_dt=0.02)
    requested_steps = max(int(steps), 1)
    nominal_dt = 0.02
    target_duration_s = requested_steps * nominal_dt
    total_steps = max(int(round(target_duration_s / control_dt)), 1)
    print(
        f"[Контроллер] dt={control_dt:.4f}с, requested_steps={requested_steps}, "
        f"effective_steps={total_steps}"
    )
    object_queue = list(getattr(env, "SEQUENCE_OF_OBJECTS", []))
    print(f"[Контроллер] отрисовка_камеры={'включена' if camera_renderer.enabled else 'выключена'}")
    print(f"[Контроллер] object_queue={object_queue}")
    # Настраивайте эти значения, чтобы менять поведение демо.
    # Параметры, завязанные на время, пересчитываются через шаг симуляции,
    # потому время в секундах работают в симуляции правильно. Значения ниже
    # использовались в исходной демо‑траектории. В этой реализации они
    # сохранены для возможного использования участником, но логика движения
    # в большинстве случаев определяется конечным автоматом MissionLogic.
    warmup_s = 0.4
    ramp_s = 1.2
    trajectory_period_s = 8.0
    forward_speed_mean = 0.40
    forward_speed_amp = 0.35
    lateral_speed_amp = 0.22
    yaw_rate_amp = 0.75
    yaw_rate_damping = 0.55
    ang_vel_scale = 0.25
    # ================== USER PARAMETERS END ==================

    segment_start_t = 0.0

    # Создаём обработчик входных данных для синхронизации RGB и глубины
    # и интегрирования собственной позы. Этот класс реализует первый
    # блок архитектуры — приём и подготовку данных от сенсоров. Параметр
    # control_dt используется как fallback для интегрирования позы.
    input_handler = InputHandler(control_dt)

    # Блок восприятия сцены: преобразует сырые данные сенсоров в набор
    # лучей и выполняет детекцию объектов. В последующих итерациях мы будем
    # передавать данные из этого блока в картографию и планирование.
    scene_perception = ScenePerception()

    # Блок картографии: 2D‑сетка занятости. Этот объект хранит log‑odds
    # каждой клетки и обновляет их на основе лучей из ScenePerception.
    occupancy_map = OccupancyGridMap()

    # Блок памяти объектов: хранит сведения об обнаруженных маркерах,
    # их позициях и статусах (discovered/active/visited).
    object_memory = ObjectMemory()

    # Блок логики миссии: конечный автомат, который решает, исследовать
    # окружение или двигаться к выбранной цели. Он выбирает следующий
    # объект из object_queue, когда тот появляется в памяти, и
    # вычисляет желаемые скорости для робота. Параметры mission logic
    # (exploration_speed, target_speed, arrival_threshold) можно
    # настроить в соответствии с требуемой динамикой движения.
    mission_logic = MissionLogic(
        object_queue=object_queue,
        occupancy_map=occupancy_map,
        object_memory=object_memory,
        exploration_speed=1.0,
        target_speed=3.0,
        arrival_threshold=1.5,
    )

    # Множество для отслеживания уже залогированных объектов. Это
    # предотвращает многократное логирование одного и того же маркера.
    logged_object_ids: set[int] = set()

    # Навигационный планировщик: отвечает за построение траектории с помощью
    # A* и локальное следование по ней (Pure Pursuit). Скорости, используемые
    # планировщиком, синхронизированы с mission_logic.target_speed.
    navigation_planner = NavigationPlanner(
        occupancy_map,
        lookahead=0.8,
        max_speed=mission_logic.target_speed,
        margin=20,
    )

    # Аварийное поведение возле стены и ограничитель ускорений.
    wall_recovery = CloseObstacleRecovery(
        trigger_distance=0.50,
        reverse_speed=-0.35,
        turn_speed=1.2,
        backup_distance_m=1.0,
        turn_angle_rad=math.pi / 2.0,
        control_dt=control_dt,
        cooldown_steps=10,
    )
    prev_state: str = "explore"
    prev_target_id: int | None = None
    # Счётчик для действия X (робот стоит на месте 3 секунды)
    after_visit_countdown: int = 0
    spin_mode: bool = False
    current_spin_speed: float = 0.0

    try:
        initial_observation = robot.get_observation()
        initial_camera_payload = robot.get_camera()
        print(
            "[Контроллер] Предпросмотр API:"
            f" observation_type={type(initial_observation).__name__},"
            f" camera_payload={'да' if initial_camera_payload is not None else 'нет'}"
        )
        if initial_camera_payload is None:
            print(
                "[Контроллер] Предупреждение: данные фронтальной камеры недоступны. "
                "Проверьте, что симулятор не запущен в headless-режиме и что включён front_camera_enabled."
            )

        for step_index in range(total_steps):
            state = robot.get_state()

            # Камеру можно брать и из state, и напрямую через robot.get_camera().
            camera_payload = robot.get_camera()
            camera_state = state.camera
            if (camera_state.rgb is None or camera_state.depth is None) and isinstance(camera_payload, dict):
                camera_state = CameraState(
                    rgb=camera_payload.get("image"),
                    depth=camera_payload.get("depth"),
                )
            elif (camera_state.rgb is None or camera_state.depth is None) and isinstance(camera_payload, CameraState):
                camera_state = camera_payload
            camera_renderer.show(camera_state)
            # state.imu.wz напрямую.
            omega_z = state.imu.wz / ang_vel_scale

            # Используем InputHandler для формирования пакета входных данных.
            # Этот вызов синхронно обновляет оценку собственной позы и
            # возвращает словарь, содержащий rgb, depth, pose, timestamp и
            # параметры калибровки камеры.
            input_data = input_handler.get_frame(state, camera_state)
            # Передаём пакет данных в блок восприятия сцены для извлечения
            # лучей и обнаружения объектов. Полученная структура scene_data
            # содержит локальные окончания лучей (rays), список маркеров
            # (markers) и актуальную позу. Пока эти данные не используются
            # в логике демо, но они будут необходимы для картографии и
            # планирования в следующих этапах.
            scene_data = scene_perception.process(input_data)

            # Обновляем карту занятости по текущим лучам глубины.
            occupancy_map.update(scene_data.get("rays", []), scene_data.get("pose"))

            # Обновляем память объектов и определяем новые ID в кадре.
            existing_ids = set(object_memory.entries.keys())
            object_memory.update_with_detections(scene_data.get("markers", []), scene_data.get("pose"))
            # Новые объекты, которые появились в этой итерации
            new_ids = set(object_memory.entries.keys()) - existing_ids
            for new_id in new_ids:
                # Определяем ближайший непройденный объект из очереди
                next_unvisited_id = None
                for obj_id in object_queue:
                    entry = object_memory.entries.get(obj_id)
                    if entry is None or entry.status != "visited":
                        next_unvisited_id = obj_id
                        break
                # Логирование по сценарию в зависимости от позиции в очереди
                if next_unvisited_id is not None and new_id == next_unvisited_id:
                    print(f"Нашёл объект {new_id}")
                    print("Проверяю, в какой последовательности нужно зафиксировать объекты")
                else:
                    print(f"Объект {new_id} находится дальше в очереди, сейчас его фиксировать не нужно")
                    print("Запомнил координаты")
                    print("Ищу дальше объекты")

            # ================= USER CONTROL LOGIC START =================
            # Это основной блок для логики участника.
            # Здесь нужно читать измерения, принимать решение и формировать
            # команды движения. Логирование найденного объекта тоже делается
            # отсюда.
            #
            # Формат данных эквивалентен данных:
            # - вход команды: vx, vy, wz
            # - выход состояния: measured_vx, measured_vy, measured_wz
            # - joint_states: joint_names, relative_dof_pos, dof_vel
            # - imu: base_ang_vel, base_lin_acc
            # - camera: camera_data["image"], camera_data["depth"]
            # - порядок объектов: object_queue
            #
            # Ниже приведён обязательный шаблон. Участник должен:
            # 1. реализовать get_found_object_id(...)
            # 2. при обнаружении объекта вернуть его id
            # 3. обязательно вызвать log_found_object(...)
            #
            # Если объект не найден, верните None.
            sim_t = state.sim_time_s

            joint_names = state.joints.name
            relative_dof_pos = state.q
            dof_vel = state.q_dot
            measured_vx = state.vx
            measured_vy = state.vy
            measured_wz = state.wz
            base_ang_vel = state.imu.angular_velocity_xyz
            base_lin_acc = np.zeros(3, dtype=np.float32)
            camera_data = camera_payload if isinstance(camera_payload, dict) else {
                "image": camera_state.rgb,
                "depth": camera_state.depth,
            }

            # Разделение интерфейсов по назначению:
            # 1) сенсорные входы навигации;
            navigation_rgb = camera_data["image"]
            navigation_depth = camera_data["depth"]
            # 2) служебные входы миссии;
            mission_object_queue = object_queue
            # 3) исполнительные выходы будут сформированы ниже в виде vx, vy, vw.

            def log_found_object(object_id: int) -> None:
                """Фиксирует найденный объект в судейском логе."""
                logger.log_detected_object_at_time(int(object_id), float(sim_t))

            pending_found_object_ids = [
                det_id
                for (det_id, _loc, _conf) in scene_data.get("markers", [])
                if det_id in mission_object_queue
                and det_id not in logged_object_ids
                and object_memory.entries.get(det_id) is not None
                and object_memory.entries[det_id].status != "visited"
            ]

            def get_found_object_id(
                current_state,
                current_camera_data,
                current_object_queue,
            ):
                """Возвращает id следующего подтверждённого объекта для логирования или None."""
                _ = current_state
                _ = current_camera_data
                _ = current_object_queue
                if pending_found_object_ids:
                    return pending_found_object_ids.pop(0)
                return None

            while True:
                detected_object_id = get_found_object_id(
                    state,
                    {"image": navigation_rgb, "depth": navigation_depth},
                    mission_object_queue,
                )
                if detected_object_id is None:
                    break
                log_found_object(detected_object_id)
                logged_object_ids.add(detected_object_id)

            # Обновляем состояние конечного автомата, используя текущую позу
            mission_logic.update(scene_data.get("pose"))
            current_state = mission_logic.state
            current_target = mission_logic.current_target_id

            # Считаем число посещённых объектов
            visited_count = sum(1 for e in object_memory.entries.values() if e.status == "visited")
            total_targets = len(mission_object_queue)
            # Проверяем завершение миссии: если все объекты посещены
            if visited_count >= total_targets and not spin_mode:
                spin_mode = True
                current_spin_speed = 0.0
                print("Все объекты найдены, начинаю вращаться на месте")

            # Режим вращения после завершения миссии
            if spin_mode:
                # Увеличиваем скорость вращения до предела 2 рад/с
                current_spin_speed = min(4.0, current_spin_speed + 0.10)
                vx = 0.0
                vy = 0.0
                vw = current_spin_speed
                print(f"Вращаюсь на месте, скорость {vw:.2f}")
            # Режим действия X (трёхсекундная пауза)
            elif after_visit_countdown > 0:
                after_visit_countdown -= 1
                vx = 0.0
                vy = 0.0
                vw = 0.0
                print("Действие X: стою на месте")
            else:
                # Логирование смены состояния и целевого объекта
                if prev_state != current_state:
                    if current_state == "explore":
                        # Возврат к поиску после фиксации
                        if prev_state == "go_to_target" and prev_target_id is not None:
                            print(f"Зафиксировал объект {prev_target_id}")
                            print("Ищу следующий объект")
                            # Запускаем действие X на 2 секунды
                            after_visit_countdown = int(round(2.0 / max(control_dt, 1e-6)))
                        else:
                            # Начало поиска
                            print("Ищу объекты")
                    elif current_state == "go_to_target" and current_target is not None:
                        print(f"Иду к объекту {current_target}")
                elif current_state == "explore" and prev_state == "explore" and step_index == 0:
                    # Выводим сообщение в самом начале или после возврата к поиску
                    print("Ищу объекты")
                # Вычисляем скорости в зависимости от режима
                if current_state == "go_to_target" and current_target is not None:
                    entry = object_memory.entries.get(current_target)
                    if entry is not None:
                        vx, vy, vw = navigation_planner.compute_command(scene_data.get("pose"), entry.position)
                    else:
                        vx, vy, vw = 0.0, 0.0, 0.0
                else:
                    vx, vy, vw = mission_logic.compute_velocity(scene_data.get("pose"), sim_t)
            # Аварийное поведение, если робот вплотную подошёл к стене.
            depth_frame = input_data.get("depth")
            has_override, override_cmd, override_msg = wall_recovery.compute_override(depth_frame)
            if has_override:
                print(override_msg)
                vx, vy, vw = override_cmd

            # Обновляем предыдущие значения состояний
            prev_state = current_state
            prev_target_id = current_target
            # ================== USER CONTROL LOGIC END ==================

            # Исполнительный выход: целевые команды движения.
            robot.set_speed(vx, vy, vw)
            robot.step()
            logger.log_step(step_index * control_dt)
            robot.get_observation()  # Пример доступа к наблюдению после step().

            if robot.is_fallen():
                robot.stop()
                robot.reset()
                segment_start_t = (step_index + 1) * control_dt
                print("[Контроллер] робот упал -> сброс")
                continue
    finally:
        logger.close()
        camera_renderer.close()
        robot.stop()