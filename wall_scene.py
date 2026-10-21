"""Head-coupled window: lean sideways to peek behind a foreground wall."""
import cv2
import numpy as np

from controls import head_to_cam
from scene import (W, H, FOCAL, CUBE_C, VERTICES, FACES, NORMALS,
                   ROOM_LINES, ROOM_COLORS, LIGHT)


def project_window(points, camera):
    """Off-axis perspective: the screen is a fixed window in the z=0 plane."""
    points = np.asarray(points, dtype=float)
    camera = np.asarray(camera, dtype=float)
    relative = points - camera
    depth = -relative[:, 2]
    return np.stack([
        W / 2 + FOCAL * (relative[:, 0] / depth + camera[0] / camera[2]),
        H / 2 - FOCAL * (relative[:, 1] / depth + camera[1] / camera[2]),
    ], axis=1)


def head_to_window(dx, dy, wfrac):
    distance = head_to_cam(dx, dy, wfrac)[2]
    return np.array([np.clip(dx, -1, 1) * 5, np.clip(dy, -1, 1) * 2.5, distance])


# Static objects at several depths, including one initially hidden by the wall.
OBJECTS = [
    (np.array([-1.2, 0.1, -4]), 0.65, (210, 180, 45)),
    (np.array([1.5, -0.5, -6]), 0.8, (60, 130, 225)),
    (np.array([-2.0, 0.9, -7]), 0.5, (160, 90, 210)),
]
OBJECT_VERTICES = np.array([(VERTICES - CUBE_C) * size + center
                            for center, size, _ in OBJECTS])
POLYGONS = OBJECT_VERTICES[:, np.array(FACES)].reshape(-1, 4, 3)
POLYGON_CENTERS = POLYGONS.mean(axis=1)
POLYGON_NORMALS = np.tile(NORMALS, (len(OBJECTS), 1))
POLYGON_COLORS = [tuple(int(c * (0.6 + 0.4 * max(0, normal @ LIGHT))) for c in color)
                  for _, _, color in OBJECTS for normal in NORMALS]

# Smaller smooth wall, entirely in front of the objects: z=-2 to z=-2.6.
WALL_VERTICES = (VERTICES - CUBE_C) * [2, 2, 0.3] + [-2, 0, -2.3]
WALL_POLYGONS = WALL_VERTICES[np.array(FACES)]
WALL_CENTERS = WALL_POLYGONS.mean(axis=1)
WALL_FRONT = WALL_POLYGONS[1]
WALL_SIDE = WALL_POLYGONS[3]
WALL_EDGE = WALL_VERTICES[[5, 7]]
WALL_COLORS = [tuple(int(c * (0.6 + 0.4 * max(0, normal @ LIGHT))) for c in (100, 112, 130))
               for normal in NORMALS]

# Four solid sections leave a real opening: x=-3..-1, y=-0.8..0.8.
WINDOW_FRONT = np.array([[-3, -0.8, -2], [-1, -0.8, -2],
                         [-1, 0.8, -2], [-3, 0.8, -2]], float)
WALL_SECTIONS = [(-4, -3, -2, 2), (-1, 0, -2, 2),
                 (-3, -1, -2, -0.8), (-3, -1, 0.8, 2)]
wall_parts = []
for left, right, bottom, top in WALL_SECTIONS:
    center = [(left + right) / 2, (bottom + top) / 2, -2.3]
    half_size = [(right - left) / 2, (top - bottom) / 2, 0.3]
    vertices = (VERTICES - CUBE_C) * half_size + center
    wall_parts.append(vertices[np.array(FACES)])
WALL_MESH = np.array(wall_parts).reshape(-1, 4, 3)
WALL_MESH_CENTERS = WALL_MESH.mean(axis=1)
WALL_MESH_NORMALS = np.tile(NORMALS, (len(WALL_SECTIONS), 1))


def draw_wall(image, camera):
    polygons = np.round(project_window(WALL_MESH.reshape(-1, 3), camera)
                        ).astype(np.int32).reshape(-1, 4, 2)
    visible = np.einsum('ij,ij->i', WALL_MESH_NORMALS, camera - WALL_MESH_CENTERS) > 0
    for index in np.argsort(WALL_MESH_CENTERS[:, 2]):
        if visible[index]:
            cv2.fillConvexPoly(image, polygons[index], WALL_COLORS[index % 6], cv2.LINE_AA)
    for contour in (WALL_FRONT, WINDOW_FRONT):
        xy = np.round(project_window(contour, camera)).astype(np.int32)
        cv2.polylines(image, [xy], True, (130, 150, 175), 1, cv2.LINE_AA)


def render_wall(dx, dy, wfrac, background):
    camera = head_to_window(dx, dy, wfrac)
    image = background.copy()
    room = np.round(project_window(ROOM_LINES.reshape(-1, 3), camera)
                    ).astype(np.int32).reshape(-1, 2, 2)
    for line, color in zip(room, ROOM_COLORS):
        cv2.line(image, tuple(line[0]), tuple(line[1]), color, 1, cv2.LINE_AA)
    projected = np.round(project_window(POLYGONS.reshape(-1, 3), camera)
                         ).astype(np.int32).reshape(-1, 4, 2)
    visible = np.einsum('ij,ij->i', POLYGON_NORMALS, camera - POLYGON_CENTERS) > 0
    for index in np.argsort(POLYGON_CENTERS[:, 2]):
        if visible[index]:
            cv2.fillConvexPoly(image, projected[index], POLYGON_COLORS[index], cv2.LINE_AA)
            cv2.polylines(image, [projected[index]], True, (220, 235, 245), 2, cv2.LINE_AA)
    draw_wall(image, camera)
    return image
