"""Static cube and room rendered from an orbiting camera."""
import cv2
import numpy as np

W, H = 1280, 720
FOCAL = 0.9 * W
FLOOR_Y = -2.0
WALL_Z = -10.0
BG_TOP, BG_BOT = (38, 32, 26), (8, 7, 5)
WALL_C, FLOOR_C = (110, 105, 95), (75, 105, 70)
EDGE_C = (240, 240, 240)
LIGHT = np.array([-0.5, 0.8, 0.6])
LIGHT /= np.linalg.norm(LIGHT)
CUBE_C = np.array([0.0, 0.2, 0.0])
CUBE_H = 1.0
VERTICES = CUBE_C + CUBE_H * np.array([
    [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]], float)
FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 2, 6, 4),
         (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6)]
FACE_POINTS = VERTICES[np.array(FACES)]
FACE_CENTERS = FACE_POINTS.mean(axis=1)
NORMALS = FACE_CENTERS - CUBE_C
NORMALS /= np.linalg.norm(NORMALS, axis=1, keepdims=True)
BASE_COLORS = [(60, 130, 220), (200, 120, 60), (240, 200, 120),
               (90, 90, 160), (80, 180, 120), (150, 90, 170)]
FACE_COLORS = [tuple(int(c * (0.6 + 0.4 * max(0, normal @ LIGHT))) for c in color)
               for normal, color in zip(NORMALS, BASE_COLORS)]
EDGES = sorted({tuple(sorted((face[i], face[(i + 1) % 4])))
                for face in FACES for i in range(4)})
EDGE_FACES = [[i for i, face in enumerate(FACES) if a in face and b in face]
              for a, b in EDGES]

# Build static world geometry once, not every video frame.
WALL_LINES = np.array(
    [[[x, FLOOR_Y, WALL_Z], [x, 8, WALL_Z]] for x in np.arange(-16, 16.01, 1.25)]
    + [[[-16, y, WALL_Z], [16, y, WALL_Z]] for y in np.arange(FLOOR_Y, 8.01, 1)])
FLOOR_LINES = np.array(
    [[[x, FLOOR_Y, WALL_Z], [x, FLOOR_Y, 0.4]] for x in np.arange(-15, 15.01, 1.5)]
    + [[[-16, FLOOR_Y, z], [16, FLOOR_Y, z]] for z in np.arange(-19, 0.01, 1.2)])
ROOM_LINES = np.concatenate([WALL_LINES, FLOOR_LINES])
ROOM_COLORS = [WALL_C] * len(WALL_LINES) + [FLOOR_C] * len(FLOOR_LINES)
SHADOW = np.array([[x * CUBE_H * 1.45, FLOOR_Y + 0.01, z * CUBE_H * 1.45]
                   for x, z in ((-1, -1), (1, -1), (1, 1), (-1, 1))])

OBJECT_POINTS = np.concatenate([VERTICES, SHADOW])


def view_basis(theta, cy, dist):
    cam = np.array([dist * np.sin(theta), cy, dist * np.cos(theta)])
    forward = CUBE_C - cam
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return cam, right, up, forward


def to_cam(points, cam, right, up, forward):
    relative = np.asarray(points, float) - cam
    return np.stack([relative @ right, relative @ up, -(relative @ forward)], axis=1)


def project(points):
    depth = -points[:, 2]
    return np.stack([W / 2 + FOCAL * points[:, 0] / depth,
                     H / 2 - FOCAL * points[:, 1] / depth], axis=1)


def clip_line(points, near=0.3):
    lines, visible = clip_lines(np.asarray(points).reshape(1, 2, 3), near)
    return lines[0] if visible[0] else None


def clip_lines(lines, near=0.3):
    """Clip a batch of camera-space segments before perspective division."""
    lines = np.array(lines, dtype=float, copy=True)
    depths = -lines[:, :, 2]
    visible = np.any(depths > near, axis=1)
    for index in (0, 1):
        other = 1 - index
        mask = visible & (depths[:, index] <= near)
        fraction = (near - depths[mask, index]) / (depths[mask, other] - depths[mask, index])
        lines[mask, index] += (lines[mask, other] - lines[mask, index]) * fraction[:, None]
    return lines, visible


def render_scene(theta, cy, dist, background):
    basis = view_basis(theta, cy, dist)
    image = background.copy()

    camera_lines = to_cam(ROOM_LINES.reshape(-1, 3), *basis).reshape(-1, 2, 3)
    lines, visible = clip_lines(camera_lines)
    projected = np.round(project(lines[visible].reshape(-1, 3))).astype(np.int32).reshape(-1, 2, 2)
    for xy, index in zip(projected, np.flatnonzero(visible)):
        cv2.line(image, tuple(xy[0]), tuple(xy[1]), ROOM_COLORS[index], 1, cv2.LINE_AA)

    xy = np.round(project(to_cam(OBJECT_POINTS, *basis))).astype(np.int32)
    cv2.fillPoly(image, [xy[8:12]], (35, 35, 35))
    visible_faces = np.einsum('ij,ij->i', NORMALS, basis[0] - FACE_CENTERS) > 1e-8
    for index in np.flatnonzero(visible_faces):
        cv2.fillConvexPoly(image, xy[list(FACES[index])], FACE_COLORS[index], cv2.LINE_AA)
    for (a, b), adjacent in zip(EDGES, EDGE_FACES):
        if visible_faces[adjacent].any():
            cv2.line(image, tuple(xy[a]), tuple(xy[b]), EDGE_C, 2, cv2.LINE_AA)
    return image


def make_bg():
    fraction = np.linspace(0, 1, H).reshape(H, 1, 1)
    return (np.array(BG_TOP) * (1 - fraction) + np.array(BG_BOT) * fraction
            ).astype(np.uint8).repeat(W, axis=1)
