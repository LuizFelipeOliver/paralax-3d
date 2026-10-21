import unittest
from unittest.mock import Mock

import numpy as np

from controls import EMA, REF_FACE, YAW_GAIN, face_position, head_to_cam, smooth_position
from scene import (CUBE_C, W, H, EDGES,
                   FACE_CENTERS, NORMALS, clip_line, clip_lines, make_bg,
                   project, render_scene, to_cam, view_basis)
from tracking import detect
from wall_scene import (project_window, render_wall, head_to_window, POLYGON_COLORS,
                        WALL_FRONT, WALL_SIDE, WALL_EDGE, draw_wall)


class PythonTests(unittest.TestCase):
    def test_wall_window_is_an_opening_not_a_painted_rectangle(self):
        image = np.full((H, W, 3), (1, 2, 3), dtype=np.uint8)
        camera = np.array([0, 0, 7])
        draw_wall(image, camera)
        points = project_window([[-2, 0, -2], [-3.5, 0, -2]], camera).round().astype(int)
        window, solid = [image[y, x] for x, y in points]
        np.testing.assert_equal(window, [1, 2, 3])
        self.assertFalse(np.array_equal(solid, [1, 2, 3]))

    def test_window_projection_anchors_screen_and_shifts_background(self):
        screen_points = np.array([[0, 0, 0], [-1, 1, 0], [1, -1, 0]])
        baseline = project_window(screen_points, [0, 0, 7])
        moved = project_window(screen_points, [3, 1, 7])
        np.testing.assert_allclose(moved, baseline)
        background = np.array([[-1.2, 0.1, -4]])
        self.assertGreater(project_window(background, [3, 0, 7])[0, 0],
                           project_window(background, [0, 0, 7])[0, 0])
        self.assertGreater(head_to_window(0.5, 0, REF_FACE)[0], 0)

    def test_leaning_reveals_hidden_object_and_wall_has_thickness(self):
        background = make_bg()
        center = render_wall(0, 0, REF_FACE, background)
        right = render_wall(0.65, 0, REF_FACE, background)
        self.assertFalse(np.array_equal(center[:, :W // 2], right[:, :W // 2]))
        camera = head_to_window(0.65, 0, REF_FACE)
        edge = project_window(WALL_EDGE, camera)
        self.assertTrue(np.all(edge[:, 0] > W // 2))
        side = project_window(WALL_SIDE, camera)
        self.assertGreater(abs(side[1, 0] - side[0, 0]), 10)
        self.assertGreater(np.ptp(WALL_SIDE[:, 2]), 0)
        self.assertEqual(np.ptp(WALL_FRONT[:, 2]), 0)
        colors = np.array(POLYGON_COLORS[:6])
        def count_object(frame):
            return np.any(np.all(frame[:, W // 2:, None, :] == colors, axis=-1), axis=-1).sum()
        self.assertGreater(count_object(right), count_object(center))
        for dx in (-1, 1):
            for dy in (-1, 1):
                frame = render_wall(dx, dy, 1, background)
                self.assertEqual(frame.shape, (H, W, 3))

    def test_position_and_original_smoothing(self):
        center = face_position([76, 55, 48, 40], 200, 150)
        np.testing.assert_allclose(center, [0, 0, REF_FACE])
        right = face_position([86, 55, 48, 40], 200, 150)
        moved = smooth_position(center, right)
        self.assertAlmostEqual(moved[0], 0.1 * EMA)
        self.assertGreater(head_to_cam(*moved)[0], 0)
        self.assertLess(head_to_cam(-moved[0], 0, REF_FACE)[0], 0)
        self.assertEqual(head_to_cam(5, 5, REF_FACE)[0], YAW_GAIN)

    def test_detector_boxes_match_for_haar_and_yunet(self):
        image = np.zeros((150, 200, 3), np.uint8)
        cascade = Mock()
        cascade.detectMultiScale.return_value = [[1, 2, 30, 40]]
        haar = detect(image, None, cascade)
        yunet = Mock()
        yunet.detect.return_value = (1, np.array([[1, 2, 30, 40] + [0] * 11]))
        neural = detect(image, yunet, None)
        np.testing.assert_equal(haar, neural)
        np.testing.assert_equal(haar[0] * 1.5, [1.5, 3, 45, 60])
        cascade.detectMultiScale.return_value = ()
        self.assertEqual(detect(image, None, cascade).shape, (0, 4))
        yunet.detect.return_value = (0, None)
        self.assertEqual(detect(image, yunet, None).shape, (0, 4))

    def test_near_plane_clipping(self):
        points = clip_line([[1, 0, 1], [1, 0, -2]])
        self.assertAlmostEqual(points[0, 2], -0.3)
        self.assertTrue(np.isfinite(project(points)).all())
        self.assertIsNone(clip_line([[0, 0, 1], [1, 1, 2]]))

    def test_batch_clipping_preserves_visible_segments(self):
        lines = np.array([[[1, 0, 1], [1, 0, -2]],
                          [[0, 0, -2], [1, 1, -3]],
                          [[0, 0, 1], [1, 1, 2]],
                          [[1, 0, -2], [1, 0, 1]]], dtype=float)
        clipped, visible = clip_lines(lines)
        np.testing.assert_equal(visible, [True, True, False, True])
        np.testing.assert_equal(clipped[1], lines[1])
        self.assertTrue(np.all(-clipped[visible, :, 2] >= 0.3 - 1e-9))
        self.assertTrue(np.isfinite(project(clipped[visible].reshape(-1, 3))).all())

    def test_cube_geometry_and_vertical_controls(self):
        self.assertEqual(len(EDGES), 12)
        for direction, face_index in ((0.5, 5), (-0.5, 4)):
            theta, height, distance = head_to_cam(0, direction, REF_FACE)
            cam, *_ = view_basis(theta, height, distance)
            self.assertGreater(NORMALS[face_index] @ (cam - FACE_CENTERS[face_index]), 0)

    def test_camera_keeps_cube_centered_and_render_is_static(self):
        background = make_bg()
        for theta in (-1.57, 0, 1.57):
            for height, distance in ((-7, 4), (7, 4), (-7, 11), (7, 11)):
                basis = view_basis(theta, height, distance)
                xy = project(to_cam([CUBE_C], *basis))
                np.testing.assert_allclose(xy[0], [W / 2, H / 2])
                frame = render_scene(theta, height, distance, background)
                self.assertEqual(frame.shape, (H, W, 3))
                np.testing.assert_equal(frame, render_scene(theta, height, distance, background))


if __name__ == "__main__":
    unittest.main()
