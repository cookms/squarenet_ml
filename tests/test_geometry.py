import unittest

try:
    import numpy as np
    from squarenet.geometry import plane_frame_from_axis, unit
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"optional test dependency is not installed: {exc.name}") from exc


class GeometryTests(unittest.TestCase):
    def test_unit_returns_zero_vector_unchanged(self):
        np.testing.assert_array_equal(unit(np.array([0.0, 0.0, 0.0])), np.array([0.0, 0.0, 0.0]))

    def test_plane_frame_from_axis_returns_orthonormal_frame(self):
        lattice = np.eye(3)

        e1, e2, normal = plane_frame_from_axis(lattice, "c")

        self.assertAlmostEqual(float(np.linalg.norm(e1)), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(e2)), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0)
        self.assertAlmostEqual(float(np.dot(e1, e2)), 0.0)
        self.assertAlmostEqual(float(np.dot(e1, normal)), 0.0)
        self.assertAlmostEqual(float(np.dot(e2, normal)), 0.0)

    def test_plane_frame_rejects_unknown_axis(self):
        with self.assertRaises(ValueError):
            plane_frame_from_axis(np.eye(3), "z")


if __name__ == "__main__":
    unittest.main()
