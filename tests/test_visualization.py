import os
import tempfile
import unittest
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "squarenet_mpl"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from pymatgen.core import Lattice, Structure

    from squarenet.detect import find_square_net_planes
    from squarenet.visualization import (
        VisualizationError,
        _dedupe_edges,
        _tile_projected_points,
        plot_adjacent_plane_environment,
        plot_candidates_per_material,
        plot_detection_summary,
        plot_material_layer_summary,
        plot_missingness,
        plot_pass_fail_counts,
        plot_projected_layer,
        plot_score_distribution,
        plot_score_vs_environment,
        plot_site_geometry,
        save_figure,
        select_representative_site,
    )
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"optional visualization dependency is not installed: {exc.name}") from exc


def _structure(a=3.0, b=3.0):
    return Structure(
        Lattice.from_parameters(a, b, 6.0, 90, 90, 90),
        ["Si", "S", "S"],
        [[0, 0, 0.5], [0, 0, 0.25], [0, 0, 0.75]],
    )


def _detect(structure, preserve=True):
    return find_square_net_planes(
        structure,
        axes=("c",),
        species=("Si",),
        compute_crystalnn_features=False,
        enforce_no_out_of_plane_same_species_bonds=False,
        preserve_visualization_data=preserve,
    )[0]


class VisualizationTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_noninteractive_backend(self):
        self.assertIn("agg", matplotlib.get_backend().lower())

    def test_detector_preserves_projected_diagnostics_for_positive_example(self):
        result = _detect(_structure())

        self.assertTrue(result.passes)
        self.assertIsNotNone(result.visualization_data)
        self.assertEqual(result.visualization_data.projected_coordinates.shape, (1, 2))
        self.assertGreaterEqual(len(result.visualization_data.selected_neighbor_edges), 4)

    def test_negative_distorted_example_fails_and_plots(self):
        structure = _structure(a=3.0, b=4.5)
        result = _detect(structure)

        self.assertFalse(result.passes)
        fig, axes = plot_detection_summary(structure, result)

        self.assertEqual(set(axes), {"plane_3d", "projected_layer", "site_geometry", "score_components"})
        self.assertEqual(len(fig.axes), 4)

    def test_periodic_tiling_helper(self):
        lattice = np.eye(3)
        basis = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

        tiled, origins, offsets = _tile_projected_points(np.array([[0.0, 0.0]]), lattice, "c", basis, (3, 3))

        self.assertEqual(tiled.shape, (9, 2))
        self.assertEqual(origins.tolist(), [0] * 9)
        self.assertIn((-1, -1, 0), [tuple(x) for x in offsets.tolist()])
        self.assertIn((1, 1, 0), [tuple(x) for x in offsets.tolist()])

    def test_duplicate_edge_removal(self):
        edges = [
            {"start": np.array([0.0, 0.0]), "end": np.array([1.0, 0.0])},
            {"start": np.array([1.0, 0.0]), "end": np.array([0.0, 0.0])},
        ]

        self.assertEqual(len(_dedupe_edges(edges)), 1)

    def test_representative_site_selection_and_geometry_plot(self):
        result = _detect(_structure())

        site = select_representative_site(result, "median")
        fig, ax = plot_site_geometry(result, site)

        self.assertEqual(site, 0)
        self.assertGreater(len(ax.patches), 0)
        self.assertEqual(len(fig.axes), 1)

    def test_graceful_projected_plot_without_optional_diagnostics(self):
        structure = _structure()
        result = _detect(structure, preserve=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig, ax = plot_projected_layer(structure, result)

        self.assertIsNone(result.visualization_data)
        self.assertEqual(len(fig.axes), 1)
        self.assertGreaterEqual(len(ax.collections), 1)
        self.assertTrue(any("diagnostics" in str(w.message) for w in caught))

    def test_positive_projected_and_adjacent_environment_plots(self):
        structure = _structure()
        result = _detect(structure)

        fig1, ax1 = plot_projected_layer(structure, result, annotate_distances=True, annotate_indices=True)
        fig2, ax2 = plot_adjacent_plane_environment(structure, result)

        self.assertGreater(len(ax1.lines), 0)
        self.assertGreater(len(ax2.collections), 0)
        self.assertEqual(len(fig1.axes), 1)
        self.assertEqual(len(fig2.axes), 1)

    def test_dataset_level_plots(self):
        materials = pd.DataFrame(
            {
                "material_id": ["pos", "neg"],
                "has_any_pass": [True, False],
                "dominant_has_pass": [True, False],
                "n_layers_total": [2, 1],
            }
        )
        layers = pd.DataFrame(
            {
                "material_id": ["pos", "pos", "neg"],
                "axis": ["c", "a", "c"],
                "species": ["Si", "Si", "Si"],
                "plane_center_frac": [0.5, 0.25, 0.5],
                "mean_score": [1.0, 0.4, 0.05],
                "passes2": [True, False, False],
                "n_sites": [1, 1, 1],
                "nn_intra_min": [3.0, 3.1, 3.0],
                "min_adj_dist_any_atom": [2.0, 1.5, 1.0],
            }
        )

        figs = [
            plot_pass_fail_counts(materials)[0],
            plot_score_distribution(layers)[0],
            plot_candidates_per_material(materials)[0],
            plot_score_vs_environment(layers, color_by="passes2")[0],
            plot_missingness(layers, columns=["mean_score", "nn_intra_min"])[0],
            plot_material_layer_summary(layers, "pos", pass_column="passes2")[0],
        ]

        self.assertTrue(all(len(fig.axes) >= 1 for fig in figs))

    def test_missing_dataframe_columns_raise_meaningful_error(self):
        with self.assertRaises(VisualizationError):
            plot_score_vs_environment(pd.DataFrame({"mean_score": [1.0]}))

    def test_save_figure_png_and_svg(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        with tempfile.TemporaryDirectory() as tmp:
            png = save_figure(fig, os.path.join(tmp, "figure.png"))
            svg = save_figure(fig, os.path.join(tmp, "figure.svg"))
            self.assertGreater(os.path.getsize(png), 0)
            self.assertGreater(os.path.getsize(svg), 0)
            with self.assertRaises(FileExistsError):
                save_figure(fig, png)


if __name__ == "__main__":
    unittest.main()
