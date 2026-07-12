import os
import unittest
from unittest.mock import patch

from example_run import build_config, resolve_api_key


class ExampleRunConfigTests(unittest.TestCase):
    @patch.dict(os.environ, {"MP_API_KEY": "preferred-key", "MAPI_KEY": "fallback-key"}, clear=True)
    def test_resolve_api_key_prefers_mp_api_key(self):
        self.assertEqual(resolve_api_key(), "preferred-key")

    @patch.dict(os.environ, {"MAPI_KEY": "fallback-key"}, clear=True)
    def test_resolve_api_key_supports_legacy_mapi_key(self):
        self.assertEqual(resolve_api_key(), "fallback-key")

    @patch.dict(os.environ, {}, clear=True)
    def test_resolve_api_key_returns_none_when_unset(self):
        self.assertIsNone(resolve_api_key())

    def test_build_config_uses_explicit_values(self):
        cfg = build_config(
            api_key="test-key",
            material_ids_path="example-mpids.txt",
            out_dir="example-output",
            limit=3,
        )

        self.assertEqual(cfg.mp.api_key, "test-key")
        self.assertEqual(cfg.mp.material_ids_path, "example-mpids.txt")
        self.assertEqual(cfg.mp.limit, 3)
        self.assertEqual(cfg.preprocess.structure_source, "processed")
        self.assertEqual(cfg.preprocess.sym_supercell, (3, 3, 3))
        self.assertEqual(cfg.output.out_dir, "example-output")
        self.assertEqual(tuple(cfg.detect.axes), ("c", "a", "b"))
        self.assertEqual(cfg.detect.plane_tol, 0.01)
        self.assertEqual(cfg.detect.k_nn, 9)
        self.assertEqual(cfg.detect.ang_tol_deg, 5.0)
        self.assertEqual(cfg.detect.min_pass_fraction, 0.55)
        self.assertTrue(cfg.detect.enforce_no_out_of_plane_same_species_bonds)
        self.assertTrue(cfg.detect.compute_crystalnn_features)

    def test_build_config_does_not_embed_api_key_placeholder(self):
        cfg = build_config()
        self.assertIsNone(cfg.mp.api_key)


if __name__ == "__main__":
    unittest.main()
