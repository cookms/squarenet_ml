import io
import os
import tempfile
import unittest

from squarenet.config import DetectConfig, MPQueryConfig, PipelineConfig, load_pipeline_config
from squarenet.pipeline import _build_run_metadata

try:
    import yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class YamlConfigTests(unittest.TestCase):
    @unittest.skipUnless(HAS_YAML, "PyYAML is not installed")
    def test_loads_uploaded_bytes_and_file_objects(self):
        yaml_bytes = b"""
detect:
  axes: [c]
  score_threshold: 0.72
output:
  out_dir: custom-output
meta:
  experiment: upload-test
"""
        for source in (yaml_bytes, io.BytesIO(yaml_bytes)):
            cfg = load_pipeline_config(source)
            self.assertEqual(cfg.detect.axes, ["c"])
            self.assertEqual(cfg.detect.score_threshold, 0.72)
            self.assertEqual(cfg.output.out_dir, "custom-output")
            self.assertEqual(cfg.meta["experiment"], "upload-test")

    @unittest.skipUnless(HAS_YAML, "PyYAML is not installed")
    def test_loads_yaml_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "detector.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("detect:\n  plane_tol: 0.025\n")
            self.assertEqual(load_pipeline_config(path).detect.plane_tol, 0.025)

    @unittest.skipUnless(HAS_YAML, "PyYAML is not installed")
    def test_rejects_unknown_detector_setting(self):
        with self.assertRaisesRegex(ValueError, "score_treshold"):
            load_pipeline_config(b"detect:\n  score_treshold: 0.8\n")

    def test_metadata_records_effective_settings_and_redacts_secret(self):
        cfg = PipelineConfig(
            mp=MPQueryConfig(api_key="secret"),
            detect=DetectConfig(plane_tol_A=0.03),
        )
        meta = _build_run_metadata(cfg)
        self.assertEqual(meta["mp_query"]["api_key"], "<redacted>")
        self.assertEqual(meta["detector"]["plane_tol_A"], 0.03)
        self.assertEqual(meta["detector_effective"]["plane_tol"], 0.03)
        self.assertEqual(len(meta["config_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
