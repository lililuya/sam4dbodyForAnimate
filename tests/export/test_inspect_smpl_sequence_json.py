import json
import os
import shutil
import tempfile
import unittest


class InspectSmplSequenceJsonTests(unittest.TestCase):
    def _write_sample_sequence(self, root_dir: str) -> str:
        input_path = os.path.join(root_dir, "smpl_sequence.json")
        payload = {
            "sample_id": "demo",
            "sample_uuid": "uuid-demo",
            "track_id": 1,
            "source_path": "/dataset/demo.mp4",
            "frame_count": 2,
            "records": [
                {
                    "frame_stem": "00000000",
                    "person_output": {
                        "bbox": [10.0, 20.0, 30.0, 40.0],
                        "pred_keypoints_2d": [[1.0, 2.0], [3.0, 4.0]],
                        "pred_vertices": [[0.0, 0.0, 0.0] for _ in range(32)],
                    },
                },
                {
                    "frame_stem": "00000001",
                    "person_output": {
                        "bbox": [11.0, 21.0, 31.0, 41.0],
                        "pred_keypoints_2d": [[5.0, 6.0], [7.0, 8.0]],
                        "pred_vertices": [[1.0, 1.0, 1.0] for _ in range(32)],
                    },
                },
            ],
        }
        with open(input_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return input_path

    def test_analyze_smpl_sequence_fields_ranks_largest_field_first(self):
        from scripts.inspect_smpl_sequence_json import analyze_smpl_sequence_fields

        temp_dir = tempfile.mkdtemp(prefix="inspect_smpl_sequence_")
        try:
            input_path = self._write_sample_sequence(temp_dir)

            summary = analyze_smpl_sequence_fields(input_path)

            self.assertEqual(summary["record_count"], 2)
            self.assertEqual(summary["inspected_record_count"], 2)
            self.assertGreater(summary["total_field_bytes"], 0)
            self.assertEqual(summary["fields"][0]["field"], "pred_vertices")
            self.assertEqual(summary["fields"][0]["count"], 2)
            self.assertEqual(summary["fields"][0]["sample_shape"], [32, 3])
            field_sizes = {item["field"]: item["total_bytes"] for item in summary["fields"]}
            self.assertIn("bbox", field_sizes)
            self.assertIn("pred_keypoints_2d", field_sizes)
            self.assertLess(field_sizes["bbox"], field_sizes["pred_vertices"])
            self.assertLess(field_sizes["pred_keypoints_2d"], field_sizes["pred_vertices"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_main_writes_output_json_summary(self):
        from scripts.inspect_smpl_sequence_json import main

        temp_dir = tempfile.mkdtemp(prefix="inspect_smpl_sequence_cli_")
        try:
            input_path = self._write_sample_sequence(temp_dir)
            output_path = os.path.join(temp_dir, "field_summary.json")

            exit_code = main(
                [
                    "--input_json",
                    input_path,
                    "--output_json",
                    output_path,
                    "--top_n",
                    "2",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.isfile(output_path))
            with open(output_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["fields"][0]["field"], "pred_vertices")
            self.assertEqual(len(payload["fields"]), 2)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
