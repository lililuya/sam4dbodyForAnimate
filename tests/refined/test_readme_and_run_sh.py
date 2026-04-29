import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ReadmeAndRunScriptContractTest(unittest.TestCase):
    def test_run_sh_exists_with_expected_subcommands(self):
        run_sh_path = os.path.join(ROOT, "run.sh")
        self.assertTrue(os.path.isfile(run_sh_path), "run.sh should exist at the repository root")

        with open(run_sh_path, "r", encoding="utf-8") as handle:
            content = handle.read()

        self.assertIn("detect-debug", content)
        self.assertIn("offline-refined", content)
        self.assertIn("offline-refined-batch", content)
        self.assertIn("cache-4d", content)

    def test_readme_documents_recommended_commands_and_wrappers(self):
        readme_path = os.path.join(ROOT, "README.md")
        with open(readme_path, "r", encoding="utf-8") as handle:
            content = handle.read()

        self.assertIn("## Recommended Commands", content)
        self.assertIn("scripts/debug_human_detection.py", content)
        self.assertIn("./run.sh detect-debug", content)
        self.assertIn("./run.sh offline-refined-batch", content)
        self.assertIn("./run.sh cache-4d", content)


if __name__ == "__main__":
    unittest.main()
