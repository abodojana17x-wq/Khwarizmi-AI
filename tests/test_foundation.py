import unittest
from pathlib import Path
from unittest.mock import patch

from rafig.app import main
from rafig.config import Settings, get_project_paths
from rafig.rafig import Rafiq


class FoundationTests(unittest.TestCase):
    def test_settings_defaults(self) -> None:
        settings = Settings()
        self.assertEqual(settings.project_name, "RAFIQ")
        self.assertEqual(settings.version, "0.1.0")
        self.assertTrue(settings.offline_mode)

    def test_project_paths_created(self) -> None:
        settings = Settings(project_root=Path("/tmp/rafig-test"))
        paths = get_project_paths(settings)
        self.assertTrue(paths["root"].exists())

    def test_rafig_start_and_shutdown(self) -> None:
        engine = Rafiq(settings=Settings(project_root=Path("/tmp/rafig-test")))
        engine.start()
        self.assertTrue(engine.started)
        self.assertIsNotNone(engine.diagnostics)
        engine.run()
        engine.shutdown()
        self.assertFalse(engine.started)

    def test_main_entrypoint_returns_zero(self) -> None:
        with patch("rafig.app.Rafiq.run", lambda self: None):
            self.assertEqual(main([]), 0)


if __name__ == "__main__":
    unittest.main()
