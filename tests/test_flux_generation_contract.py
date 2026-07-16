from __future__ import annotations

import importlib.util
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/flux-image-generation/scripts/run_generation.py"


def _load_generation_module():
    spec = importlib.util.spec_from_file_location("flux_run_generation", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FluxGenerationContractTests(unittest.TestCase):
    def test_download_reencodes_png_payload_to_jpeg_output(self) -> None:
        module = _load_generation_module()
        payload = BytesIO()
        Image.new("RGB", (12, 8), color=(20, 40, 60)).save(payload, format="PNG")

        class Response:
            content = payload.getvalue()
            headers = {"Content-Type": "image/png"}

            @staticmethod
            def raise_for_status() -> None:
                return None

        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "generated.jpg"
            with patch.object(module.requests, "get", return_value=Response()):
                module.download_image("https://fixture.invalid/image", output)

            with Image.open(output) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, (12, 8))


if __name__ == "__main__":
    unittest.main()
