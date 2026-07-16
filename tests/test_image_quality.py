from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from util.image_quality import audit_image_diversity


def _pattern(path: Path, *, vertical: bool) -> None:
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    if vertical:
        draw.rectangle((0, 0, 20, 63), fill="black")
    else:
        draw.rectangle((0, 0, 63, 20), fill="black")
    image.save(path)


class ImageQualityTests(unittest.TestCase):
    def test_duplicate_generation_fails_diversity_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            reference = root / "reference.png"
            duplicate = root / "duplicate.png"
            diverse = root / "diverse.png"
            _pattern(reference, vertical=True)
            _pattern(duplicate, vertical=True)
            _pattern(diverse, vertical=False)

            result = audit_image_diversity(reference, [duplicate, diverse])
            self.assertFalse(result["passed"])
            self.assertEqual(result["from_reference"][0]["distance"], 0.0)
            self.assertTrue(result["warnings"])


if __name__ == "__main__":
    unittest.main()
