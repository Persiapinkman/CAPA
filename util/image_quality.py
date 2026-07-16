from __future__ import annotations

from pathlib import Path

from PIL import Image


def difference_hash(path: str | Path, *, size: int = 16) -> tuple[bool, ...]:
    with Image.open(path) as image:
        resized = image.convert("L").resize(
            (size + 1, size), Image.Resampling.LANCZOS
        )
        pixels = resized.load()
        return tuple(
            bool(pixels[x, y] > pixels[x + 1, y])
            for y in range(size)
            for x in range(size)
        )


def normalized_hash_distance(left: tuple[bool, ...], right: tuple[bool, ...]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Image hashes must have the same non-zero length")
    return sum(a != b for a, b in zip(left, right)) / len(left)


def audit_image_diversity(
    reference_path: str | Path,
    generated_paths: list[str | Path],
    *,
    min_reference_distance: float = 0.12,
    min_pairwise_distance: float = 0.08,
) -> dict:
    reference = Path(reference_path).resolve()
    generated = [Path(path).resolve() for path in generated_paths]
    reference_hash = difference_hash(reference)
    generated_hashes = [difference_hash(path) for path in generated]
    from_reference = [
        {
            "image": path.name,
            "distance": round(
                normalized_hash_distance(reference_hash, generated_hash), 6
            ),
        }
        for path, generated_hash in zip(generated, generated_hashes)
    ]
    pairwise: list[dict] = []
    for left_idx, left_path in enumerate(generated):
        for right_idx in range(left_idx + 1, len(generated)):
            pairwise.append(
                {
                    "left": left_path.name,
                    "right": generated[right_idx].name,
                    "distance": round(
                        normalized_hash_distance(
                            generated_hashes[left_idx], generated_hashes[right_idx]
                        ),
                        6,
                    ),
                }
            )
    reference_passed = all(
        row["distance"] >= min_reference_distance for row in from_reference
    )
    pairwise_passed = all(
        row["distance"] >= min_pairwise_distance for row in pairwise
    )
    warnings: list[str] = []
    if not reference_passed:
        warnings.append("one or more generated images are near-duplicates of the reference")
    if not pairwise_passed:
        warnings.append("generated images are insufficiently diverse from each other")
    return {
        "method": "difference_hash_16x16",
        "passed": bool(generated and reference_passed and pairwise_passed),
        "thresholds": {
            "min_reference_distance": min_reference_distance,
            "min_pairwise_distance": min_pairwise_distance,
        },
        "from_reference": from_reference,
        "pairwise_generated": pairwise,
        "content_compliance_checked": False,
        "warnings": warnings,
    }
