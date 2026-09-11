"""Unit tests for Krea profile and image metadata helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from settings_utils import (
    APP_ID,
    build_settings,
    extract_image_settings,
    normalize_custom_loras,
    parse_settings_text,
    validate_custom_base_model,
    validate_custom_lora,
    write_png_metadata,
)


class SettingsTests(unittest.TestCase):
    def settings(self):
        return build_settings(
            mode="edit",
            prompt="",
            edit_prompt="change the coat to blue",
            width=1024,
            height=768,
            target_megapixels=1.4,
            grounding_px=768,
            ref_boost=1.0,
            ref_boost_a=1.0,
            steps=8,
            cfg=1.0,
            sampler_name="euler",
            scheduler="beta",
            seed=2,
            randomize_seed=False,
            gen_budget=0,
            effective_seed=2,
            base_model="pornmasterKrea2_v2TurboInt8.safetensors",
            catalog_loras=[{"hf_filename": "slider.safetensors", "weight": 0.8}],
            custom_loras=[{
                "repo_id": "org/repo",
                "filename": "custom.safetensors",
                "revision": "main",
                "weight": 0.5,
            }],
        )

    def test_custom_base_model_is_preserved_in_settings(self):
        settings = build_settings(
            mode="text2image",
            prompt="a landscape",
            edit_prompt="",
            width=1024,
            height=1024,
            target_megapixels=1.4,
            grounding_px=768,
            ref_boost=1.0,
            ref_boost_a=1.0,
            steps=8,
            cfg=1.0,
            sampler_name="euler",
            scheduler="beta",
            seed=2,
            randomize_seed=False,
            gen_budget=0,
            base_model="__custom_huggingface_base_model__",
            custom_base_model={
                "repo_id": "org/repo",
                "filename": "models/base.safetensors",
                "revision": "main",
            },
        )
        self.assertEqual(settings["custom_base_model"]["repo_id"], "org/repo")
        self.assertEqual(settings["custom_base_model"]["filename"], "models/base.safetensors")

    def test_custom_base_model_validation_blocks_unsafe_paths(self):
        valid = validate_custom_base_model(["org/repo", "base.safetensors", "main"])
        self.assertEqual(valid["filename"], "base.safetensors")
        with self.assertRaises(ValueError):
            validate_custom_base_model({
                "repo_id": "org/repo",
                "filename": "../escape.safetensors",
            })

    def test_png_metadata_round_trip(self):
        settings = self.settings()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            destination = Path(directory) / "output.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(source)
            write_png_metadata(source, destination, settings)
            loaded, warnings = extract_image_settings(destination)
        self.assertEqual(loaded["app"], APP_ID)
        self.assertEqual(loaded["edit_prompt"], "change the coat to blue")
        self.assertEqual(loaded["ref_boost"], 1.0)
        self.assertEqual(loaded["base_model"], "pornmasterKrea2_v2TurboInt8.safetensors")
        self.assertEqual(loaded["catalog_loras"][0]["weight"], 0.8)
        self.assertEqual(loaded["custom_loras"][0]["repo_id"], "org/repo")
        self.assertEqual(warnings, [])

    def test_json_profile_round_trip(self):
        parsed, warnings = parse_settings_text(__import__("json").dumps(self.settings()))
        self.assertEqual(warnings, [])
        self.assertEqual(parsed["mode"], "edit")
        self.assertEqual(parsed["grounding_px"], 768)
        self.assertEqual(parsed["custom_loras"][0]["filename"], "custom.safetensors")

    def test_a1111_parameter_text(self):
        parsed, warnings = parse_settings_text(
            "a portrait\nSteps: 8, CFG scale: 1, Sampler: euler, Seed: 7, Size: 512x768"
        )
        self.assertEqual(warnings, [])
        self.assertEqual(parsed["prompt"], "a portrait")
        self.assertEqual(parsed["width"], 512)
        self.assertEqual(parsed["effective_seed"], 7)

    def test_custom_lora_validation_blocks_unsafe_rows(self):
        valid = validate_custom_lora(["org/repo", "sub/style.safetensors", "main", 0.8])
        self.assertEqual(valid["filename"], "sub/style.safetensors")
        normalized, warnings = normalize_custom_loras([
            ["", "", "", 0],
            ["org/repo", "../escape.safetensors", "", 1],
        ])
        self.assertEqual(normalized, [])
        self.assertEqual(len(warnings), 1)
        with self.assertRaises(ValueError):
            validate_custom_lora({
                "repo_id": "not-a-repo",
                "filename": "style.safetensors",
                "weight": 0.5,
            })


if __name__ == "__main__":
    unittest.main()
