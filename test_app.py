"""Pure runtime tests; no ComfyUI clone or model download is required."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["KREA_SKIP_STARTUP"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

import app  # noqa: E402


class WorkflowTests(unittest.TestCase):
    def test_text_to_image_loader_and_prompt_injection(self):
        workflow = app._t2i_workflow()
        app._inject_t2i(
            workflow,
            prompt="a blue glass sculpture",
            width=1024,
            height=768,
            steps=8,
            cfg=1.0,
            sampler="euler",
            scheduler="beta",
            seed=42,
        )
        self.assertEqual(workflow["1"]["inputs"]["unet_name"], app.CUSTOM_KREA_FILE)
        self.assertEqual(app.BASE_MODELS[app.CUSTOM_KREA_FILE]["repo"], "mpasila/Krea-2-Models")
        self.assertIn(app.MUSE_KREA_FILE, app.BASE_MODELS)
        self.assertEqual(workflow["2"]["inputs"]["clip_name"], "qwen3vl_4b_fp8_scaled.safetensors")
        self.assertEqual(workflow["5"]["inputs"]["text"], "a blue glass sculpture")
        self.assertEqual(workflow["7"]["inputs"]["height"], 768)
        self.assertEqual(workflow["8"]["inputs"]["seed"], 42)

    def test_edit_workflow_has_mandatory_adapter_and_optional_second_reference(self):
        workflow = app._edit_workflow(True)
        self.assertEqual(workflow["6"]["class_type"], "LoraLoaderModelOnly")
        self.assertTrue(workflow["6"]["inputs"]["lora_name"].endswith(app.IDENTITY_FILE))
        self.assertIn("source_latent_b", workflow["9"]["inputs"])
        self.assertIn("image_b", workflow["10"]["inputs"])

    def test_edit_workflow_without_second_reference_omits_b_inputs(self):
        workflow = app._edit_workflow(False)
        self.assertNotIn("2", workflow)
        self.assertNotIn("source_latent_b", workflow["9"]["inputs"])
        self.assertNotIn("source_image_b", workflow["9"]["inputs"])
        self.assertNotIn("image_b", workflow["10"]["inputs"])

    def test_text_to_image_supports_selected_model_and_ordered_loras(self):
        workflow = app._t2i_workflow(app.MUSE_KREA_FILE)
        app._inject_t2i(
            workflow,
            prompt="a portrait",
            width=1024,
            height=1024,
            steps=8,
            cfg=1.0,
            sampler="euler",
            scheduler="beta",
            seed=7,
            enabled_loras=[("krea2/first.safetensors", 0.5), ("huggingface/second.safetensors", -0.25)],
        )
        self.assertEqual(workflow["1"]["inputs"]["unet_name"], app.MUSE_KREA_FILE)
        self.assertEqual(workflow["user_lora_0"]["inputs"]["model"], ["1", 0])
        self.assertEqual(workflow["user_lora_1"]["inputs"]["model"], ["user_lora_0", 0])
        self.assertEqual(workflow["4"]["inputs"]["model"], ["user_lora_1", 0])
        self.assertEqual(workflow["5"]["inputs"]["clip"], ["user_lora_1", 1])

    def test_custom_base_model_download_uses_managed_namespace(self):
        original_models = app.MODELS
        with tempfile.TemporaryDirectory() as directory:
            app.MODELS = Path(directory) / "models"

            def fake_download(**kwargs):
                destination = Path(kwargs["local_dir"]) / kwargs["filename"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"test")
                return str(destination)

            try:
                with patch.object(app, "hf_hub_download", side_effect=fake_download):
                    relative = app._ensure_custom_base_model({
                        "repo_id": "org/repo",
                        "filename": "sub/base.safetensors",
                        "revision": "main",
                    })
            finally:
                app.MODELS = original_models

        self.assertTrue(relative.startswith("huggingface/"))
        self.assertTrue(relative.endswith("/sub/base.safetensors"))
        workflow = app._t2i_workflow(relative)
        self.assertEqual(workflow["1"]["inputs"]["unet_name"], relative)

    def test_edit_user_loras_follow_identity_adapter(self):
        workflow = app._edit_workflow(False, app.MUSE_KREA_FILE)
        app._inject_edit(
            workflow,
            primary_name="source.png",
            second_name=None,
            width=1024,
            height=1024,
            edit_prompt="change the shirt",
            grounding_px=768,
            ref_boost=1.0,
            ref_boost_a=1.0,
            steps=8,
            cfg=1.0,
            sampler="euler",
            scheduler="beta",
            seed=7,
            enabled_loras=[("krea2/style.safetensors", 1.0)],
        )
        self.assertEqual(workflow["5"]["inputs"]["unet_name"], app.MUSE_KREA_FILE)
        self.assertEqual(workflow["user_lora_0"]["inputs"]["model"], ["6", 0])
        self.assertEqual(workflow["9"]["inputs"]["model"], ["user_lora_0", 0])
        self.assertEqual(workflow["10"]["inputs"]["clip"], ["user_lora_0", 1])
        self.assertEqual(workflow["6"]["class_type"], "LoraLoaderModelOnly")

    def test_custom_lora_download_uses_managed_namespace(self):
        original_root = app.LORA_ROOT
        original_custom_root = app.CUSTOM_LORA_DEST_DIR
        with tempfile.TemporaryDirectory() as directory:
            app.LORA_ROOT = Path(directory) / "loras"
            app.CUSTOM_LORA_DEST_DIR = app.LORA_ROOT / "huggingface"

            def fake_download(**kwargs):
                destination = Path(kwargs["local_dir"]) / kwargs["filename"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"test")
                return str(destination)

            try:
                with patch.object(app, "hf_hub_download", side_effect=fake_download):
                    relative = app._ensure_custom_lora({
                        "repo_id": "org/repo",
                        "filename": "sub/style.safetensors",
                        "revision": "main",
                        "weight": 0.8,
                    })
            finally:
                app.LORA_ROOT = original_root
                app.CUSTOM_LORA_DEST_DIR = original_custom_root

        self.assertTrue(relative.startswith("huggingface/"))
        self.assertTrue(relative.endswith("/sub/style.safetensors"))

    def test_mode_validation(self):
        with self.assertRaises(ValueError):
            app._validate_request("text2image", "", "", None)
        with self.assertRaises(ValueError):
            app._validate_request("edit", "", "change the color", None)
        with self.assertRaises(ValueError):
            app._validate_request("edit", "", "", "source.png")
        app._validate_request("edit", "", "change the color", "source.png")
        app._validate_request("edit", "change the color", "", "source.png")

    def test_gpu_duration_accepts_spaces_positional_callback(self):
        values = [
            "text2image", "prompt", "", None, None,
            1024, 1024, 1.4, 768, 1.0, 1.0,
            8, 1.0, "euler", "beta", 2, False, 0, None,
        ]
        duration = app.get_gpu_duration(*values)
        self.assertGreaterEqual(duration, app.MIN_GPU_SECONDS)

    def test_gpu_duration_accepts_keyword_callback(self):
        duration = app.get_gpu_duration(steps=8, width=1024, height=1024, gen_budget=120)
        self.assertEqual(duration, 120)

    def test_runtime_uses_ram_pressure_executor_configuration(self):
        with open(app.__file__, encoding="utf-8") as source_file:
            source = source_file.read()
        self.assertIn("cache_type=execution.CacheType.RAM_PRESSURE", source)
        self.assertIn('"ram": 2.0', source)
        self.assertIn('"ram_inactive": 8.0', source)


if __name__ == "__main__":
    unittest.main()
