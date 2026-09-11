---
title: Krea 2 Turbo Image Generator
emoji: 🖌️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.44.1
python_version: '3.12'
app_file: app.py
pinned: true
short_description: Krea 2 Turbo text2image and image editing
license: other
---

# Krea 2 Turbo Image Generator

This Space runs Krea 2 Turbo through headless ComfyUI execution. It provides:

- Text-to-image generation.
- Instruction-based image editing with an optional second reference image.
- Krea 2 Identity Edit v1.2 conditioning for the edit mode.
- Two selectable custom Krea 2 Turbo checkpoints.
- An option to download and use a custom base checkpoint from any accessible Hugging Face repository.
- A catalog of Krea 2 LoRAs with signed weight controls and search/filtering.
- Custom LoRAs from any accessible Hugging Face repository.
- Seed, sampling, grounding, and reference-fidelity controls.
- PNG metadata and portable JSON settings profiles.

## Models

The text encoder and VAE are downloaded from the ungated [Comfy-Org/Krea-2 mirror](https://huggingface.co/Comfy-Org/Krea-2).
The base-model selector provides both checkpoints from [mpasila/Krea-2-Models](https://huggingface.co/mpasila/Krea-2-Models):

| File | Role |
|------|------|
| `museByStableYogi_v25EXTENDEDTURBO.safetensors` | Muse By Stable Yogi Krea2 diffusion model |
| `pornmasterKrea2_v2TurboInt8.safetensors` | PornMaster Krea 2 Turbo diffusion model |
| `qwen3vl_4b_fp8_scaled.safetensors` | Krea 2 Qwen3-VL text encoder |
| `qwen_image_vae.safetensors` | Krea 2 VAE |

Choose **Custom Hugging Face base model** in the base checkpoint selector to provide a repository,
relative model filename, and optional revision. Supported files use `.safetensors`, `.ckpt`, `.pt`,
or `.bin` extensions. The checkpoint is downloaded lazily into ComfyUI's managed diffusion-model
directory and cached in a stable repository/revision namespace. Private repositories require the
`HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` environment variable.

Edit mode also downloads `krea2_identity_edit_v1_2.safetensors` from
[conradlocke/krea2-identity-edit](https://huggingface.co/conradlocke/krea2-identity-edit).
The edit nodes come from [ComfyUI-Krea2Edit](https://github.com/lbouaraba/comfyui-krea2edit).

## LoRAs

The built-in catalog is loaded from [`krea2_loras.json`](../krea2_loras.json:1) and downloaded
lazily from [mpasila/Krea-2-LoRAs](https://huggingface.co/mpasila/Krea-2-LoRAs). A weight of zero
disables an entry. Enabled catalog entries are applied in the order listed in the catalog, then
custom rows are applied in their table order. Signed weights are supported for slider LoRAs.

Custom Hugging Face LoRAs use one table row per file:

| Column | Example |
|------|------|
| `repo_id` | `org/repository` |
| `filename` | `subfolder/style.safetensors` |
| `revision` | `main` or a commit/tag (optional) |
| `weight` | `0.8` |

Only relative model files with supported extensions are accepted. Private repositories require the
`HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` environment variable. Catalog and custom LoRAs are cached
under ComfyUI’s managed LoRA directory and are downloaded only when their weight is non-zero.
In edit mode, the mandatory Krea 2 Identity Edit adapter remains enabled before the user LoRA
chain; it is not included in the selectable catalog.

All model and adapter licenses remain applicable. Review the Krea 2 Community License Agreement
and the adapter repository’s responsible-use guidance before deploying this Space. Do not use the
identity-edit workflow for non-consensual or harmful impersonation of real people.

## Usage

Choose **text-to-image** for a new image, or **edit** to provide a primary source image and an
optional second reference. In edit mode, image order matters: the primary image is the scene/source
and the second image is the subject or additional reference.

The edit workflow uses `fit` reference geometry. The target megapixel setting controls the output
size while preserving the primary image’s aspect ratio. Lower grounding values usually follow the
edit instruction more strongly; higher values usually preserve reference identity more strongly.

Generated PNGs include a `krea_settings` JSON object and a readable `parameters` field. Source image
bytes are never stored in exported settings profiles. Profiles also preserve the selected checkpoint,
custom base-model repository/file/revision, catalog weights, and custom Hugging Face LoRA rows.
