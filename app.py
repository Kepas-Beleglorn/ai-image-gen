"""Krea 2 Turbo text-to-image and image-editing Hugging Face Space.

The supplied ComfyUI workflows are bundled as the graph references for this
Space.  The runtime uses their supported first-pass/edit paths and replaces
optional UI-heavy stages with a small deterministic API graph so startup does
not depend on face, hand, detailer, or image-saver node packs.
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from typing import Any

import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image

try:
    import spaces
except ImportError:  # pragma: no cover - lets pure unit tests import helpers.
    class _SpacesFallback:
        @staticmethod
        def GPU(**_kwargs):
            def decorate(function):
                return function
            return decorate

    spaces = _SpacesFallback()

from settings_utils import (
    APP_ID,
    PROFILE_SCHEMA_VERSION,
    build_settings,
    extract_image_settings,
    normalize_custom_loras,
    parse_settings_text,
    stable_custom_base_model_namespace,
    stable_custom_lora_namespace,
    validate_custom_base_model,
    validate_custom_lora,
    write_png_metadata,
)


ROOT = pathlib.Path(__file__).resolve().parent
COMFY = ROOT / "ComfyUI"
MODELS = COMFY / "models"
INPUT = COMFY / "input"
OUTPUT = COMFY / "output"
CUSTOM_NODES = COMFY / "custom_nodes"

T2I_SOURCE = ROOT / "lustifyWorkflowsKrea2_krea2.json"
EDIT_SOURCE = ROOT / "lustifyWorkflowsKrea2_krea2Edit.json"

KREA_REPO = "Comfy-Org/Krea-2"
CUSTOM_KREA_REPO = "mpasila/Krea-2-Models"
CUSTOM_KREA_FILE = "pornmasterKrea2_v2TurboInt8.safetensors"
MUSE_KREA_FILE = "museByStableYogi_v25EXTENDEDTURBO.safetensors"
DEFAULT_BASE_MODEL = CUSTOM_KREA_FILE
CUSTOM_BASE_MODEL = "__custom_huggingface_base_model__"
BASE_MODELS = {
    MUSE_KREA_FILE: {
        "label": "Muse By Stable Yogi Krea2",
        "repo": CUSTOM_KREA_REPO,
        "filename": MUSE_KREA_FILE,
    },
    CUSTOM_KREA_FILE: {
        "label": "PornMaster 色情大師-Krea2",
        "repo": CUSTOM_KREA_REPO,
        "filename": CUSTOM_KREA_FILE,
    },
}
IDENTITY_REPO = "conradlocke/krea2-identity-edit"
IDENTITY_FILE = "krea2_identity_edit_v1_2.safetensors"
KREA_EDIT_NODES = "https://github.com/lbouaraba/comfyui-krea2edit.git"
LORA_CATALOG_FILE = "krea2_loras.json"
LORA_HF_REPO = "mpasila/Krea-2-LoRAs"
LORA_DEST_DIR = MODELS / "loras" / "krea2"
LORA_ROOT = MODELS / "loras"
CUSTOM_LORA_DEST_DIR = LORA_ROOT / "huggingface"

DOWNLOADS = [
    (KREA_REPO, "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", MODELS / "text_encoders" / "qwen3vl_4b_fp8_scaled.safetensors", "Qwen3-VL text encoder"),
    (KREA_REPO, "vae/qwen_image_vae.safetensors", MODELS / "vae" / "qwen_image_vae.safetensors", "Qwen image VAE"),
    (IDENTITY_REPO, IDENTITY_FILE, MODELS / "loras" / "krea" / IDENTITY_FILE, "Krea 2 identity-edit adapter"),
]

SAMPLERS = [
    "euler", "euler_ancestral", "euler_a", "dpmpp_2m", "dpmpp_2m_sde",
    "dpmpp_sde", "heun", "lms",
]
SCHEDULERS = ["beta", "normal", "karras", "exponential", "sgm_uniform", "simple"]

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_TARGET_MP = 1.4
MAX_WIDTH = 2048
MAX_HEIGHT = 2048
MAX_TARGET_MP = 4.0
DEFAULT_GROUNDING = 768
DEFAULT_REF_BOOST = 1.0
DEFAULT_STEPS = 8
DEFAULT_CFG = 1.0
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "beta"
DEFAULT_SEED = 2
MIN_GPU_SECONDS = int(os.environ.get("MIN_GPU_SECONDS", "45"))
MAX_GPU_SECONDS = int(os.environ.get("MAX_GPU_SECONDS", "300"))

_comfy_ready = False
_nodes_ready = False
_workflow_cache: dict[str, dict[str, Any]] = {}
_lora_catalog: list[dict[str, Any]] = []
_lora_by_filename: dict[str, dict[str, Any]] = {}


def _run(command: list[str], cwd: pathlib.Path | None = None, check: bool = True) -> None:
    print("[setup]", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=check)


def _pip_install(arguments: list[str]) -> None:
    _run([sys.executable, "-m", "pip", "install", "--no-cache-dir", *arguments], check=False)


def _install_filtered_requirements(path: pathlib.Path) -> None:
    """Install ComfyUI requirements without replacing Space's torch stack."""
    if not path.exists():
        return
    blocked = {"torch", "torchvision", "torchaudio", "transformers", "huggingface-hub", "accelerate"}
    requirements: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        item = raw.strip()
        if not item or item.startswith("#"):
            continue
        package = re.split(r"[<>=!~;\[\s]", item.lower().replace("_", "-"), maxsplit=1)[0]
        if package not in blocked:
            requirements.append(item)
    if requirements:
        _pip_install(requirements)


def _ensure_repo(path: pathlib.Path, url: str) -> None:
    if not path.exists():
        _run(["git", "clone", "--depth", "1", url, str(path)])


def _restore_utils_namespace() -> None:
    """Undo the old root ``utils`` rename if a persistent Space has one."""
    source = COMFY / "utils"
    target = COMFY / "utilities"
    if not source.exists() and target.exists():
        target.rename(source)
    if not source.exists():
        return
    for path in COMFY.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = re.sub(r"\bfrom utilities\b", "from utils", text)
        updated = re.sub(r"\bimport utilities\b", "import utils", updated)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def _ensure_comfy() -> None:
    global _comfy_ready
    if _comfy_ready:
        return
    _ensure_repo(COMFY, "https://github.com/comfyanonymous/ComfyUI.git")
    _install_filtered_requirements(COMFY / "requirements.txt")
    CUSTOM_NODES.mkdir(parents=True, exist_ok=True)
    _ensure_repo(CUSTOM_NODES / "comfyui-krea2edit", KREA_EDIT_NODES)
    _restore_utils_namespace()
    for folder in (
        "diffusion_models",
        "text_encoders",
        "vae",
        "loras/krea",
        "loras/krea2",
        "loras/huggingface",
    ):
        (MODELS / folder).mkdir(parents=True, exist_ok=True)
    INPUT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _comfy_ready = True


def _init_comfy_nodes() -> None:
    global _nodes_ready
    if _nodes_ready:
        return
    comfy_path = str(COMFY)
    sys.path = [item for item in sys.path if item != comfy_path]
    sys.path.insert(0, comfy_path)
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            del sys.modules[name]
    os.chdir(COMFY)
    import execution
    import nodes
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    loop.run_until_complete(nodes.init_extra_nodes())
    _nodes_ready = True


def _download_to_dest(repo: str, filename: str, destination: pathlib.Path, label: str) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    path = pathlib.Path(filename)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    downloaded = pathlib.Path(
        hf_hub_download(
            repo_id=repo,
            filename=path.name,
            subfolder=None if str(path.parent) == "." else str(path.parent),
            local_dir=str(destination.parent),
            token=token,
        )
    )
    if downloaded.resolve() != destination.resolve():
        shutil.move(str(downloaded), str(destination))
    print(f"[models] ready: {label}", flush=True)


def _load_lora_catalog() -> None:
    """Load the shipped Krea LoRA catalog in its documented order."""
    global _lora_catalog, _lora_by_filename
    path = ROOT / LORA_CATALOG_FILE
    if not path.exists():
        path = ROOT.parent / LORA_CATALOG_FILE
    if not path.exists():
        print(f"[lora] {LORA_CATALOG_FILE} not found; catalog disabled", flush=True)
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    catalog: list[dict[str, Any]] = []
    for row in data.get("loras", []):
        if not isinstance(row, dict) or not row.get("hf_filename"):
            continue
        filename = pathlib.PurePosixPath(str(row["hf_filename"]).replace("\\", "/"))
        if filename.is_absolute() or any(part in {"", ".", ".."} for part in filename.parts):
            continue
        catalog.append(dict(row, hf_filename=filename.as_posix()))
    _lora_catalog = catalog
    _lora_by_filename = {str(item["hf_filename"]): item for item in catalog}
    print(f"[lora] loaded {len(_lora_catalog)} Krea LoRAs", flush=True)


def _ensure_base_model(base_model: str, progress: gr.Progress | None = None) -> str:
    """Download an approved base checkpoint and return its ComfyUI name."""
    if base_model not in BASE_MODELS:
        raise ValueError("unsupported Krea base model")
    item = BASE_MODELS[base_model]
    _download_to_dest(
        item["repo"],
        item["filename"],
        MODELS / "diffusion_models" / item["filename"],
        item["label"],
    )
    return base_model


def _ensure_custom_base_model(value: dict[str, Any]) -> str:
    """Download a validated custom HF base model into a managed ComfyUI folder."""
    normalized = validate_custom_base_model(value)
    namespace = stable_custom_base_model_namespace(
        normalized["repo_id"], normalized["filename"], normalized["revision"]
    )
    diffusion_root = MODELS / "diffusion_models"
    destination_dir = diffusion_root / "huggingface" / namespace
    relative_file = pathlib.PurePosixPath(normalized["filename"])
    destination = destination_dir.joinpath(*relative_file.parts)
    if destination.exists():
        relative = destination.resolve().relative_to(diffusion_root.resolve())
        return pathlib.PurePosixPath(*relative.parts).as_posix()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    kwargs: dict[str, Any] = {
        "repo_id": normalized["repo_id"],
        "filename": normalized["filename"],
        "local_dir": str(destination_dir),
        "token": token,
    }
    if normalized["revision"]:
        kwargs["revision"] = normalized["revision"]
    try:
        downloaded = pathlib.Path(hf_hub_download(**kwargs))
    except Exception as exc:
        raise RuntimeError(
            f"failed to download custom base model "
            f"{normalized['repo_id']}/{normalized['filename']}: {exc}"
        ) from exc
    if not downloaded.exists():
        raise RuntimeError(f"HF Hub returned a missing custom base-model path: {downloaded}")
    try:
        relative = downloaded.resolve().relative_to(diffusion_root.resolve())
    except ValueError as exc:
        raise RuntimeError("custom base-model download escaped the managed model directory") from exc
    print(f"[models] ready: custom base model {normalized['repo_id']}/{normalized['filename']}", flush=True)
    return pathlib.PurePosixPath(*relative.parts).as_posix()


def _resolve_base_model(
    base_model: str,
    custom_base_model: dict[str, Any] | None = None,
) -> str:
    """Resolve the UI selection to the relative model name expected by ComfyUI."""
    if base_model == CUSTOM_BASE_MODEL:
        if not custom_base_model:
            raise ValueError("enter a custom Hugging Face base-model repository and file")
        return _ensure_custom_base_model(custom_base_model)
    if base_model not in BASE_MODELS:
        raise ValueError("unsupported Krea base model")
    if custom_base_model:
        raise ValueError("custom base-model details require the custom model option")
    return _ensure_base_model(base_model)


def _ensure_models(
    base_model: str = DEFAULT_BASE_MODEL,
    custom_base_model: dict[str, Any] | None = None,
    progress: gr.Progress | None = None,
) -> str:
    total = len(DOWNLOADS) + 1
    for index, (repo, filename, destination, label) in enumerate(DOWNLOADS):
        if progress:
            progress(index / total, desc=f"downloading {label}")
        _download_to_dest(repo, filename, destination, label)
    if base_model == CUSTOM_BASE_MODEL:
        if progress:
            progress(len(DOWNLOADS) / total, desc="downloading custom Hugging Face checkpoint")
        return _resolve_base_model(base_model, custom_base_model)
    if progress:
        progress(len(DOWNLOADS) / total, desc="downloading selected Krea checkpoint")
    return _resolve_base_model(base_model)


def _ensure_lora(hf_filename: str) -> str:
    """Lazy-download one catalog LoRA and return its ComfyUI path."""
    normalized = pathlib.PurePosixPath(str(hf_filename).replace("\\", "/")).as_posix()
    if normalized not in _lora_by_filename:
        raise ValueError(f"catalog LoRA is not available: {hf_filename}")
    destination = LORA_DEST_DIR.joinpath(*pathlib.PurePosixPath(normalized).parts)
    if destination.exists():
        return pathlib.PurePosixPath("krea2", normalized).as_posix()
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    path = pathlib.PurePosixPath(normalized)
    downloaded = pathlib.Path(
        hf_hub_download(
            repo_id=LORA_HF_REPO,
            filename=path.name,
            subfolder=None if str(path.parent) == "." else str(path.parent),
            local_dir=str(LORA_DEST_DIR),
            token=token,
        )
    )
    if downloaded.resolve() != destination.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded), str(destination))
    return pathlib.PurePosixPath("krea2", normalized).as_posix()


def _ensure_custom_lora(row: dict[str, Any]) -> str:
    """Download one validated custom HF LoRA and return its ComfyUI path."""
    normalized = validate_custom_lora(row)
    namespace = stable_custom_lora_namespace(
        normalized["repo_id"], normalized["filename"], normalized["revision"]
    )
    destination_dir = CUSTOM_LORA_DEST_DIR / namespace
    relative_file = pathlib.PurePosixPath(normalized["filename"])
    destination = destination_dir.joinpath(*relative_file.parts)
    if destination.exists():
        relative = destination.resolve().relative_to(LORA_ROOT.resolve())
        return pathlib.PurePosixPath(*relative.parts).as_posix()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    kwargs: dict[str, Any] = {
        "repo_id": normalized["repo_id"],
        "filename": normalized["filename"],
        "local_dir": str(destination_dir),
        "token": token,
    }
    if normalized["revision"]:
        kwargs["revision"] = normalized["revision"]
    try:
        downloaded = pathlib.Path(hf_hub_download(**kwargs))
    except Exception as exc:
        raise RuntimeError(
            f"failed to download custom LoRA {normalized['repo_id']}/{normalized['filename']}: {exc}"
        ) from exc
    if not downloaded.exists():
        raise RuntimeError(f"HF Hub returned a missing custom LoRA path: {downloaded}")
    try:
        relative = downloaded.resolve().relative_to(LORA_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("custom LoRA download escaped the managed LoRA directory") from exc
    return pathlib.PurePosixPath(*relative.parts).as_posix()


def _read_source_workflow(path: pathlib.Path) -> dict[str, Any]:
    """Load a bundled visual workflow and fail early if it was not shipped."""
    if not path.exists():
        raise FileNotFoundError(f"workflow file is missing: {path.name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("nodes"):
        raise ValueError(f"workflow file has no nodes: {path.name}")
    return data


def _ref(node: str, output: int = 0) -> list[Any]:
    return [node, output]


def _validate_model_name(model_name: str) -> str:
    """Accept only a relative model name produced by the managed model loader."""
    normalized = str(model_name).replace("\\", "/")
    path = pathlib.PurePosixPath(normalized)
    if not normalized or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("invalid Krea base model path")
    return path.as_posix()


def _t2i_workflow(base_model: str = DEFAULT_BASE_MODEL) -> dict[str, Any]:
    base_model = _validate_model_name(base_model)
    cache_key = f"text2image:{base_model}"
    if cache_key in _workflow_cache:
        return json.loads(json.dumps(_workflow_cache[cache_key]))
    _read_source_workflow(T2I_SOURCE)
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": base_model, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors", "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": _ref("1"), "shift": 4.0}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": _ref("2"), "text": ""}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": _ref("5")}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": DEFAULT_WIDTH, "height": DEFAULT_HEIGHT, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"model": _ref("4"), "positive": _ref("5"), "negative": _ref("6"), "latent_image": _ref("7"), "seed": DEFAULT_SEED, "steps": DEFAULT_STEPS, "cfg": DEFAULT_CFG, "sampler_name": DEFAULT_SAMPLER, "scheduler": DEFAULT_SCHEDULER, "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": _ref("8"), "vae": _ref("3")}},
        "10": {"class_type": "SaveImage", "inputs": {"images": _ref("9"), "filename_prefix": "krea2_turbo"}},
    }
    _workflow_cache[cache_key] = workflow
    return json.loads(json.dumps(workflow))


def _edit_workflow(
    has_second_reference: bool,
    base_model: str = DEFAULT_BASE_MODEL,
) -> dict[str, Any]:
    base_model = _validate_model_name(base_model)
    _read_source_workflow(EDIT_SOURCE)
    workflow: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors", "type": "krea2", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "5": {"class_type": "UNETLoader", "inputs": {"unet_name": base_model, "weight_dtype": "default"}},
        "6": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": _ref("5"), "lora_name": "krea/" + IDENTITY_FILE, "strength_model": 1.0}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": _ref("1"), "vae": _ref("4")}},
        "8": {"class_type": "EmptySD3LatentImage", "inputs": {"width": DEFAULT_WIDTH, "height": DEFAULT_HEIGHT, "batch_size": 1}},
        "9": {"class_type": "Krea2EditModelPatch", "inputs": {"model": _ref("6"), "source_latent": _ref("7"), "vae": _ref("4"), "source_image": _ref("1"), "target_latent": _ref("8"), "ref_boost": DEFAULT_REF_BOOST, "ref_boost_a": DEFAULT_REF_BOOST, "fit_mode": "fit"}},
        "10": {"class_type": "Krea2EditGroundedEncode", "inputs": {"clip": _ref("3"), "image": _ref("1"), "prompt": "", "grounding_px": DEFAULT_GROUNDING}},
        "11": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": _ref("10")}},
        "12": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": _ref("9"), "shift": 4.0}},
        "13": {"class_type": "KSampler", "inputs": {"model": _ref("12"), "positive": _ref("10"), "negative": _ref("11"), "latent_image": _ref("8"), "seed": DEFAULT_SEED, "steps": DEFAULT_STEPS, "cfg": DEFAULT_CFG, "sampler_name": DEFAULT_SAMPLER, "scheduler": DEFAULT_SCHEDULER, "denoise": 1.0}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": _ref("13"), "vae": _ref("4")}},
        "15": {"class_type": "SaveImage", "inputs": {"images": _ref("14"), "filename_prefix": "krea2_edit"}},
    }
    if has_second_reference:
        workflow["2"] = {"class_type": "LoadImage", "inputs": {"image": ""}}
        workflow["16"] = {"class_type": "VAEEncode", "inputs": {"pixels": _ref("2"), "vae": _ref("4")}}
        workflow["9"]["inputs"]["source_latent_b"] = _ref("16")
        workflow["9"]["inputs"]["source_image_b"] = _ref("2")
        workflow["10"]["inputs"]["image_b"] = _ref("2")
    return workflow


def _find_node(workflow: dict[str, Any], class_type: str) -> str:
    for node_id, node in workflow.items():
        if node.get("class_type") == class_type:
            return node_id
    raise KeyError(f"workflow does not contain {class_type}")


def _inject_lora_chain(
    workflow: dict[str, Any],
    enabled_loras: list[tuple[str, float]],
    *,
    model_source: list[Any],
    clip_source: list[Any],
    model_consumers: list[tuple[str, str]],
    clip_consumers: list[tuple[str, str]],
) -> None:
    """Append an ordered model+CLIP LoRA chain to a workflow."""
    if not enabled_loras:
        return
    previous_model = model_source
    previous_clip = clip_source
    for index, (filename, strength) in enumerate(enabled_loras):
        node_id = f"user_lora_{index}"
        workflow[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": previous_model,
                "clip": previous_clip,
                "lora_name": filename,
                "strength_model": float(strength),
                "strength_clip": float(strength),
            },
        }
        previous_model = _ref(node_id)
        previous_clip = _ref(node_id, 1)
    for node_id, input_name in model_consumers:
        workflow[node_id]["inputs"][input_name] = previous_model
    for node_id, input_name in clip_consumers:
        workflow[node_id]["inputs"][input_name] = previous_clip


def _prepare_edit_image(path: str, target_megapixels: float) -> tuple[str, int, int]:
    """Resize the primary source to a 64-pixel grid while preserving its AR."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        megapixels = max(0.25, min(MAX_TARGET_MP, float(target_megapixels)))
        scale = (megapixels * 1_000_000 / max(1, image.width * image.height)) ** 0.5
        width = max(64, int(round(image.width * scale / 64) * 64))
        height = max(64, int(round(image.height * scale / 64) * 64))
        width = min(MAX_WIDTH, width)
        height = min(MAX_HEIGHT, height)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        name = f"input_{uuid.uuid4().hex[:12]}.png"
        image.save(INPUT / name, format="PNG")
    return name, width, height


def _stage_image(path: str, prefix: str) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        name = f"{prefix}_{uuid.uuid4().hex[:12]}.png"
        image.save(INPUT / name, format="PNG")
    return name


def _validate_request(mode: str, prompt: str, edit_prompt: str, primary: str | None) -> None:
    if mode not in {"text2image", "edit"}:
        raise ValueError("unsupported generation mode")
    if mode == "text2image" and not (prompt or "").strip():
        raise ValueError("enter a prompt")
    if mode == "edit":
        if not primary:
            raise ValueError("upload a primary image for edit mode")
        if not (edit_prompt or prompt or "").strip():
            raise ValueError("enter an edit instruction in the prompt or edit instruction field")


def _inject_t2i(
    workflow: dict[str, Any],
    *,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    enabled_loras: list[tuple[str, float]] | None = None,
) -> None:
    _inject_lora_chain(
        workflow,
        enabled_loras or [],
        model_source=_ref("1"),
        clip_source=_ref("2"),
        model_consumers=[("4", "model")],
        clip_consumers=[("5", "clip")],
    )
    workflow["5"]["inputs"]["text"] = prompt.strip()
    workflow["7"]["inputs"].update(width=int(width), height=int(height))
    workflow["8"]["inputs"].update(seed=int(seed), steps=int(steps), cfg=float(cfg), sampler_name=sampler, scheduler=scheduler, denoise=1.0)


def _inject_edit(
    workflow: dict[str, Any],
    *,
    primary_name: str,
    second_name: str | None,
    width: int,
    height: int,
    edit_prompt: str,
    grounding_px: int,
    ref_boost: float,
    ref_boost_a: float,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    enabled_loras: list[tuple[str, float]] | None = None,
) -> None:
    _inject_lora_chain(
        workflow,
        enabled_loras or [],
        model_source=_ref("6"),
        clip_source=_ref("3"),
        model_consumers=[("9", "model")],
        clip_consumers=[("10", "clip")],
    )
    workflow["1"]["inputs"]["image"] = primary_name
    workflow["8"]["inputs"].update(width=int(width), height=int(height))
    workflow["9"]["inputs"].update(ref_boost=float(ref_boost), ref_boost_a=float(ref_boost_a))
    workflow["10"]["inputs"].update(prompt=edit_prompt.strip(), grounding_px=int(grounding_px))
    workflow["13"]["inputs"].update(seed=int(seed), steps=int(steps), cfg=float(cfg), sampler_name=sampler, scheduler=scheduler, denoise=1.0)
    if second_name:
        workflow["2"]["inputs"]["image"] = second_name


def _execute_workflow(workflow: dict[str, Any]) -> list[str]:
    import execution
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    executor = execution.PromptExecutor(
        server_instance,
        cache_type=execution.CacheType.RAM_PRESSURE,
        cache_args={"lru": 0, "ram": 2.0, "ram_inactive": 8.0},
    )
    prompt_id = str(uuid.uuid4())
    save_id = _find_node(workflow, "SaveImage")
    executor.execute(workflow, prompt_id, extra_data={}, execute_outputs=[save_id])
    if not executor.success:
        message = executor.status_messages[-1] if executor.status_messages else "ComfyUI execution failed"
        raise RuntimeError(str(message))
    paths: list[pathlib.Path] = []
    for output in executor.history_result.get("outputs", {}).values():
        for items in output.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or not item.get("filename"):
                    continue
                base = OUTPUT if item.get("type", "output") == "output" else COMFY / item.get("type", "output")
                candidate = base / item.get("subfolder", "") / item["filename"]
                if candidate.exists():
                    paths.append(candidate)
    if not paths:
        paths = sorted(
            [pathlib.Path(item) for item in glob.glob(str(OUTPUT / "**" / "*.png"), recursive=True)],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    if not paths:
        raise RuntimeError("ComfyUI finished without an output image")
    return [str(path) for path in paths]


def _prepare_runtime(
    base_model: str = DEFAULT_BASE_MODEL,
    custom_base_model: dict[str, Any] | None = None,
    progress: gr.Progress | None = None,
) -> str:
    _ensure_comfy()
    resolved_base_model = _ensure_models(base_model, custom_base_model, progress)
    _init_comfy_nodes()
    return resolved_base_model


def get_gpu_duration(*args: Any, **kwargs: Any) -> int:
    """Estimate GPU time for Spaces' positional or keyword callback forms."""
    # spaces.zero invokes a duration callback with the complete generate()
    # argument list positionally. Keep the indexes aligned with generate().
    steps = kwargs.get("steps", args[11] if len(args) > 11 else DEFAULT_STEPS)
    width = kwargs.get("width", args[5] if len(args) > 5 else DEFAULT_WIDTH)
    height = kwargs.get("height", args[6] if len(args) > 6 else DEFAULT_HEIGHT)
    gen_budget = kwargs.get("gen_budget", args[17] if len(args) > 17 else 0)
    if gen_budget and int(gen_budget) > 0:
        return max(MIN_GPU_SECONDS, min(MAX_GPU_SECONDS, int(gen_budget)))
    lora_weights = kwargs.get("lora_weights", args[22] if len(args) > 22 else {}) or {}
    custom_loras = kwargs.get("custom_loras", args[23] if len(args) > 23 else []) or []
    lora_count = sum(
        1 for value in lora_weights.values()
        if value and abs(float(value)) > 1e-6
    )
    lora_count += sum(1 for row in custom_loras if isinstance(row, (dict, list, tuple)))
    estimate = int(
        35
        + (int(width) * int(height) / 1_000_000)
        * int(steps)
        * 3.0
        * (1 + 0.05 * lora_count)
    )
    return max(MIN_GPU_SECONDS, min(MAX_GPU_SECONDS, estimate))


@spaces.GPU(duration=get_gpu_duration)
def generate(
    mode: str,
    prompt: str,
    edit_prompt: str,
    primary_image: str | None,
    second_image: str | None,
    width: int,
    height: int,
    target_megapixels: float,
    grounding_px: int,
    ref_boost: float,
    ref_boost_a: float,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    randomize_seed: bool,
    gen_budget: float,
    base_model: str = DEFAULT_BASE_MODEL,
    custom_base_repo: str = "",
    custom_base_filename: str = "",
    custom_base_revision: str = "",
    lora_weights: dict[str, float] | None = None,
    custom_loras: list[dict[str, Any]] | None = None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[list[str], str, int]:
    """Validate inputs, execute the selected Krea graph, and persist outputs."""
    effective_seed = random.randint(0, 2**32 - 1) if randomize_seed or int(seed) < 0 else int(seed)
    staged: list[pathlib.Path] = []
    try:
        _validate_request(mode, prompt, edit_prompt, primary_image)
        effective_edit_prompt = (edit_prompt or prompt or "").strip()
        if sampler not in SAMPLERS or scheduler not in SCHEDULERS:
            raise ValueError("unsupported sampler or scheduler")
        custom_base = None
        if base_model == CUSTOM_BASE_MODEL:
            custom_base = validate_custom_base_model({
                "repo_id": custom_base_repo,
                "filename": custom_base_filename,
                "revision": custom_base_revision,
            })
        elif base_model not in BASE_MODELS:
            raise ValueError("unsupported Krea base model")
        resolved_base_model = _prepare_runtime(base_model, custom_base, progress)

        enabled_loras: list[tuple[str, float]] = []
        active_catalog: list[dict[str, Any]] = []
        for filename, weight in (lora_weights or {}).items():
            if filename not in _lora_by_filename:
                raise ValueError(f"catalog LoRA is not available: {filename}")
            numeric_weight = float(weight)
            if numeric_weight < -3.0 or numeric_weight > 3.0:
                raise ValueError(f"catalog LoRA weight is out of range: {filename}")
            if abs(numeric_weight) > 1e-6:
                enabled_loras.append((_ensure_lora(filename), numeric_weight))
                active_catalog.append({"hf_filename": filename, "weight": numeric_weight})

        normalized_custom, custom_warnings = normalize_custom_loras(custom_loras)
        if custom_warnings:
            raise ValueError("; ".join(custom_warnings))
        active_custom = [
            row for row in normalized_custom if abs(float(row["weight"])) > 1e-6
        ]
        for row in active_custom:
            enabled_loras.append((_ensure_custom_lora(row), float(row["weight"])))

        if mode == "text2image":
            width = max(512, min(MAX_WIDTH, int(width) // 64 * 64))
            height = max(512, min(MAX_HEIGHT, int(height) // 64 * 64))
            workflow = _t2i_workflow(resolved_base_model)
        else:
            primary_name, width, height = _prepare_edit_image(primary_image, target_megapixels)
            staged.append(INPUT / primary_name)
            second_name = _stage_image(second_image, "reference") if second_image else None
            if second_name:
                staged.append(INPUT / second_name)
            workflow = _edit_workflow(bool(second_name), resolved_base_model)

        if mode == "text2image":
            _inject_t2i(
                workflow,
                prompt=prompt,
                width=width,
                height=height,
                steps=int(steps),
                cfg=float(cfg),
                sampler=sampler,
                scheduler=scheduler,
                seed=effective_seed,
                enabled_loras=enabled_loras,
            )
        else:
            _inject_edit(
                workflow,
                primary_name=primary_name,
                second_name=second_name,
                width=width,
                height=height,
                edit_prompt=effective_edit_prompt,
                grounding_px=int(grounding_px),
                ref_boost=float(ref_boost),
                ref_boost_a=float(ref_boost_a),
                steps=int(steps),
                cfg=float(cfg),
                sampler=sampler,
                scheduler=scheduler,
                seed=effective_seed,
                enabled_loras=enabled_loras,
            )

        settings = build_settings(
            mode=mode,
            prompt=prompt,
            edit_prompt=effective_edit_prompt,
            width=width,
            height=height,
            target_megapixels=float(target_megapixels),
            grounding_px=int(grounding_px),
            ref_boost=float(ref_boost),
            ref_boost_a=float(ref_boost_a),
            steps=int(steps),
            cfg=float(cfg),
            sampler_name=sampler,
            scheduler=scheduler,
            seed=int(seed),
            randomize_seed=bool(randomize_seed),
            gen_budget=float(gen_budget),
            effective_seed=effective_seed,
            base_model=base_model,
            custom_base_model=custom_base,
            catalog_loras=active_catalog,
            custom_loras=active_custom,
        )
        progress(0.35, desc=f"generating {mode}")
        result_paths = _execute_workflow(workflow)
        destination_dir = pathlib.Path(tempfile.mkdtemp(prefix="krea2_outputs_"))
        output_paths: list[str] = []
        for index, source in enumerate(result_paths):
            destination = destination_dir / f"output_{index}.png"
            write_png_metadata(source, destination, settings)
            output_paths.append(str(destination))
        return output_paths, f"done — {len(output_paths)} image(s), seed {effective_seed}", effective_seed
    except Exception as exc:
        print(traceback.format_exc(), flush=True)
        raise gr.Error(f"generation failed: {str(exc)[:500]}") from exc
    finally:
        for path in staged:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _profile_from_values(
    mode: str,
    prompt: str,
    edit_prompt: str,
    width: int,
    height: int,
    target_mp: float,
    grounding: int,
    ref_boost: float,
    ref_boost_a: float,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    randomize: bool,
    budget: float,
    base_model: str = DEFAULT_BASE_MODEL,
    custom_base_repo: str = "",
    custom_base_filename: str = "",
    custom_base_revision: str = "",
    catalog_loras: list[dict[str, Any]] | None = None,
    custom_loras: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return build_settings(
        mode=mode,
        prompt=prompt,
        edit_prompt=edit_prompt,
        width=int(width),
        height=int(height),
        target_megapixels=float(target_mp),
        grounding_px=int(grounding),
        ref_boost=float(ref_boost),
        ref_boost_a=float(ref_boost_a),
        steps=int(steps),
        cfg=float(cfg),
        sampler_name=sampler,
        scheduler=scheduler,
        seed=int(seed),
        randomize_seed=bool(randomize),
        gen_budget=float(budget),
        base_model=base_model,
        custom_base_model=(
            validate_custom_base_model({
                "repo_id": custom_base_repo,
                "filename": custom_base_filename,
                "revision": custom_base_revision,
            })
            if base_model == CUSTOM_BASE_MODEL
            else None
        ),
        catalog_loras=catalog_loras,
        custom_loras=custom_loras,
    )


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="Krea 2 Turbo Image Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🖌️ Krea 2 Turbo Image Generator\nGenerate new images or edit a source image with Krea 2 Identity Edit.")
        gr.Markdown(
            "> ⚠️ **This Space might be taken down soon.** "
            "Please **duplicate** this Space (top-right menu → *Duplicate this Space*) or **clone** it locally "
            "(`git clone https://huggingface.co/spaces/mpasila/Krea-2-Turbo_I2I`) so you keep a working copy. "
            "Once you have a local copy you can modify it to run on **Runpod**, a local GPU, or any other environment."
        )
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(["text2image", "edit"], value="text2image", label="mode")
                base_model = gr.Dropdown(
                    choices=[
                        *[(item["label"], filename) for filename, item in BASE_MODELS.items()],
                        ("Custom Hugging Face base model", CUSTOM_BASE_MODEL),
                    ],
                    value=DEFAULT_BASE_MODEL,
                    label="base Krea checkpoint",
                )
                with gr.Column(visible=False) as custom_base_controls:
                    custom_base_repo = gr.Textbox(
                        label="custom base-model Hugging Face repository",
                        placeholder="namespace/repository",
                    )
                    custom_base_filename = gr.Textbox(
                        label="custom base-model filename",
                        placeholder="path/to/model.safetensors",
                    )
                    custom_base_revision = gr.Textbox(
                        label="custom base-model revision (optional)",
                        placeholder="main, tag, or commit hash",
                    )
                    gr.Markdown(
                        "Use a relative `.safetensors`, `.ckpt`, `.pt`, or `.bin` file. "
                        "Private repositories use `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`."
                    )
                with gr.Column(visible=False) as image_inputs:
                    primary = gr.Image(type="filepath", label="primary image / scene")
                    second = gr.Image(type="filepath", label="optional second reference")
                prompt = gr.Textbox(value="A cinematic portrait in soft natural light", label="prompt", lines=3)
                edit_prompt = gr.Textbox(label="edit instruction", lines=3, visible=False, placeholder="recolor the jacket to matte black")
                with gr.Column() as t2i_resolution:
                    with gr.Row():
                        width = gr.Slider(512, MAX_WIDTH, value=DEFAULT_WIDTH, step=64, label="width")
                        height = gr.Slider(512, MAX_HEIGHT, value=DEFAULT_HEIGHT, step=64, label="height")
                with gr.Column(visible=False) as edit_controls:
                    target_mp = gr.Slider(0.25, MAX_TARGET_MP, value=DEFAULT_TARGET_MP, step=0.05, label="target megapixels")
                    grounding = gr.Slider(384, 1536, value=DEFAULT_GROUNDING, step=32, label="grounding resolution")
                    ref_boost = gr.Slider(0.0, 12.0, value=DEFAULT_REF_BOOST, step=0.1, label="primary reference strength")
                    ref_boost_a = gr.Slider(0.0, 12.0, value=DEFAULT_REF_BOOST, step=0.1, label="second reference strength")
                with gr.Accordion(f"LoRAs ({len(_lora_catalog)} available)", open=False):
                    gr.Markdown(
                        "Set a weight other than zero to enable a catalog LoRA. "
                        "Enabled catalog entries are stacked in catalog order, followed "
                        "by custom rows. Negative weights are supported for sliders."
                    )
                    with gr.Row():
                        lora_search = gr.Textbox(
                            label="search catalog LoRAs",
                            placeholder="title, trigger word, tag, or category",
                            lines=1,
                        )
                        lora_category = gr.Dropdown(
                            choices=["all"] + sorted({str(item.get("category", "uncategorized")) for item in _lora_catalog}),
                            value="all",
                            label="category",
                        )
                    lora_slider_map: dict[str, gr.Slider] = {}
                    for item in _lora_catalog:
                        filename = str(item.get("hf_filename", ""))
                        if not filename:
                            continue
                        title = str(item.get("title", filename))
                        category = str(item.get("category", "uncategorized"))
                        triggers = ", ".join(str(value) for value in item.get("trigger_words", [])[:3])
                        label = f"[{category}] {title}"
                        if triggers:
                            label += f" — {triggers}"
                        lora_slider_map[filename] = gr.Slider(
                            minimum=-3.0,
                            maximum=3.0,
                            value=0.0,
                            step=0.05,
                            label=label,
                        )
                    custom_loras = gr.Dataframe(
                        headers=["repo_id", "filename", "revision", "weight"],
                        datatype=["str", "str", "str", "number"],
                        value=[],
                        row_count=(0, "dynamic"),
                        col_count=(4, "fixed"),
                        type="array",
                        label="custom Hugging Face LoRAs",
                        interactive=True,
                    )
                    gr.Markdown(
                        "Custom rows use `repo_id`, `filename`, optional `revision`, and `weight`. "
                        "Private repositories use `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`."
                    )
                with gr.Accordion("sampling", open=False):
                    steps = gr.Slider(4, 40, value=DEFAULT_STEPS, step=1, label="steps")
                    cfg = gr.Slider(1.0, 5.0, value=DEFAULT_CFG, step=0.1, label="CFG")
                    with gr.Row():
                        sampler = gr.Dropdown(SAMPLERS, value=DEFAULT_SAMPLER, label="sampler")
                        scheduler = gr.Dropdown(SCHEDULERS, value=DEFAULT_SCHEDULER, label="scheduler")
                with gr.Row():
                    seed = gr.Number(value=DEFAULT_SEED, precision=0, label="seed")
                    randomize = gr.Checkbox(value=False, label="randomize seed")
                with gr.Accordion("settings", open=False):
                    profile_name = gr.Textbox(label="profile name")
                    export_button = gr.Button("export settings", size="sm")
                    export_file = gr.File(label="profile", visible=False)
                    import_file = gr.File(label="import JSON profile", file_types=[".json"], type="filepath")
                    metadata_image = gr.Image(type="filepath", label="read settings from image")
                    read_metadata = gr.Button("read image metadata", size="sm")
                    profile_status = gr.Textbox(label="settings status", interactive=False)
                gen_budget = gr.Slider(0, MAX_GPU_SECONDS, value=0, step=10, label="GPU budget (0 = automatic)")
                button = gr.Button("generate", variant="primary", size="lg")
            with gr.Column(scale=1):
                gallery = gr.Gallery(label="output", columns=2, height=600)
                status = gr.Textbox(label="status", interactive=False)
                used_seed = gr.Number(label="used seed", interactive=False)

        def on_mode_change(value: str):
            editing = value == "edit"
            return gr.update(visible=editing), gr.update(visible=not editing), gr.update(visible=editing), gr.update(visible=editing)

        mode.change(on_mode_change, inputs=[mode], outputs=[image_inputs, t2i_resolution, edit_controls, edit_prompt])

        def on_base_model_change(value: str):
            return gr.update(visible=value == CUSTOM_BASE_MODEL)

        base_model.change(on_base_model_change, inputs=[base_model], outputs=[custom_base_controls])

        all_lora_filenames = list(lora_slider_map.keys())
        all_lora_sliders = list(lora_slider_map.values())

        def _filter_lora_sliders(search_text: str, category: str):
            search = (search_text or "").strip().lower()
            selected_category = (category or "all").strip()
            updates = []
            for item, filename in zip(_lora_catalog, all_lora_filenames):
                searchable = " ".join([
                    str(item.get("title", "")),
                    str(item.get("category", "")),
                    " ".join(str(value) for value in item.get("tags", [])),
                    " ".join(str(value) for value in item.get("trigger_words", [])),
                    filename,
                ]).lower()
                visible = (
                    (selected_category == "all" or item.get("category", "uncategorized") == selected_category)
                    and (not search or search in searchable)
                )
                updates.append(gr.update(visible=visible))
            return updates

        lora_search.change(
            _filter_lora_sliders,
            inputs=[lora_search, lora_category],
            outputs=all_lora_sliders,
        )
        lora_category.change(
            _filter_lora_sliders,
            inputs=[lora_search, lora_category],
            outputs=all_lora_sliders,
        )

        def _catalog_weights(values: list[Any]) -> dict[str, float]:
            return {
                filename: float(weight)
                for filename, weight in zip(all_lora_filenames, values)
                if weight and abs(float(weight)) > 1e-6
            }

        def _generate_wrapper(*values):
            base_values = values[:18]
            base_model_value = values[18]
            custom_repo = values[19]
            custom_filename = values[20]
            custom_revision = values[21]
            custom_rows = values[22]
            lora_weights = _catalog_weights(values[23:])
            return generate(
                *base_values,
                base_model=base_model_value,
                custom_base_repo=custom_repo,
                custom_base_filename=custom_filename,
                custom_base_revision=custom_revision,
                lora_weights=lora_weights,
                custom_loras=custom_rows,
            )

        generation_inputs = [
            mode, prompt, edit_prompt, primary, second, width, height, target_mp,
            grounding, ref_boost, ref_boost_a, steps, cfg, sampler, scheduler,
            seed, randomize, gen_budget, base_model, custom_base_repo, custom_base_filename,
            custom_base_revision, custom_loras, *all_lora_sliders,
        ]
        button.click(_generate_wrapper, inputs=generation_inputs, outputs=[gallery, status, used_seed])

        profile_inputs = [
            mode, prompt, edit_prompt, width, height, target_mp, grounding,
            ref_boost, ref_boost_a, steps, cfg, sampler, scheduler, seed,
            randomize, gen_budget, base_model, custom_base_repo, custom_base_filename,
            custom_base_revision, custom_loras, *all_lora_sliders,
            profile_name,
        ]

        def export_profile(*values):
            name_value = str(values[-1] or "krea2")
            catalog = [
                {"hf_filename": filename, "weight": float(weight or 0.0)}
                for filename, weight in zip(all_lora_filenames, values[21:-1])
                if weight and abs(float(weight)) > 1e-6
            ]
            normalized_custom, warnings = normalize_custom_loras(values[20])
            if warnings:
                return None, "settings export failed — " + "; ".join(warnings)
            try:
                data = _profile_from_values(
                    *values[:16],
                    values[16],
                    values[17],
                    values[18],
                    values[19],
                    catalog,
                    normalized_custom,
                )
            except (TypeError, ValueError) as exc:
                return None, f"settings export failed — {exc}"
            name = re.sub(r"[^A-Za-z0-9_-]", "_", name_value)[:40]
            handle = tempfile.NamedTemporaryFile(prefix=f"{name}_", suffix=".json", mode="w", encoding="utf-8", delete=False)
            json.dump(data, handle, indent=2)
            handle.close()
            return handle.name, "settings exported"

        export_button.click(export_profile, inputs=profile_inputs, outputs=[export_file, profile_status]).then(lambda: gr.update(visible=True), outputs=[export_file])

        settings_outputs = [
            mode, base_model, custom_base_repo, custom_base_filename, custom_base_revision,
            prompt, edit_prompt, width, height, target_mp, grounding,
            ref_boost, ref_boost_a, steps, cfg, sampler, scheduler, seed,
            randomize, gen_budget, custom_loras, *all_lora_sliders, profile_status, image_inputs,
            t2i_resolution, edit_controls, custom_base_controls,
        ]
        settings_keys = [
            "mode", "base_model", "custom_base_repo", "custom_base_filename", "custom_base_revision",
            "prompt", "edit_prompt", "width", "height",
            "target_megapixels", "grounding_px", "ref_boost", "ref_boost_a",
            "steps", "cfg", "sampler_name", "scheduler", "effective_seed",
            "randomize_seed", "gen_budget",
        ]

        def settings_updates(data: dict[str, Any], warnings: list[str], error: str = ""):
            view_data = dict(data)
            custom_base = data.get("custom_base_model")
            if custom_base:
                try:
                    normalized_base = validate_custom_base_model(custom_base)
                except ValueError as exc:
                    warnings.append(f"ignored custom base model: {exc}")
                else:
                    view_data.update({
                        "custom_base_repo": normalized_base["repo_id"],
                        "custom_base_filename": normalized_base["filename"],
                        "custom_base_revision": normalized_base["revision"],
                    })
            editing = view_data.get("mode") == "edit"
            custom_base_visible = view_data.get("base_model") == CUSTOM_BASE_MODEL
            updates = []
            for key in settings_keys:
                if key not in view_data:
                    updates.append(gr.update())
                elif key == "base_model" and view_data[key] not in (*BASE_MODELS, CUSTOM_BASE_MODEL):
                    warnings.append(f"ignored unsupported base model: {view_data[key]!r}")
                    updates.append(gr.update())
                elif key == "edit_prompt" and "mode" in view_data:
                    updates.append(gr.update(value=view_data[key], visible=editing))
                else:
                    updates.append(gr.update(value=view_data[key]))
            custom_rows: list[list[Any]] = []
            if "custom_loras" in view_data:
                for row in view_data.get("custom_loras", []) or []:
                    try:
                        normalized = validate_custom_lora(row)
                    except ValueError as exc:
                        warnings.append(f"ignored custom LoRA: {exc}")
                        continue
                    custom_rows.append([
                        normalized["repo_id"], normalized["filename"],
                        normalized["revision"], normalized["weight"],
                    ])
            updates.append(gr.update(value=custom_rows) if "custom_loras" in view_data else gr.update())
            catalog_values = {filename: 0.0 for filename in all_lora_filenames}
            for row in view_data.get("catalog_loras", []) or []:
                if not isinstance(row, dict):
                    warnings.append("ignored malformed catalog LoRA entry")
                    continue
                filename = str(row.get("hf_filename", ""))
                if filename not in catalog_values:
                    warnings.append(f"catalog LoRA not found: {filename}")
                    continue
                try:
                    weight = float(row.get("weight", 0.0))
                except (TypeError, ValueError):
                    warnings.append(f"ignored invalid catalog LoRA weight: {filename}")
                    continue
                if not -3.0 <= weight <= 3.0:
                    warnings.append(f"ignored out-of-range catalog LoRA weight: {filename}")
                    continue
                catalog_values[filename] = weight
            updates.extend(gr.update(value=catalog_values[filename]) for filename in all_lora_filenames)
            message = error or ("settings loaded" if not warnings else "; ".join(warnings[:6]))
            if len(warnings) > 6:
                message += f" (+{len(warnings) - 6} more)"
            updates.extend([
                message,
                gr.update(visible=editing) if "mode" in view_data else gr.update(),
                gr.update(visible=not editing) if "mode" in view_data else gr.update(),
                gr.update(visible=editing) if "mode" in view_data else gr.update(),
                gr.update(visible=custom_base_visible) if "base_model" in view_data else gr.update(),
            ])
            return tuple(updates)

        def import_profile(path: str | None):
            if not path:
                return settings_updates({}, [], "select a profile first")
            try:
                data, warnings = parse_settings_text(pathlib.Path(path).read_text(encoding="utf-8"))
            except Exception as exc:
                return settings_updates({}, [], f"could not read profile: {exc}")
            return settings_updates(data, warnings)

        import_file.change(import_profile, inputs=[import_file], outputs=settings_outputs)

        def import_metadata(path: str | None):
            if not path:
                return settings_updates({}, [], "upload an image first")
            data, warnings = extract_image_settings(path)
            return settings_updates(data, warnings)

        read_metadata.click(import_metadata, inputs=[metadata_image], outputs=settings_outputs)
    return demo


def _on_startup() -> None:
    if os.environ.get("KREA_SKIP_STARTUP") == "1":
        return
    try:
        _prepare_runtime()
    except Exception as exc:
        print(f"[startup] setup incomplete ({type(exc).__name__}: {exc}); generation will retry", flush=True)


_on_startup()
_load_lora_catalog()
demo = create_ui()
demo.queue()

if __name__ == "__main__":
    demo.launch()
