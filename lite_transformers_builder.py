import os
import shutil
import subprocess
import logging
from pathlib import Path
from typing import List

# --- Configuration ---
REPO_URL = "https://github.com/huggingface/transformers.git"
TARGET_DIR = "lite_transformers"

NEW_MODELS = [
    "aformer", "bformer", "cformer", "dformer", "eformer", "mformer",
    "nformer", "oformer", "sformer", "tformer", "vformer"
]

PRESERVE_MODELS = [
    "bark", "clip", "clip_text_model", "clip_vision_model", "clipseg", "gemma", "gemma2", "gemma3", 
    "gemma3_text", "gemma3n", "gemma3n_audio", "gemma3n_text", "gemma3n_vision", "llama", "llama4", 
    "llama4_text", "llava", "llava_next", "llava_next_video", "llava_onevision", "mistral", "mistral3", 
    "mixtral", "mllama", "mobilenet_v1", "mobilenet_v2", "mobilevit", "mobilevitv2", "openai-gpt", 
    "paligemma", "phi", "phi3", "phi4_multimodal", "phimoe", "pix2struct", "pixtral", "qwen2", 
    "qwen2_5_omni", "qwen2_5_vl", "qwen2_5_vl_text", "qwen2_audio", "qwen2_audio_encoder", 
    "qwen2_moe", "qwen2_vl", "qwen2_vl_text", "qwen3", "qwen3_moe", "sam", "sam_hq", 
    "sam_hq_vision_model", "sam_vision_model", "shieldgemma2", "siglip", "siglip2", 
    "siglip_vision_model", "smollm3", "smolvlm", "smolvlm_vision", "timm_backbone", 
    "timm_wrapper", "video_llava", "vit", "vit_hybrid", "vit_mae", "vit_msn", "vitdet", 
    "vitmatte", "vitpose", "vitpose_backbone", "vits", "vivit", "whisper", "yolos"
]

KEEP_MODELS = set(NEW_MODELS + PRESERVE_MODELS + ["auto"])
SCRIPT_DIR = Path(__file__).resolve().parent

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="🛠️  %(message)s")
log = logging.getLogger(__name__)


# --- Utilities ---
def run_subprocess(command: List[str]):
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        log.error(f"💥 Subprocess failed: {' '.join(command)}\n{e}")
        raise SystemExit(1)


def safe_rmtree(path: Path):
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
        log.info(f"📁 Removed folder: {path}")


def safe_remove_file(path: Path):
    if path.exists() and path.is_file():
        path.unlink()
        log.info(f"🗑️ Removed file: {path}")


# --- Repo Setup ---
def clone_repo():
    if Path(TARGET_DIR).exists():
        safe_rmtree(Path(TARGET_DIR))
    run_subprocess(["git", "clone", REPO_URL, TARGET_DIR])


# --- Cleanup Tasks ---
def cleanup_unwanted_models():
    os.chdir(TARGET_DIR)
    models_dir = Path("src/transformers/models")
    for subdir in models_dir.iterdir():
        if subdir.is_dir() and subdir.name not in KEEP_MODELS:
            safe_rmtree(subdir)

    for backend in ["flax", "tensorflow"]:
        for model in NEW_MODELS:
            safe_rmtree(models_dir / model / backend)


def cleanup_docs():
    docs_dir = Path("docs/source")
    for lang_dir in docs_dir.glob("*/"):
        if lang_dir.name != "en":
            safe_rmtree(lang_dir)


def cleanup_test_models():
    test_models_dir = Path("tests/models")
    if not test_models_dir.exists():
        return

    for subdir in test_models_dir.iterdir():
        if subdir.is_dir() and subdir.name not in KEEP_MODELS:
            safe_rmtree(subdir)

    for file in test_models_dir.rglob("*.py"):
        if not any(model in str(file) for model in KEEP_MODELS):
            safe_remove_file(file)


def cleanup_auto_models():
    auto_dir = Path("src/transformers/models/auto")
    for file_name in ["modeling_flax_auto.py", "modeling_tf_auto.py"]:
        safe_remove_file(auto_dir / file_name)


def cleanup_transformers_folders():
    base = Path("src/transformers")
    for folder in ["onnx", "sagemaker"]:
        safe_rmtree(base / folder)


def cleanup_generation_files():
    gen_dir = Path("src/transformers/generation")
    for fname in [
        "flax_logits_process.py", "flax_utils.py",
        "tf_logits_process.py", "tf_utils.py"
    ]:
        safe_remove_file(gen_dir / fname)


def cleanup_transformers_files():
    base = Path("src/transformers")
    files = [
        "convert_graph_to_onnx.py", "convert_pytorch_checkpoint_to_tf2.py",
        "convert_tf_hub_seq_to_seq_bert_to_pytorch.py", "modeling_flax_outputs.py",
        "modeling_flax_pytorch_utils.py", "modeling_flax_utils.py",
        "modeling_tf_outputs.py", "modeling_tf_pytorch_utils.py",
        "modeling_tf_utils.py", "optimization_tf.py", "tf_utils.py"
    ]
    for f in files:
        safe_remove_file(base / f)


def cleanup_model_docs():
    doc_dir = Path("docs/source/en/model_doc")
    if not doc_dir.exists():
        log.warning(f"⚠️ Model doc directory not found: {doc_dir}")
        return

    for file in doc_dir.glob("*.md"):
        if file.stem not in KEEP_MODELS:
            safe_remove_file(file)

    for model in NEW_MODELS:
        doc_file = doc_dir / f"{model}.md"
        doc_file.write_text(f"# {model.capitalize()} placeholder documentation\n")
        log.info(f"📄 Created doc for: {model}")


def cleanup_additional_files():
    safe_rmtree(Path("i18n"))
    safe_remove_file(Path("utils/check_tf_ops.py"))
    safe_rmtree(Path("docker/transformers-tensorflow-gpu"))

    docker_dir = Path("docker")
    for name in [
        "examples-tf", "pipeline-tf", "tf-light",
        "torch-tf-light", "torch-jax-light", "jax-light"
    ]:
        safe_remove_file(docker_dir / name)


def cleanup_examples():
    examples_dir = Path("examples")
    for folder in ["flax", "legacy", "tensorflow"]:
        safe_rmtree(examples_dir / folder)


# --- Additions ---
def add_placeholder_models():
    for model in NEW_MODELS:
        model_dir = Path("src/transformers/models") / model
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "README.md").write_text(f"# {model} placeholder\n")
        (model_dir / "model_card.md").write_text(f"# Model Card for {model}\n")

        test_dir = model_dir / "tests"
        test_dir.mkdir(exist_ok=True)
        (test_dir / "README.md").write_text(f"# Tests for {model} (placeholder)\n")
        log.info(f"📦 Placeholder created for model: {model}")


def create_test_folders():
    base_dir = Path("tests/models")
    base_dir.mkdir(parents=True, exist_ok=True)

    for model in NEW_MODELS:
        folder = base_dir / model
        folder.mkdir(parents=True, exist_ok=True)
        readme = folder / "README.md"
        readme.write_text(f"""# Tests for {model.capitalize()}

        This directory contains tests for the {model.capitalize()} model.

        ## Test Structure

        - `test_modeling_{model}.py` - Model architecture tests
        - `test_tokenization_{model}.py` - Tokenizer tests
        - `test_configuration_{model}.py` - Configuration tests

        ## Running Tests

        ```bash
        python -m pytest tests/models/{model}/
        ```

        ## Notes

        Add specific testing notes for {model.capitalize()} here.
        """)
        log.info(f"🧪 Created test folder for: {model}")


def update_main_init():
    init_path = Path("src/transformers/models/__init__.py")
    if not init_path.exists():
        log.warning(f"⚠️ models/__init__.py not found at: {init_path}")
        return

    current_content = init_path.read_text()
    with init_path.open("a", encoding="utf-8") as f:
        for model in NEW_MODELS:
            import_line = f"from . import {model}\n"
            if import_line not in current_content:
                f.write(import_line)
                log.info(f"📥 Added import for: {model}")


def reorganize_docs_structure():
    docs_dir = Path("docs")
    source_en = docs_dir / "source/en"
    target_dir = docs_dir

    onnx_file = source_en / "main_classes/onnx.md"
    safe_remove_file(onnx_file)

    if source_en.exists():
        for item in source_en.iterdir():
            dest = target_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(item), str(dest))
                log.info(f"📁 Moved folder: {item} → {dest}")
            else:
                shutil.move(str(item), str(dest))
                log.info(f"📄 Moved file: {item} → {dest}")

    source_dir = docs_dir / "source"
    safe_rmtree(source_dir)
    log.info(f"🧹 Removed docs/source directory")


def replace_auto_files():
    src_auto_dir = SCRIPT_DIR / "patches/src/transformers/models/auto"
    dest_auto_dir = Path("src/transformers/models/auto")

    if not src_auto_dir.exists():
        log.warning("⚠️  Source 'auto' directory not found in patches.")
        return

    for item in dest_auto_dir.glob("*"):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
        log.info(f"🗑️  Removed existing item in auto/: {item.name}")

    for item in src_auto_dir.iterdir():
        dest = dest_auto_dir / item.name
        if item.is_file():
            shutil.copy2(item, dest)
        elif item.is_dir():
            shutil.copytree(item, dest)
        log.info(f"📄 Copied {item.name} → {dest_auto_dir}")


def replace_root_files():
    patches_dir = SCRIPT_DIR / "patches"
    dest_dir = SCRIPT_DIR / TARGET_DIR

    if not patches_dir.exists():
        log.warning("⚠️  'patches' directory not found.")
        return

    if not dest_dir.exists():
        log.warning(f"⚠️  Target directory not found: {dest_dir}")
        return

    root_files = ["Makefile", "pyproject.toml", "setup.py"]
    for file_name in root_files:
        src_file = patches_dir / file_name
        dest_file = dest_dir / file_name

        if src_file.exists():
            shutil.copyfile(src_file, dest_file)
            log.info(f"📄 Replaced root file: {file_name}")
        else:
            log.warning(f"⚠️  Missing file in 'patches': {file_name}")


def cleanup_keep_model_files():
    base_dir = Path("src/transformers/models")
    # Prefixes for files to remove
    prefixes_to_remove = ("convert_", "modeling_flax_", "modeling_tf_")
    # Prefixes to match lines in __init__.py to remove
    import_line_prefixes = (
        "from .modeling_flax_",
        "from .modeling_tf_",
        "from .convert_"
    )

    for model in KEEP_MODELS:
        model_dir = base_dir / model
        if not model_dir.exists():
            continue

        # Remove files with specified prefixes
        for file in model_dir.glob("*.py"):
            if file.name.startswith(prefixes_to_remove):
                safe_remove_file(file)

        # Edit __init__.py to remove import lines starting with given prefixes
        init_file = model_dir / "__init__.py"
        if init_file.exists():
            lines = init_file.read_text().splitlines()
            updated_lines = []
            for line in lines:
                if any(line.strip().startswith(prefix) for prefix in import_line_prefixes):
                    log.info(f"✂️  Removed import line from {model}/__init__.py: {line.strip()}")
                    continue
                updated_lines.append(line)
            init_file.write_text("\n".join(updated_lines) + "\n")



# Update `main()` to include this function:
def main():
    clone_repo()
    cleanup_unwanted_models()
    cleanup_docs()
    cleanup_test_models()
    cleanup_auto_models()
    cleanup_transformers_folders()
    cleanup_generation_files()
    cleanup_transformers_files()
    cleanup_model_docs()
    cleanup_additional_files()
    cleanup_examples()
    add_placeholder_models()
    create_test_folders()
    update_main_init()
    reorganize_docs_structure()
    replace_auto_files()
    replace_root_files()
    cleanup_keep_model_files()  # ← Add this

    log.info(f"\n✅ LiteFormer package is ready at: {TARGET_DIR}")
    log.info(f"📦 Models included: {', '.join(sorted(KEEP_MODELS))}")
    log.info("🧪 You can now develop and test your custom transformer architectures!")



if __name__ == "__main__":
    main()
