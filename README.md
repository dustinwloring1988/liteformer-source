# LiteFormer

[![License](https://img.shields.io/github/license/dustinwloring1988/liteformer)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Stars](https://img.shields.io/github/stars/dustinwloring1988/liteformer?style=social)](https://github.com/dustinwloring1988/liteformer)

**LiteFormer** is a lightweight, research-focused fork of [Hugging Face Transformers](https://github.com/huggingface/transformers).

> ⚠️ **Not intended for production use.** This repository serves as a rapid prototyping and experimentation playground for transformer research.

---

## ✨ Features

- ✅ **Lightweight**: Removes TensorFlow, Flax, ONNX, Sagemaker, and multilingual doc support.
- ✅ **Focused**: Keeps only your experimental models:  
  `aformer`, `vformer`, `oformer`, `mformer`, `nformer`, `cformer`, `sformer`, `eformer`
- ✅ **Modular & Clean**: Cleaner structure and smaller size for fast exploration.
- ✅ **Custom Code Injection**: Injects your own model and tokenizer files automatically.
- ✅ **Placeholder Docs & Tests**: Basic test folders and markdown docs are generated for each model.

---

## ⚙️ Setup

Clone this repo and run the builder script:

```bash
python lite_transformers_builder.py
````

This will:

* Clone the official `transformers` repo
* Remove all unused backends, examples, tests, and files
* Retain only the models you define in the script
* Add placeholder folders and inject your own files for experimentation

---

## 🧠 Included Models

These models are retained and initialized for experimentation:

```
- aformer
- eformer
- vformer
- oformer
- mformer
- nformer
- cformer
- sformer
```

Each model gets:

* A `src/transformers/models/{model}` folder
* A test folder in `tests/models/{model}`
* A documentation stub in `docs/model_doc/{model}.md`

---

## 🗂 Output Structure

```plaintext
liteformer/
├── src/
│   └── transformers/
│       ├── models/
│       │   ├── aformer/
│       │   ├── ...
│       │   └── auto/
│       ├── generation/
│       └── ...
├── tests/
│   └── models/
│       ├── aformer/
│       ├── ...
├── docs/
│   ├── model_doc/
│   ├── index.rst
│   └── ...
├── utils/
├── docker/
└── examples/
```

---

## 🧪 Adding Your Own Model

To add a new model (e.g. `newformer`):

1. Create your files:

   * `__init__.py`
   * `modular_newformer.py`
   * `config_newformer.py`
   * `tokenization_newformer.py`
   * `tokenization_newformer_fast.py` (optional)
   * `processing_newformer.py` (required if multimodality)
   * `image_processing_newformer.py` (required for vision)
   * `image_processing_newformer_fast.py` (optional for vision)
   * `feature_extraction_newformer.py` (required for audio)

2. Add `"newformer"` to the `NEW_MODELS` list in `lite_transformers_builder.py`

3. Run the script again:

```bash
python lite_transformers_builder.py
```

---

## 🎯 Goals

* Minimize boilerplate and complexity during research
* Focus on experimental transformer variants
* Speed up iteration time when prototyping architectures
* Provide a simple foundation for building new ideas

---

## 🧰 Requirements

* Python 3.8+
* Git
* PyTorch
* (Optional) `pytest` for running tests

---

## 📋 TODO

* [ ] Make repo compile by removing or fixing invalid imports and paths
* [ ] Replace the current __version__ in `src/transformers/__init__.py` with __version__ = "0.1.0-lite"
* [ ] Reduce auto classes to only what is minimally necessary
* [ ] Replace placeholder models with real SOTA architectures (more info coming)
* [ ] Write clean, user-focused documentation on usage and architecture
* [ ] Upload as a package to PyPI for easier installation

---

## 🤝 Contributing

This is a personal research tool but you're welcome to fork it and adapt it for your own projects. Pull requests are not expected but feedback is welcome.

---

## 🧭 Why This Project Exists

The main Hugging Face `transformers` repository is incredibly powerful but often too large and slow to iterate within. **LiteFormer** aims to strip things down to the essentials — making it easier to test hypotheses, design new model types, and experiment with architecture changes in a clean environment.

---

## 📜 License

This project is a derivative work of [Transformers](https://github.com/huggingface/transformers), licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

---

## 👤 Author

Created by [Dustin Loring](https://github.com/dustinwloring1988)

GitHub Repo: [github.com/dustinwloring1988/liteformer](https://github.com/dustinwloring1988/liteformer)
