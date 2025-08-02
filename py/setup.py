import shutil
from pathlib import Path
from setuptools import find_packages, setup

# Remove stale egg-info
egg_info = Path(__file__).parent / "light_transformers.egg-info"
if egg_info.exists():
    print(f"Removing stale egg-info: {egg_info}")
    shutil.rmtree(egg_info)

install_requires = [
    "torch>=2.1",
    "tokenizers>=0.21,<0.22",
    "numpy>=1.17",
    "packaging>=20.0",
    "pyyaml>=5.1",
    "regex!=2019.12.17",
    "requests",
    "safetensors>=0.4.3",
    "tqdm>=4.27",
]

extras = {
    "dev": [
        "pytest>=7.2.0",
        "pytest-asyncio",
        "pytest-rerunfailures",
        "pytest-xdist",
        "pytest-order",
        "pytest-timeout",
        "parameterized>=0.9",
        "psutil",
        "dill",
        "evaluate>=0.2.0",
        "datasets>=2.15.0",
        "ruff==0.11.2",
        "GitPython<3.1.19",
        "pandas<2.3.0",
        "tensorboard",
    ],
    "quality": ["ruff==0.11.2"],
}

setup(
    name="light-transformers",
    version="0.1.0",
    author="Dustin Loring",
    author_email="dloring1988@gmail.com",
    description="Lightweight Transformers fork for specific models only",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    url="https://github.com/dustinwloring1988/light-transformers",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    extras_require=extras,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
