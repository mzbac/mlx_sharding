from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

static_files = package_files('static')

setup(
    name="mlx-sharding",
    version="0.1.2",
    author="Anchen",
    author_email="li.anchen.au@gmail.com",
    description="A package for MLX model sharding and distributed inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mzbac/mlx_sharding",
    packages=find_packages(),
    python_requires=">=3.12.0",
    install_requires=[
        "mlx",
        "mlx_lm>=0.16.1",
        "numpy",
        "grpcio",
        "grpcio-tools",
        "transformers",
        "protobuf",
    ],
    entry_points={
        "console_scripts": [
            "mlx-sharding-server=shard.main:main",
            "mlx-sharding-api=shard.openai_api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "shard": ["protos/*.proto"],
        "": static_files, 
    },
)