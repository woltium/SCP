from setuptools import setup, find_packages

setup(
    name="scp",
    packages=[package for package in find_packages() if package.startswith("scp")],
    install_requires=[
        "numpy==1.24",
        "Cython",
        "openai",
        "open3d",
        "kmeans-pytorch",
        "scikit-learn",
        "einops",
        "Pillow",
        "tqdm",
        "parse",
        "trimesh",
        "imageio",
        "scikit-image",
        "h5py",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "ftfy",
        "regex",
        "httpx[socks]",
    ],
    version="0.1.0",
)