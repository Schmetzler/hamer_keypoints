from setuptools import setup, find_packages

version = "0.1.1"

setup(
    description='HaMeR as a package',
    name='hamer',
    version=version,
    packages=find_packages(),
    package_data={"hamer": ["*.task"]},
    install_requires=[
        'numpy',
        'opencv-python',
        'scikit-image',
        'smplx==0.1.28',
        'torch',
        'torchvision',
        'yacs',
        'mediapipe',
        'chumpy @ git+https://github.com/nico-von-huene/chumpy.git',
        'einops',
        'safetensorts',
    ],
)
