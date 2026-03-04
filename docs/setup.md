### Setup a Conda Environment
1. Create a Conda environment
```
conda create -n gausstwin python=3.10 # use python 3.10 with isaacsim 4.5.0, python 3.11 with higher isaacsim version
conda activate gausstwin
```

2. Clone the repo
```
git clone --recursive
```

3. Install [Isaacsim and Issaclab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html)
```
pip install isaaclab[isaacsim,all]==2.1.0 --extra-index-url https://pypi.nvidia.com
```
To verify the installation, run: `isaacsim`, reply `Yes`.

4. Install torch
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

5. Install [Nvidia Warp](https://github.com/NVIDIA/warp)
```
pip install warp-lang[extras]==1.9 # warp version >= 1.10 does not work
```

6. Install cuda
```
conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit -y
```
<!-- You can also install cuda toolkit in your global environment. You can download the file on the [NVIDIA website](https://developer.nvidia.com/cuda-12-4-0-download-archive) and install it. -->
If you already have cuda 12.4 installed in your global environment, you do not need to install it again in the conda environment. 
Just remember to add your cuda path to these environment variables:
```
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.4/bin:$PATH 
```
You can simply put theses command in `env.sh`, which you always need to run first before running any scripts in this repo.

7. Install ssim
```
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation
```
If you are using gcc and g++ version >= 14.0, you might fail to build this package. Follow these steps:
```
conda install -c conda-forge "gcc_linux-64=13" "gxx_linux-64=13"  

# verify
$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc --version
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CXX"

pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation
```
Or install them in the global environment:
```
sudo apt-get update
sudo apt-get install -y gcc-13 g++-13

export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDAHOSTCXX=/usr/bin/g++-13

pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation
```
If the installation still fails, check whether you have specify the path to cuda in these environment variables:
```
# if you install cuda in the conda environment
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
Or
```
# if you install cuda in the global environement
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.4/bin:$PATH 
```


8. Others
```
pip install scipy
pip install trimesh
pip install polyscope
pip install open3d
pip install lycon
pip install scikit-image
```

9. Install this repo
```
python -m pip install -e .
```

10. Install submodules
```
cd submodules/vision
cd sam2
pip install -e .
cd ../efficient_track_anything
pip install -e .
```

Tipps: When you run code using `gsplat` at the first time, the package would compile the cuda code. If the gcc and g++ version are not correct, you will meet the same issue. 
Follow the same steps while installing ssim.
