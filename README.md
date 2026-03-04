# Gausstwin: Unified Simulation and Correction with Gaussian Splatting for Robotic Digital Twins

## Setup
We provide two options to setup the environment for running this repository.
The simpler way is to use [pixi](https://pixi.prefix.dev/latest/).
You can also follow the steps below to setup a conda environment.

### Prerequist

The setup has been tested on Ubuntu 24.04 with cuda 12.4.

For a higher cuda version, you need to switch to different isaacsim, isaaclab, torch version and specify the cuda version you use while installing packages which require cuda.

**Tipp**: The isaacsim and isaaclab could easily mess up the conda environment(python version, torch version, etc.). You can install them in a separate envrionment. We only use them to visualize the tracking result and to build the robot 3DGS model. 

---
### Setup Using Pixi
1. Clone the repository.
```
git clone --recursive
```

2. Install [pixi](https://pixi.prefix.dev/latest/) on your pc.
```
curl -fsSL https://pixi.sh/install.sh | sh
```
We have tried the setup under pixi version 0.63.2. If you meet problem setting up the environment using a newer pixi version, you can try to switch to this version.

3. Go to the folder where you clone the repository and install dependencies, then run:
```
pixi install
```

4. Build the repo and submodules.
```
pixi run build_repo
```
This would take some time.

To run any script in the repo, first run
```
pixi shell
```
to source the pixi environment. 
You can run `exit` to quit the environment.

**Tipp**: If you forget to add `--recursive` argument while cloning the repo, run the following command to fetch the submodules:
```
git submodule update --init --recursive
```

---

### Setup a Conda Envrionment
See [docs/setup.md](docs/setup.md) for more details.


## Quick Demo
### Download Weights and Dataset
Download the pre-trained weights for segmentation model.
```
pixi run download_ckpts
```
You can find the weights under folder `checkpoints`.

Download the videos for the demo.
```
pixi run download_demo
```
You can find the videos under `dataset/demo`.

### Edit the config file
All the configurations are under the `config` folder. 
Change the `tam_checkpoint`, `sam2_checkpoint`, and `data_dir` to where you download the weights and dataset.
`tam_config` and `sam2_config` are relevant directory. You do not need to change them. 

The config file for tracking is under `config/tracking`.

### Rope Tracking
Process the data.
```
pixi run process_video_rope
```
This script run segmentation for the rope to obtain its point cloud for visualization.
It would take some time to compile the model when you run the segmentation model for the first time.
A file with pointcloud of the object at each time stamp would be saved.

Run tracking.
```
pixi run track_rope
```
A file `result.usda` would be saved in `scripts/tracking/rope`. Run `isaacsim` to open the simulation and drag this file into the window and click play, which is at the 

### Rigid Body Tracking
Process the data
```
pixi run process_video_rigid
```
Run tracking.
```
pixi run track_rope
```

## Build Robot Gaussians

See [docs/build_robot_gaussians.md](docs/build_robot_gaussians.md) for more details.

## Build Objects
See [docs/build_object.md](docs/build_object.md) for more details.

## Acknowledgements

Some code of this repo is adapted from [embodied_gaussians](https://github.com/bdaiinstitute/embodied_gaussians).
Thanks for making the code public.

## Citation
@todo
