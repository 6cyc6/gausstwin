# Build Object

before running other scripts in the repo, run:
```
source env.sh
```
to activate the environment variables first.

## Build Rigid Body 

1. Save the camera configs and check whether the objects are in view.
First modify the config file in `config/builder/rigid_body_builder.yaml`. Then run:
```
cd scripts/process_data
python save_cam_cfg.py --obj rigid
```
This would save json config files for each camera.

2. Build the Particle-3DGS representation for the object
```
cd ../object_build/rigid
python build_rigid_body.py
```
A window will jump out to ask for prompts.
The prompts consist of a couple of positive and negative points and a bounding box.
Follow the instructions in the terminal to add prompts.

Give the prompts for the ground first, then for each object in the scene.
After the prompts are given, the Particle-3DGS representation of the objects will be trained and saved in `src/gausstwin/save`.

## Build Rope
