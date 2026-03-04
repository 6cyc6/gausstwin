# Build Object

before running other scripts in the repo, run:
```
pixi shell
```
or for conda 
```
source env.sh
```
to activate the environment variables first.

## Build Rigid Body 

1. Save the camera configs and check whether the objects are in view.
First modify the config file in `config/builder/rigid_body_builder.yaml`. Then run:
```
pixi run preprocess_rigid
```
or for conda
```
cd scripts/process_data
python save_cam_cfg.py --obj rigid
```
This would save json config files for each camera.

2. Build the Particle-3DGS representation for the object.
Run
```
pixi run build_rigid
```
or for conda
```
cd ../object_build/rigid
python build_rigid_body.py
```
A window will jump out to ask for prompts.
The prompts consist of a couple of positive and negative points and a bounding box.
We found it the best to add several positive points and a bounding box.
Follow the instructions in the terminal to add prompts.

Give the prompts for the ground first, then for each object in the scene.
After the prompts are given, the Particle-3DGS representation of the objects will be trained and saved in `src/gausstwin/save`.

## Build Rope

1. Save the camera configs and check whether the objects are in view.
First modify the config file in `config/builder/rope_builder.yaml`. Then run:
```
pixi run preprocess_rope
```
or for conda
```
cd scripts/process_data
python save_cam_cfg.py --obj rope
```
This would save json config files for each camera.

2. Build the Particle-3DGS representation for the object.
Run
```
pixi run build_rigid
```
or for conda
```
cd ../object_build/rope
python build_rope.py
```
Follow the same steps as building the rigid body.
