## Build Robot Gaussians

before running other scripts in the repo, run:
```
source env.sh
```
to activate the environment variables first.

We have provided a Particle-3DGS representation of the panda robot with our manipulator in `src/gausstwin/save/fr3_v3.json`.
You can follow the instructions here to train the representation by yourself or custom your own robot model for training.

### Check the Representation
```
cd script/robot_build
python check_robot_model.py
```

## Generate Dataset and Train the Representation
### Save Images
We first load each link of the franka robot in isaacsim and save the images. 
We keep the camera fixed and rotate each link to obtain 200 viewpoints.
```
cd script/robot_build
python save_robot_img.py --enable_cameras --link link0
python save_robot_img.py --enable_cameras --link link1
python save_robot_img.py --enable_cameras --link link2
python save_robot_img.py --enable_cameras --link link3
python save_robot_img.py --enable_cameras --link link4
python save_robot_img.py --enable_cameras --link link5
python save_robot_img.py --enable_cameras --link link6
python save_robot_img.py --enable_cameras --link link7
python save_robot_img.py --enable_cameras --link finger_v3
```
Press `ctrl + c` in terminal to stop the simulation after images from all viewpoints are captured.
The relevant information is saved in `src/gausstwin/fig/fr3_v3`.

Then, run:
```
python build_robot.py
```

The Particle-3DGS representation of the robot will be saved in `src/gausstwin/save/fr3_v3.json`.


## Customize your own manipulator
You can build Particle-3DGS representation for your own manipulator by following these steps. You might need to install [blender](https://www.blender.org/) and isaacsim.

### Build 3DGS
First, you need to prepare a `usd` file for your link such that you can load it in isaacsim to capture images.

1. Import the `.dae` mesh model with textures and materials in blender. For `.stl` mesh without texture, import it in blender and add the color (texture, material, ...) for it. 
2. Click `file` -> `Export` -> `Universal Scene Description (.usd)`, rename the file `<your_link_name>.usd`.
3. In terminal, activate the conda environment and run `isaacsim` to start isaacsim.
4. Open the `your_link_name.usd` file, delete the `env_light`. At the right panel, right click -> `Create` -> `Xform`. Rename it. Then, drag all the stuff under this `Xform`. 
5. Right click the `Xform` -> `Add` -> `Physics` -> `Rigid Body with Colliders Preset`. This would set the link as a rigid body in the simulation. Then, save the file.

You might not see the model because there is no light source currently. If we open the saved `.usd` file again, you can see the model.

Don't forget to modify the `urdf` file for computing the forward kinematics.

### Build Particles 
We need to define the positions and raius of each particles for PBD. To do that, We need to manually assign them in `blender`. We can load the mesh in `blender` and click viewport shading at the tool box at top right (next to the right panel). This would make the object transparent. Then, we can click `Add` at the top tool box, -> `Add` -> `Mesh` -> `UV Sphere` to add spheres and fill the mesh with spheres. After filling the mesh, we manually save the position and radius of each sphere in a config file. You can refer to `/cfg/robot/fr3_cfg.py`.

An alternative is to use the same algorithm as the initialization step of the rigid body. We save the image with segmentation mask in isaacsim. Then, we get the bounding box of the pointcloud and fill the bounding box with uniformly spaced spheres. Next, we check whether the center of each particle is inside of the object mask. 

To improve efficiency of the simulator, you can only add particles close to the surface. 
For the gripper, we use a series of overlapped spheres to better approximate its actual shape for more precise collision reaction.
