import torch
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class GaussianState:
    means: torch.Tensor             # (n, 3)
    quats: torch.Tensor             # (n, 4) (w, x, y, z)
    colors_logits: torch.Tensor     # (n, 3)
    opacities_logits: torch.Tensor  # (n,)
    scale_log: torch.Tensor         # (n, 3)

    def copy(self, gs: "GaussianState"):
        with torch.no_grad():
            self.means.copy_(gs.means)
            self.quats.copy_(gs.quats)
            self.colors_logits.copy_(gs.colors_logits)
            self.opacities_logits.copy_(gs.opacities_logits)
            self.scale_log.copy_(gs.scale_log)


    def clone(self):
        with torch.no_grad():
            return GaussianState(
                means=self.means.clone(),
                quats=self.quats.clone(),
                colors_logits=self.colors_logits.clone(),
                opacities_logits=self.opacities_logits.clone(),
                scale_log=self.scale_log.clone(),
            )


    def slice(self, slice_obj):
        with torch.no_grad():
            return GaussianState(
                means=self.means[slice_obj],
                quats=self.quats[slice_obj],
                colors_logits=self.colors_logits[slice_obj],
                opacities_logits=self.opacities_logits[slice_obj],
                scale_log=self.scale_log[slice_obj],
            )


    def reshape(self, shape):
        with torch.no_grad():
            return GaussianState(
                means=self.means.reshape(*shape, 3),
                quats=self.quats.reshape(*shape, 4),
                colors_logits=self.colors_logits.reshape(*shape, 3),
                opacities_logits=self.opacities_logits.reshape(*shape),
                scale_log=self.scale_log.reshape(*shape, 3),
            )
    
    
    @property
    def colors(self):
        return self.colors_logits.sigmoid()
    

    @property
    def opacities(self):
        return self.opacities_logits.sigmoid()
    

    @property
    def scales(self):
        return self.scale_log.exp()
    

    @property
    def num_gaussians(self):
        return self.means.shape[0]
    

@dataclass
class GaussianModel:
    means: torch.Tensor      # (n, 3)
    quats: torch.Tensor      # (n, 4) (w, x, y, z)
    scales: torch.Tensor     # (n, 3)
    opacities: torch.Tensor  # (n,)
    colors: torch.Tensor     # (n, 3)
    body_ids: torch.Tensor   # (n,)

    @property
    def num_gaussians(self):
        return self.means.shape[0]

    @property
    def device(self):
        return self.means.device

    def state(self):
        return GaussianState(
            means=self.means.clone(),
            quats=self.quats.clone(),
            colors_logits=self.colors.logit(),
            opacities_logits=self.opacities.logit(),
            scale_log=self.scales.log(),
        )

    def copy_from_state(self, state: GaussianState):
        with torch.no_grad():
            # self.means.copy_(state.means)
            # self.quats.copy_(state.quats)
            self.scales.copy_(state.scales)
            self.colors.copy_(state.colors)
            self.opacities.copy_(state.opacities)


@dataclass
class GaussianRigidModel:
    means: torch.Tensor      # (n, 3)
    quats: torch.Tensor      # (n, 4) (w, x, y, z)
    scales: torch.Tensor     # (n, 3)
    opacities: torch.Tensor  # (n,)
    colors: torch.Tensor     # (n, 3)
    gs_body_ids: torch.Tensor  # (n,)
    gs_robot_indices: torch.Tensor  # (n,)
    gs_obj_indices: torch.Tensor  # (n,)
    gs_robot_body_ids: torch.Tensor  # (n,)
    gs_obj_body_ids: torch.Tensor  # (n,)
    gs_obj_body_local_ids: torch.Tensor  # (n,)

    @property
    def num_gaussians(self):
        return self.means.shape[0]

    @property
    def num_robot_gaussians(self):
        return self.gs_robot_indices.shape[0]

    @property
    def num_obj_gaussians(self):
        return self.gs_obj_indices.shape[0]

    @property
    def device(self):
        return self.means.device

    def state(self):
        return GaussianState(
            means=self.means.clone(),
            quats=self.quats.clone(),
            colors_logits=self.colors.logit(),
            opacities_logits=self.opacities.logit(),
            scale_log=self.scales.log(),
        )

    def copy_from_state(self, state: GaussianState):
        with torch.no_grad():
            # self.means.copy_(state.means)
            # self.quats.copy_(state.quats)
            self.scales.copy_(state.scales)
            self.colors.copy_(state.colors)
            self.opacities.copy_(state.opacities)


@dataclass
class GaussianRopeModel:
    means: torch.Tensor      # (n, 3)
    quats: torch.Tensor      # (n, 4) (w, x, y, z)
    scales: torch.Tensor     # (n, 3)
    opacities: torch.Tensor  # (n,)
    colors: torch.Tensor     # (n, 3)
    gs_robot_indices: torch.Tensor  # (n_rigid_body_gaussians,) indicies of all gaussians that are rigid body gaussians
    gs_rope_indices: torch.Tensor  # (n_rope_gaussians,) indicies of all gaussians that are rope gaussians
    gs_robot_body_ids: torch.Tensor  # (n_rigid_body_gaussians,) # indices of rigid body which the gaussian belongs to
    gs_rope_part_ids: torch.Tensor  # (n_rope_gaussians,) # indices of rope part which the gaussian belongs to
    gs_rope_quat_ids: torch.Tensor  # (n_rope_gaussians,) # indices of rope part which the gaussian belongs to

    @property
    def num_gaussians(self):
        return self.means.shape[0]

    @property
    def num_gaussians_robot(self):
        return self.gs_robot_indices.shape[0]
    
    @property
    def num_gaussians_rope(self):
        return self.gs_rope_indices.shape[0]
    
    @property
    def device(self):
        return self.means.device

    def state(self):
        return GaussianState(
            means=self.means.clone(),
            quats=self.quats.clone(),
            colors_logits=self.colors.logit(),
            opacities_logits=self.opacities.logit(),
            scale_log=self.scales.log(),
        )

    def copy_from_state(self, state: GaussianState):
        with torch.no_grad():
            # self.means.copy_(state.means)
            # self.quats.copy_(state.quats)
            self.scales.copy_(state.scales)
            self.colors.copy_(state.colors)
            self.opacities.copy_(state.opacities)


@dataclass
class GaussianUnifiedModel:
    means: torch.Tensor      # (n, 3)
    quats: torch.Tensor      # (n, 4) (w, x, y, z)
    scales: torch.Tensor     # (n, 3)
    opacities: torch.Tensor  # (n,)
    colors: torch.Tensor     # (n, 3)
    
    # Gaussian indices: robot (body), rigid body (body), rope (rod)
    robot_indices: torch.Tensor  # (n,) robot gaussians
    rigid_indices: torch.Tensor  # (n,) rigid object gaussians
    rope_indices: torch.Tensor   # (n,) rope gaussians

    # global body/part ids
    body_ids: torch.Tensor        # (num_body_gaussians,) global body ids
    robot_body_ids: torch.Tensor  # (num_robot_gaussians,) robot body ids
    rigid_body_ids: torch.Tensor  # (num_rigid_gaussians,) rigid object body ids
    rope_part_ids: torch.Tensor   # (num_rope_gaussians,) rope part ids 
    rope_quat_ids: torch.Tensor   # (num_rope_gaussians,) rope quaternion ids 

    # local body/part ids
    robot_body_local_ids: torch.Tensor  # (num_robot_gaussians,) robot local body ids
    rigid_body_local_ids: torch.Tensor  # (num_rigid_gaussians,) rigid object local body ids
    # rope_part_local_ids: torch.Tensor   # (num_rope_gaussians,) rope local part ids
    
    @property
    def num_gaussians(self):
        return self.means.shape[0]

    @property
    def num_robot_gaussians(self):
        return self.robot_indices.shape[0]

    @property
    def num_obj_gaussians(self):
        return self.rigid_indices.shape[0]

    @property
    def num_rope_gaussians(self):
        return self.rope_indices.shape[0]
    
    @property
    def num_body_gaussians(self):
        return self.robot_indices.shape[0] + self.rigid_indices.shape[0]

    @property
    def device(self):
        return self.means.device

    def state(self):
        return GaussianState(
            means=self.means.clone(),
            quats=self.quats.clone(),
            colors_logits=self.colors.logit(),
            opacities_logits=self.opacities.logit(),
            scale_log=self.scales.log(),
        )

    def copy_from_state(self, state: GaussianState):
        with torch.no_grad():
            # self.means.copy_(state.means)
            # self.quats.copy_(state.quats)
            self.scales.copy_(state.scales)
            self.colors.copy_(state.colors)
            self.opacities.copy_(state.opacities)


@dataclass
class GaussianShapeMatchModel:
    means: torch.Tensor      # (n, 3)
    quats: torch.Tensor      # (n, 4) (w, x, y, z)
    scales: torch.Tensor     # (n, 3)
    opacities: torch.Tensor  # (n,)
    colors: torch.Tensor     # (n, 3)
    
    # Gaussian indices: robot (body), rigid body (body), rope (rod)
    robot_indices: torch.Tensor  # (num_robot_gaussians,) robot gaussians
    rigid_indices: torch.Tensor  # (num_obj_gaussians,) rigid object gaussians
    bgd_indices: torch.Tensor    # (num_bgd_gaussians,) background gaussians

    # global shape ids
    shape_ids: torch.Tensor        # (num_shape_gaussians,) global shape ids
    robot_shape_ids: torch.Tensor  # (num_robot_gaussians,) robot shape ids
    rigid_shape_ids: torch.Tensor  # (num_rigid_gaussians,) rigid object shape ids

    # local shape ids
    robot_shape_local_ids: torch.Tensor  # (num_robot_gaussians,) robot local shape ids
    rigid_shape_local_ids: torch.Tensor  # (num_rigid_gaussians,) rigid object local shape ids
    
    @property
    def num_gaussians(self):
        return self.means.shape[0]
    
    @property
    def num_bgd_gaussians(self):
        return self.bgd_indices.shape[0]

    @property
    def num_robot_gaussians(self):
        return self.robot_indices.shape[0]

    @property
    def num_obj_gaussians(self):
        return self.rigid_indices.shape[0]

    @property
    def num_shape_gaussians(self):
        return self.robot_indices.shape[0] + self.rigid_indices.shape[0]

    @property
    def device(self):
        return self.means.device

    def state(self):
        return GaussianState(
            means=self.means.clone(),
            quats=self.quats.clone(),
            colors_logits=self.colors.logit(),
            opacities_logits=self.opacities.logit(),
            scale_log=self.scales.log(),
        )

    def copy_from_state(self, state: GaussianState):
        with torch.no_grad():
            # self.means.copy_(state.means)
            # self.quats.copy_(state.quats)
            self.scales.copy_(state.scales)
            self.colors.copy_(state.colors)
            self.opacities.copy_(state.opacities)
            