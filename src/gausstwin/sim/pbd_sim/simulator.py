import warp as wp
import warp.sim
import warp.sim.render

from gausstwin.cfg.sim.pbd_cfg import PBDConfig
from gausstwin.sim.model_builder.model import Model
from gausstwin.sim.solver.integrator import XPBDIntegrator
from gausstwin.tracker.rope.model_rope_builder import RopeBuilder
from gausstwin.tracker.rigid_body.model_rigid_builder import RigidBuilder
from gausstwin.utils.warp_utils import velocity_damping_particle_kernel, velocity_damping_rigid_body_kernel, velocity_damping_rod_kernel


class Simulator:
    def __init__(
        self, 
        builder: RigidBuilder | RopeBuilder, 
        cfg: PBDConfig, 
        model: Model, 
        device: str = "cuda", 
        capture_graph: bool = True,
    ):
        # save 
        self.cfg = cfg
        self.builder = builder
        self.device = device
        self.model = model
        
        # setup simulation
        self.fps = cfg.fps
        self.frame_dt = 1.0 / cfg.fps
        self.sim_substeps = cfg.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        # self.num_envs = self.model.num_envs
        self.sim_time = 0.0
        # solver
        self.solver = XPBDIntegrator(iterations=self.cfg.solver_iterations)
        self.substeps = self.cfg.substeps
        self.dt = 1.0 / self.cfg.fps / self.substeps
        # states
        self.state_0 = self.model.state() 
        self.state_1 = self.model.state()
        self.control = self.model.control()
        
        # viewer
        if cfg.render:
            self.renderer = warp.sim.render.SimRenderer(
                self.model,
                path=cfg.stage_path,
            )
        
        self.graph = None
        if capture_graph:
            self.capture()
    
    
    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None
    
    
    def simulate(self):
        """
        Perform a step of the simulation.
        """
        # detect collisions
        warp.sim.collide(self.model, self.state_0) 
        # run substeps
        for _ in range(self.sim_substeps):
            self.solver.simulate(
                self.model,
                self.state_0,
                self.state_1,
                self.dt,
            )
            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

        # velocity damping
        if self.model.body_count > 8:
            wp.launch(kernel=velocity_damping_rigid_body_kernel, dim=self.model.body_count, inputs=[0.9, self.state_0.body_qd, self.state_1.body_qd])
        if self.model.particle_count > 0:
            wp.launch(kernel=velocity_damping_particle_kernel, dim=self.model.particle_count, inputs=[0.8, self.state_0.particle_qd, self.state_1.particle_qd])
        if self.model.rod_count > 0:
            wp.launch(kernel=velocity_damping_rod_kernel, dim=self.model.rod_seg_count, inputs=[0.8, self.state_0.rod_qd, self.state_1.rod_qd])

    
    def step_pbd(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        
        if self.cfg.render:
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()
