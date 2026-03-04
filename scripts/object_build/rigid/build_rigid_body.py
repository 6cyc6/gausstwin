from gausstwin.utils.path_utils import get_builder_cfg_path
from gausstwin.cfg.builder.builder_cfg import RigidBodyConfig
from gausstwin.scene_builders.rigid_body_builder import RigidBodyBuilder


def build_rigid_body():
    cfg_path = get_builder_cfg_path("rigid_body_builder.yaml")
    cfg = RigidBodyConfig.from_yaml(cfg_path)
    builder = RigidBodyBuilder(cfg=cfg)

    builder.build()
    

if __name__ == "__main__":
    build_rigid_body()
    