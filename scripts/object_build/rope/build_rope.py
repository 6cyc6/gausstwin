from gausstwin.utils.path_utils import get_builder_cfg_path
from gausstwin.cfg.builder.builder_cfg import RopeConfig
from gausstwin.scene_builders.rope_builder import RopeBuilder


def build_rope():
    cfg_path = get_builder_cfg_path("rope_builder.yaml")
    cfg = RopeConfig.from_yaml(cfg_path)
    builder = RopeBuilder(cfg=cfg)

    builder.build()


if __name__ == "__main__":
    build_rope()
    