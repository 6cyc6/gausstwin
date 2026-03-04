import click
from gausstwin.cfg.builder.builder_cfg import GSConfig
from gausstwin.utils.path_utils import get_builder_cfg_path
from gausstwin.scene_builders.robot_builder import RobotBuilder
from gausstwin.cfg.robot.robot_configs import list_available_robots



@click.command()
@click.option(
    "--robot", 
    type=click.Choice(list_available_robots(), case_sensitive=False),
    default="fr3_v3",
    help="Robot type to build",
    show_default=True
)
def build_robot(robot: str):
    """Build a robot model."""
    name = robot.lower()
    click.echo(f"Building robot: {name}")
    
    # load configuration
    cfg_path = get_builder_cfg_path("robot_builder.yaml")
    cfg = GSConfig.from_yaml(cfg_path)
    
    # get builder
    builder = RobotBuilder(cfg=cfg, name=name)
    
    # Build robot
    click.echo(f"Starting build process...")
    builder.build_robot()
    click.echo(f"✅ Successfully built robot '{name}'")


if __name__ == "__main__":
    build_robot()
