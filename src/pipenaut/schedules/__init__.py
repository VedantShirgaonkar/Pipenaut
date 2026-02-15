"""Pipeline schedule implementations."""

from pipenaut.schedules.naive import naive_pipeline_step
from pipenaut.schedules.gpipe import gpipe_pipeline_step
from pipenaut.schedules.one_f_one_b import onef_oneb_pipeline_step

SCHEDULES = {
    "naive": naive_pipeline_step,
    "gpipe": gpipe_pipeline_step,
    "1f1b": onef_oneb_pipeline_step,
}

__all__ = ["SCHEDULES", "naive_pipeline_step", "gpipe_pipeline_step", "onef_oneb_pipeline_step"]
