from .registry import PIPELINES
from .registry import READERS
from .registry import WRITERS
from .registry import EVALUATORS


def build_pipeline(cfg):
    pipeline = PIPELINES.build(cfg)
    return pipeline


def build_reader(cfg):
    reader = READERS.build(cfg)
    return reader


def build_writers(cfg_list):
    writers = []
    for cfg in cfg_list:
        writers.append(WRITERS.build(cfg))
    return writers

def build_evaluators(cfg_list):
    evaluators = []
    for cfg in cfg_list:
        evaluators.append(EVALUATORS.build(cfg))
    return evaluators