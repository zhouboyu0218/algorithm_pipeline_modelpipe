from .config.registry import Registry

READERS = Registry('reader')
WRITERS = Registry('writer')
COMPUTE_NODES = Registry('compute_node')
PIPELINES = Registry('pipeline')
EVALUATORS = Registry('evaluator')
