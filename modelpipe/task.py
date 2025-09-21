from .builder import build_pipeline, build_reader, build_writers, build_evaluators
from tqdm import tqdm


class Task:

    def __init__(self, config, device_id) -> None:
        self.config = config
        self.pipeline = build_pipeline(config.pipeline)
        self.pipeline.set_device_id(device_id)
        self.reader = build_reader(config.reader)
        self.writers = build_writers(config.writers)
        self.evaluators = build_evaluators(config.evaluators) if "evaluators" in config.to_dict() else None

    def init(self):
        self.pipeline.init()
        for writer in self.writers:
            writer.init()

    def exec(self):

        for data in tqdm(self.reader):
            if data.pipeline_reset:
                self.pipeline.reset()
            result = self.pipeline.run(data)
            if self.evaluators:
                for evaluator in self.evaluators:
                    if data.last_frame:
                        print(evaluator.process(result))
                    evaluator.process(result)
            for writer in self.writers:
                writer.process(result)

    def release(self):
        self.pipeline.release()
        import gc
        gc.collect()


def run_task(task_cfg, device_id):
    task = Task(task_cfg, device_id=device_id)
    task.init()
    task.exec()
    task.release()
