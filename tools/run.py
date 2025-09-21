import argparse
from modelpipe.config import Config
from modelpipe.task import run_task
from mpire import WorkerPool
import sys
from loguru import logger

logger.add(lambda _: sys.exit(1), level='ERROR')


def parse_args():
    parser = argparse.ArgumentParser(description='run modelpipe tasks')
    parser.add_argument(
        '--config',
        '-cfg',
        default=  # noqa
        '/home/jianxu/code/modelpipe/configs/nreal_studio_annotation.py',  # noqa
        help='task config file path')
    parser.add_argument(
        '--num_processor',
        '-p',
        type=int,
        default=16,
        help='number of processor for parallel')  # noqa
    parser.add_argument(
        '--num_gpu', '-g', type=int, default=8, help='number of gpu')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if isinstance(cfg.tasks, dict):
        for task_key, task_value in cfg.tasks.items():
            print(f'Run task {task_key}')
            tasks = list()
            for i, task_cfg in enumerate(task_value):
                device_id = i % args.num_gpu
                tasks.append((task_cfg, device_id))
            run_task(*tasks[:1])
    elif isinstance(cfg.tasks, list):
        tasks = list()
        for i, task_cfg in enumerate(cfg.tasks):
            device_id = i % args.num_gpu
            tasks.append((task_cfg, device_id))

        with WorkerPool(
                n_jobs=args.num_processor, start_method='spawn') as pool:
            pool.map(run_task, tasks, progress_bar=True)


if __name__ == '__main__':
    main()
