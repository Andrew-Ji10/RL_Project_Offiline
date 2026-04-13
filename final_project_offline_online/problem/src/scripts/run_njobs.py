import multiprocessing as mp
import shlex
from importlib import import_module


def _load_entrypoint(entrypoint_module: str):
    module = import_module(entrypoint_module)
    return module.setup_arguments, module.main


def _worker(job_str: str, entrypoint_module: str):
    setup_arguments, main = _load_entrypoint(entrypoint_module)
    job_args_list = shlex.split(job_str)
    assert job_args_list[0] == 'JOB'
    del job_args_list[0]  # Delete the dummy "JOB" prefix
    print(job_args_list)
    job_args = setup_arguments(args=job_args_list)

    main(job_args)


def main_njobs(job_specs, njobs: int, start_method: str = "spawn", entrypoint_module: str = "scripts.run"):
    try:
        mp.set_start_method(start_method, force=True)
    except RuntimeError:
        pass

    with mp.Pool(processes=njobs) as pool:
        pool.starmap(_worker, [(spec, entrypoint_module) for spec in job_specs])
