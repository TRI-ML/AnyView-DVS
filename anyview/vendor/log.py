# stdlib replacement for imaginaire.utils.log (loguru); same call signatures.

'''
Minimal logging shim: debug / info / success / warning / error / critical(message, rank0_only=True).
'''

import logging

import torch.distributed as dist

_logger = logging.getLogger('anyview')
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%H:%M:%S'))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _emit(level: int, message: str, rank0_only: bool) -> None:
    if rank0_only and _rank() != 0:
        return
    _logger.log(level, message)


def debug(message: str, rank0_only: bool = True) -> None:
    _emit(logging.DEBUG, message, rank0_only)


def info(message: str, rank0_only: bool = True) -> None:
    _emit(logging.INFO, message, rank0_only)


def success(message: str, rank0_only: bool = True) -> None:
    _emit(logging.INFO, message, rank0_only)


def warning(message: str, rank0_only: bool = True) -> None:
    _emit(logging.WARNING, message, rank0_only)


def error(message: str, rank0_only: bool = True) -> None:
    _emit(logging.ERROR, message, rank0_only)


def critical(message: str, rank0_only: bool = True) -> None:
    _emit(logging.CRITICAL, message, rank0_only)
