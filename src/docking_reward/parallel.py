"""Parallel execution backends: multiprocessing and Dask."""

import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class Executor(ABC):
    """Abstract base class for parallel executors."""

    @abstractmethod
    def map(self, func: Callable[[T], R], items: list[T]) -> list[R]:
        """
        Apply function to all items in parallel.

        Args:
            func: Function to apply to each item
            items: List of items to process

        Returns:
            List of results in the same order as items
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up executor resources."""
        pass

    def __enter__(self) -> "Executor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()


class SequentialExecutor(Executor):
    """Sequential executor for debugging and single-threaded operation."""

    def map(self, func: Callable[[T], R], items: list[T]) -> list[R]:
        """Apply function to items sequentially."""
        return [func(item) for item in items]

    def shutdown(self) -> None:
        """No cleanup needed for sequential executor."""
        pass


class MultiprocessingExecutor(Executor):
    """Executor using Python's multiprocessing.Pool."""

    def __init__(self, n_workers: int):
        """
        Initialize multiprocessing executor.

        Args:
            n_workers: Number of parallel workers
        """
        self.n_workers = n_workers
        self._pool: mp.Pool | None = None

    def _get_pool(self) -> mp.Pool:
        """Lazily create the pool."""
        if self._pool is None:
            # Use 'spawn' context for better cross-platform compatibility
            # and to avoid issues with forking and CUDA/OpenBabel
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(processes=self.n_workers)
        return self._pool

    def map(self, func: Callable[[T], R], items: list[T]) -> list[R]:
        """
        Apply function to items using multiprocessing pool.

        Args:
            func: Function to apply (must be picklable)
            items: List of items to process

        Returns:
            List of results
        """
        if not items:
            return []

        pool = self._get_pool()

        # Use imap_unordered for better memory efficiency, but collect in order
        # Actually, use map to maintain order
        try:
            results = pool.map(func, items)
            return list(results)
        except Exception as e:
            logger.error(f"Multiprocessing map failed: {e}")
            raise

    def shutdown(self) -> None:
        """Close and join the pool."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


class DaskExecutor(Executor):
    """Executor using Dask for distributed computing."""

    def __init__(self, n_workers: int, threads_per_worker: int = 1):
        """
        Initialize Dask executor.

        Args:
            n_workers: Number of Dask workers
            threads_per_worker: Threads per worker (default 1 for CPU-bound tasks)
        """
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self._client = None

    def _get_client(self):
        """Lazily create the Dask client."""
        if self._client is None:
            try:
                from dask.distributed import Client, LocalCluster

                cluster = LocalCluster(
                    n_workers=self.n_workers,
                    threads_per_worker=self.threads_per_worker,
                    processes=True,  # Use processes for CPU-bound work
                )
                self._client = Client(cluster)
                logger.info(
                    f"Dask cluster started with {self.n_workers} workers, "
                    f"{self.threads_per_worker} threads each"
                )
            except ImportError:
                raise ImportError(
                    "Dask is not installed. Install with: pip install dask[distributed]"
                )
        return self._client

    def map(self, func: Callable[[T], R], items: list[T]) -> list[R]:
        """
        Apply function to items using Dask.

        Args:
            func: Function to apply
            items: List of items to process

        Returns:
            List of results
        """
        if not items:
            return []

        client = self._get_client()

        try:
            # Submit all tasks
            futures = client.map(func, items)

            # Gather results (maintains order)
            results = client.gather(futures)
            return list(results)

        except Exception as e:
            logger.error(f"Dask map failed: {e}")
            raise

    def shutdown(self) -> None:
        """Close the Dask client and cluster."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"Error closing Dask client: {e}")
            self._client = None


def get_executor(backend: str, n_workers: int) -> Executor:
    """
    Factory function to create the appropriate executor.

    Args:
        backend: "sequential", "multiprocessing", or "dask"
        n_workers: Number of parallel workers

    Returns:
        Configured Executor instance

    Raises:
        ValueError: If backend is not recognized
    """
    if n_workers <= 1:
        logger.info("Using sequential executor (n_workers <= 1)")
        return SequentialExecutor()

    backend = backend.lower()

    if backend == "sequential":
        return SequentialExecutor()
    elif backend == "multiprocessing":
        logger.info(f"Using multiprocessing executor with {n_workers} workers")
        return MultiprocessingExecutor(n_workers)
    elif backend == "dask":
        logger.info(f"Using Dask executor with {n_workers} workers")
        return DaskExecutor(n_workers)
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Must be 'sequential', 'multiprocessing', or 'dask'"
        )
