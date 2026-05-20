from __future__ import annotations

import pytest

from agentrl_infra.resources import InMemoryEnvironmentPool, ResourceState


def test_environment_pool_lease_release_cycle() -> None:
    pool = InMemoryEnvironmentPool()

    lease = pool.register("env-1")
    assert lease.state == ResourceState.READY

    leased = pool.lease("env-1", "worker-1")
    assert leased.state == ResourceState.LEASED

    released = pool.release("env-1", healthy=False, snapshot_id="snap-1")
    assert released.state == ResourceState.UNHEALTHY
    assert released.snapshot_id == "snap-1"


def test_environment_pool_rejects_double_lease() -> None:
    pool = InMemoryEnvironmentPool()
    pool.lease("env-1", "worker-1")

    with pytest.raises(RuntimeError):
        pool.lease("env-1", "worker-2")
