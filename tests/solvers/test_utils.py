import torch
import ot
import pytest
from fugw.solvers.utils import (
    solver_sinkhorn_stabilized_sparse,
    solver_sinkhorn_eps_scaling_sparse,
    solver_sinkhorn_stabilized,
    solver_sinkhorn_eps_scaling,
)


@pytest.mark.parametrize(
    "pot_method, solver, is_log",
    [
        ("sinkhorn_stabilized", solver_sinkhorn_stabilized, False),
        ("sinkhorn_epsilon_scaling", solver_sinkhorn_eps_scaling, False),
    ],
)
def test_solvers_sinkhorn(pot_method, solver, is_log):
    ns = 151
    nt = 104
    nf = 10
    eps = 1.0

    niters, tol, eval_freq = train_params = 100, 0, 10

    ws = torch.ones(ns) / ns
    wt = torch.ones(nt) / nt

    source_features = torch.rand(ns, nf)
    target_features = torch.rand(nt, nf)

    cost = torch.cdist(source_features, target_features)
    ws_dot_wt = torch.ones(ns, nt) / (ns * nt)
    init_duals = (torch.zeros(ns), torch.zeros(nt))
    tuple_weights = ws, wt, ws_dot_wt

    uot_params = torch.tensor(float("inf")), torch.tensor(float("inf")), eps

    _, log = ot.sinkhorn(
        ws,
        wt,
        cost,
        eps,
        numItermax=niters,
        stopThr=tol,
        method=pot_method,
        log=True,
    )

    # Check the log of the scaling vectors
    if is_log:
        (log_u, log_v), _ = solver(
            cost,
            init_duals,
            uot_params,
            tuple_weights,
            train_params,
        )
        assert torch.allclose(log["log_u"], log_u)
        assert torch.allclose(log["log_v"], log_v)

    # Check the potentials
    else:
        (alpha, beta), _ = solver(
            cost,
            init_duals,
            uot_params,
            tuple_weights,
            train_params,
        )

        assert torch.allclose(
            log["alpha"],
            alpha,
        )
        assert torch.allclose(log["beta"], beta)


@pytest.mark.parametrize(
    "pot_method, solver, is_log",
    [
        ("sinkhorn_stabilized", solver_sinkhorn_stabilized_sparse, False),
        (
            "sinkhorn_epsilon_scaling",
            solver_sinkhorn_eps_scaling_sparse,
            False,
        ),
    ],
)
def test_solvers_sinkhorn_sparse(pot_method, solver, is_log):
    ns = 151
    nt = 104
    nf = 10
    eps = 1.0

    niters, tol, eval_freq = train_params = 100, 0, 10

    ws = torch.ones(ns) / ns
    wt = torch.ones(nt) / nt

    source_features = torch.rand(ns, nf)
    target_features = torch.rand(nt, nf)

    cost = torch.cdist(source_features, target_features)
    ws_dot_wt = torch.ones(ns, nt) / (ns * nt)
    init_duals = (torch.zeros(ns), torch.zeros(nt))
    tuple_weights = ws, wt, ws_dot_wt.to_sparse_csr()

    uot_params = torch.tensor(float("inf")), torch.tensor(float("inf")), eps

    gamma, log = ot.sinkhorn(
        ws,
        wt,
        cost,
        eps,
        numItermax=niters,
        stopThr=tol,
        method=pot_method,
        log=True,
    )

    # Check the log of the scaling vectors
    if is_log:
        (log_u, log_v), _ = solver(
            cost.to_sparse_csr(),
            init_duals,
            uot_params,
            tuple_weights,
            train_params,
        )
        assert torch.allclose(log["log_u"], log_u)
        assert torch.allclose(log["log_v"], log_v)

    # Check the potentials
    else:
        (alpha, beta), pi = solver(
            cost.to_sparse_csr(),
            init_duals,
            uot_params,
            tuple_weights,
            train_params,
        )

        assert torch.allclose(
            log["alpha"],
            alpha,
        )
        assert torch.allclose(log["beta"], beta)
        assert torch.allclose(gamma, pi.to_dense(), atol=1e-5)
