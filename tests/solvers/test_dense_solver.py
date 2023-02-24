import numpy as np
import pytest
import torch

from fugw.solvers import FUGWSolver


@pytest.mark.parametrize("uot_solver", ["sinkhorn", "mm", "ibpp"])
def test_solvers(uot_solver):
    torch.manual_seed(0)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    ns = 104
    ds = 3
    nt = 151
    dt = 7
    nf = 10

    source_features = torch.rand(ns, nf).to(device)
    target_features = torch.rand(nt, nf).to(device)
    source_embeddings = torch.rand(ns, ds).to(device)
    target_embeddings = torch.rand(nt, dt).to(device)

    F = torch.cdist(source_features, target_features)
    Ds = torch.cdist(source_embeddings, source_embeddings)
    Dt = torch.cdist(target_embeddings, target_embeddings)

    Ds_normalized = Ds / Ds.max()
    Dt_normalized = Dt / Dt.max()
    F_normalized = F / F.max()

    nits_bcd = 100
    eval_bcd = 2
    fugw = FUGWSolver(
        nits_bcd=nits_bcd,
        nits_uot=1000,
        tol_bcd=1e-7,
        tol_uot=1e-7,
        early_stopping_threshold=1e-5,
        eval_bcd=eval_bcd,
        eval_uot=10,
        ibpp_eps_base=1e2,
    )

    pi, gamma, duals_pi, duals_gamma, loss_steps, loss, loss_ent = fugw.solve(
        alpha=0.8,
        rho_s=2,
        rho_t=3,
        eps=0.02,
        reg_mode="independent",
        F=F_normalized,
        Ds=Ds_normalized,
        Dt=Dt_normalized,
        init_plan=None,
        uot_solver=uot_solver,
        verbose=True,
    )

    assert pi.shape == (ns, nt)
    assert gamma.shape == (ns, nt)

    if uot_solver == "mm":
        assert duals_pi is None
        assert duals_gamma is None
    else:
        assert len(duals_pi) == 2
        assert duals_pi[0].shape == (ns,)
        assert duals_pi[1].shape == (nt,)
        assert len(duals_gamma) == 2
        assert duals_gamma[0].shape == (ns,)
        assert duals_gamma[1].shape == (nt,)

    assert len(loss_steps) <= nits_bcd // eval_bcd + 1
    assert len(loss_steps) == len(loss)
    assert len(loss) == len(loss_ent)
    # Loss should decrease
    assert np.all(np.sign(np.array(loss[1:]) - np.array(loss[:-1])) == -1)