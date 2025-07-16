import os
import sys
import time

import numpy as np
import pandas as pd

import pyro
import pyro.optim
from pyro import poutine, get_param_store
from pyro.infer.autoguide import AutoDelta, AutoDiscreteParallel
from pyro.infer.autoguide.initialization import init_to_sample
from pyro.infer import SVI, TraceEnum_ELBO, infer_discrete

import torch
import torch.nn.functional as F

from sc_data import SC_Data
from sc_model import SC_Params


def validate_model(
    sc_data: SC_Data,
    sc_params: SC_Params,
    model=None,
    validate=True,
    plot_model=True,
    out_dir=None,
):
    pyro.enable_validation(False)
    pyro.clear_param_store()
    optim = pyro.optim.Adam({"lr": 0.1, "betas": [0.8, 0.99]})
    elbo = TraceEnum_ELBO(max_plate_nesting=1)
    global_guide = AutoDelta(
        poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("expose_")),
        init_loc_fn=init_to_sample,
    )
    svi = SVI(model, global_guide, optim, loss=elbo)
    loss = svi.loss(model, global_guide, sc_data, sc_params)
    print(f"validate-loss={loss}")

    guide_trace = poutine.trace(global_guide).get_trace(sc_data, sc_params)
    replay = poutine.replay(model, guide_trace)
    model_trace = poutine.trace(replay).get_trace(sc_data, sc_params)
    model_lp = model_trace.log_prob_sum()
    guide_lp = guide_trace.log_prob_sum()
    print("model log_prob_sum:", model_lp)
    print("guide log_prob_sum:", guide_lp)
    elbo = guide_lp - model_lp
    print("elbo =", elbo)

    if validate:
        print("=========================Guide=========================")
        guide_trace.compute_log_prob()
        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample":
                value = site["value"]
                lp = site["log_prob"]
                print(f"{name}: shape={site['value'].shape}")
                if (
                    torch.isnan(value).any()
                    or torch.isnan(lp).any()
                    or torch.isinf(lp).any()
                ):
                    print(f"value NaN detected at site: {name}, value={value}, lp={lp}")
                    sys.exit(1)

        print("=========================Model=========================")
        model_trace.compute_log_prob()
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                value = site["value"]
                lp = site["log_prob"]
                print(f"{name}: shape={site['value'].shape}")
                if (
                    torch.isnan(value).any()
                    or torch.isnan(lp).any()
                    or torch.isinf(lp).any()
                ):
                    print(f"value NaN detected at site: {name}, value={value}, lp={lp}")
                    sys.exit(1)
    if plot_model:
        dot = pyro.render_model(
            model,
            model_args=(sc_data, sc_params),
            render_distributions=True,
            render_params=True,
        )
        dot.graph_attr.update(
            # rankdir="TB",       # top-to-bottom layout
            ranksep="1.0",  # more space between ranks
            nodesep="1.0",  # more space between nodes
            fontsize="12",  # smaller font
            dpi="300",
        )
        dot.node_attr.update(fontsize="12")
        dot.render(os.path.join(out_dir, "model"), format="png", cleanup=True)
    return


def run_model(
    sc_data: SC_Data, sc_params: SC_Params, model=None, curr_repeat=1, out_dir=None
):
    pyro.enable_validation(False)
    torch.autograd.set_detect_anomaly(True)

    # TODO move this out
    start_time = round(time.time() * 1000)
    learning_rate = 0.1
    anneal_rate = 0.01
    min_temp = 0.5
    max_temp = 1.0
    min_iter = 10
    max_iter = 400
    rel_tol = 5e-5

    np_temp = max_temp
    losses = []

    init_seed = True
    init_niter = 1

    optim = pyro.optim.Adam({"lr": learning_rate, "betas": [0.8, 0.99]})
    elbo = TraceEnum_ELBO(max_plate_nesting=1)

    ##################################################
    def init_model(seed):
        pyro.set_rng_seed(seed)
        torch.manual_seed(seed)
        pyro.clear_param_store()

        global_guide = AutoDelta(
            poutine.block(
                model, expose_fn=lambda msg: msg["name"].startswith("expose_")
            ),
            init_loc_fn=init_to_sample,
        )
        svi = SVI(model, global_guide, optim, loss=elbo)
        print(f"compute loss {seed}")
        loss = svi.loss(model, global_guide, sc_data, sc_params)
        print(f"init model seed={seed} loss={loss}")
        return loss

    if init_seed:
        init_seeds = list(
            range(curr_repeat * init_niter, (curr_repeat + 1) * init_niter)
        )
        loss, seed = min((init_model(seed), seed) for seed in init_seeds)
        losses.append(loss)
        print(f"seed={seed}, init-loss={loss}")
    else:
        seed = curr_repeat * init_niter
        print(f"seed={seed}")

    ##################################################
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    pyro.clear_param_store()

    global_guide = AutoDelta(
        poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("expose_")),
        init_loc_fn=init_to_sample,
    )

    svi = SVI(model, global_guide, optim, loss=elbo)

    ##################################################
    for curr_iter in range(max_iter):
        # if curr_iter % 100 == 1:
        #     # temperature annealing for Gumbel-softmax (RelaxedOneHotCategorical)
        #     np_temp = np.maximum(
        #         max_temp * np.exp(-anneal_rate * curr_iter), min_temp
        #     )
        #     print(f"iteration={curr_iter}, drop temperature to {np_temp}")
        curr_loss = svi.step(sc_data, sc_params)
        losses.append(curr_loss)
        print(f"iteration={curr_iter}\tloss={curr_loss}")
        # if torch.isinf(torch.tensor(curr_loss)):
        #     print("failed")
        #     for name, value in pyro.get_param_store().items():
        #         # if site["type"] == "sample" and site["is_observed"] is False:
        #         # lp = value["log_prob"]
        #         print(f"{name} {type(value)} {value}")
        #         # if value.grad is not None and torch.isinf(value.grad).any():
        #         #     print(f"[GRAD INF] {name}")
        #         # if value.grad is not None and torch.isnan(value.grad).any():
        #         #     print(f"[GRAD NAN] {name}")
        #     sys.exit(1)
        if curr_iter >= min_iter:
            prev_loss = losses[-2]
            loss_diff = abs((curr_loss - prev_loss) / prev_loss)
            if loss_diff < rel_tol:
                print(f"ELBO converged at iteration {curr_iter}, loss-diff={loss_diff}")
                break

    ##################################################
    # guide_trace = poutine.trace(global_guide).get_trace(sc_data, sc_params)  # record the globals
    # trained_model = poutine.replay(model, trace=guide_trace)  # replay the globals
    # inferred_model = infer_discrete(
    #     trained_model, temperature=0, first_available_dim=-2
    # )  # avoid conflict with sc_data plate
    # trace = poutine.trace(inferred_model).get_trace(sc_data, sc_params)
    # zs: torch.Tensor = trace.nodes["z"]["value"]

    map_estimates = global_guide()
    # store results to LOG file
    for name in map_estimates:
        out_file = os.path.join(out_dir, f"{name}.tsv")
        map_res = map_estimates[name].detach().cpu().numpy()
        pd.DataFrame(map_res).to_csv(out_file, sep="\t", index=False)

    for name, value in get_param_store().items():
        print(f"{name}: {value.data}")

    loss_df = pd.DataFrame({"loss": losses})
    loss_df.to_csv(
        os.path.join(out_dir, "loss.tsv"), sep="\t", index=False, header=True
    )

    end_time = round(time.time() * 1000)
    print(f"rep={curr_repeat}\ttotal-elapsed={end_time - start_time}ms")
    return
