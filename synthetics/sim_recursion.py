import os
import argparse
import itertools
import time

import torch
import pandas as pd


def run_one_closed_form(
    *,
    alpha: float,
    beta_mul: int,
    sigma2: float,
    eta: float,
    sqrt_coeff: float,
    algorithm: str,
    ema: float,                 # ema==0 means "no averaging"
    seed: int,                  # kept for interface parity (recursion is deterministic)
    device: torch.device,
    dtype: torch.dtype,
    output_dir: str,
    d: int,
    N_default: int,
    # WSD-only
    wsd_p: float = None,
    wsd_T: int = None,
):
    start_time = time.time()

    # -----------------------------
    # Training length depends on algorithm
    # -----------------------------
    if algorithm == "wsd":
        if wsd_T is None:
            raise ValueError("For algorithm='wsd', you must pass wsd_T.")
        if wsd_p is None:
            raise ValueError("For algorithm='wsd', you must pass wsd_p.")
        N = int(wsd_T)
        p = float(wsd_p)
        if not (0.0 < p < 1.0):
            raise ValueError(f"wsd_p must be in (0,1), got {p}")
    else:
        N = int(N_default)
        p = None

    beta = beta_mul * alpha
    a = float(sqrt_coeff)

    # -----------------------------
    # Build spectrum + teacher (O(d))
    # -----------------------------
    i = torch.arange(1, d + 1, dtype=dtype, device=device)
    lam = i.pow(-alpha)
    lam2 = lam * lam

    # Target w*: enforce lambda_i * w_i^2 ~ i^{-beta}
    w2_unnorm = i.pow(-(beta - alpha))
    c = 1.0 / w2_unnorm.sum()
    w_star = (c * w2_unnorm).sqrt()

    # Base LR scale (python float)
    eta0 = float(eta) / lam.sum().item()
    sigma2 = float(sigma2)

    # Times to evaluate (up to N)
    fixed_eval = [500, 1000, 2000, 3000, 5000, 8000, 10000, 20000, 50000]
    fixed_eval = [t for t in fixed_eval if t <= N]

    Ts = torch.unique(torch.cat([
        torch.round(torch.logspace(0, torch.log10(torch.tensor(float(N))), steps=80)).to(torch.int64),
        torch.tensor(fixed_eval, dtype=torch.int64),
    ])).cpu().numpy()
    Ts.sort()
    K = len(Ts)

    # Save LR at each checkpoint
    lrs = torch.zeros(K, dtype=torch.float32)

    # -----------------------------
    # Closed-form recursion state for error e_t = w_t - w*
    # m_i = E[e_i], s_i = E[e_i^2]
    # -----------------------------
    m = (-w_star).clone()
    s = (w_star * w_star).clone()

    use_avg = (ema != 0.0)
    if use_avg:
        mbar = m.clone()
        sbar = s.clone()
        cbar = s.clone()  # cbar_i = E[bar_e_i * e_i]

    # Buffers
    tmp1 = torch.empty_like(lam)  # (1 - eta_t * lam)
    tmp2 = torch.empty_like(lam)  # (1 - 2 eta_t lam + 2 eta_t^2 lam^2)
    c_to_e_buf = torch.empty_like(lam) if use_avg else None

    # Output tensors (CPU)
    total = torch.zeros(K, dtype=torch.float32)
    bias_part = torch.zeros(K, dtype=torch.float32)
    var_part = torch.zeros(K, dtype=torch.float32)

    next_k = 0
    next_T = int(Ts[next_k])

    # Precompute WSD boundary
    if algorithm == "wsd":
        t_const_end = int(round(p * N))
        t_const_end = max(1, min(t_const_end, N))
        denom = max(1, (N - t_const_end))

    # timing (GPU)
    if device.type == "cuda":
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
    else:
        ev0 = ev1 = None

    for t in range(1, N + 1):
        # Step size schedule (python float)
        if algorithm == "constant":
            eta_t = eta0
        elif algorithm == "sqrt":
            eta_t = eta0 * (((a + t) / a) ** (-0.5))
        elif algorithm == "wsd":
            if t <= t_const_end:
                eta_t = eta0
            else:
                frac = (t - t_const_end) / denom
                eta_t = eta0 * (1.0 - frac)
                if eta_t < 0.0:
                    eta_t = 0.0
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        et = float(eta_t)
        et2 = et * et

        # tmp1 = 1 - et*lam  (compute once; reuse if averaging)
        tmp1.copy_(lam).mul_(-et).add_(1.0)

        # mean update
        m.mul_(tmp1)

        # r_prev = <lam, s>
        r_prev = torch.dot(lam, s)

        # tmp2 = 1 - 2et lam + 2et^2 lam^2
        tmp2.copy_(lam).mul_(-2.0 * et).add_(1.0)
        tmp2.add_(lam2, alpha=(2.0 * et2))

        # second moment update
        s.mul_(tmp2)
        s.add_(lam, alpha=(et2) * (r_prev + sigma2))

        # Averaging moments (exact for rho_t=min(ema,t)/t)
        if use_avg:
            rho = min(float(ema), float(t)) / float(t)
            one_m_rho = 1.0 - rho

            # c_to_e = (1 - et*lam) * cbar  (reuse tmp1)
            c_to_e_buf.copy_(cbar).mul_(tmp1)

            # mbar = (1-rho)mbar + rho m
            mbar.mul_(one_m_rho).add_(m, alpha=rho)

            # sbar = (1-r)^2 sbar + r^2 s + 2r(1-r)c_to_e
            sbar.mul_(one_m_rho * one_m_rho)
            sbar.add_(s, alpha=(rho * rho))
            sbar.add_(c_to_e_buf, alpha=(2.0 * rho * one_m_rho))

            # cbar = (1-r) c_to_e + r s
            cbar.copy_(c_to_e_buf).mul_(one_m_rho).add_(s, alpha=rho)

        # checkpoint
        if t == next_T:
            lrs[next_k] = et

            if use_avg:
                total_t = torch.dot(lam, sbar).item()
                bias_t = torch.dot(lam, mbar * mbar).item()
            else:
                total_t = torch.dot(lam, s).item()
                bias_t = torch.dot(lam, m * m).item()

            total[next_k] = float(total_t)
            bias_part[next_k] = float(bias_t)
            var_part[next_k] = max(0.0, float(total_t - bias_t))

            next_k += 1
            if next_k >= K:
                break
            next_T = int(Ts[next_k])

    if ev1:
        ev1.record()
        torch.cuda.synchronize()
        wall = ev0.elapsed_time(ev1) / 1000.0
    else:
        wall = time.time() - start_time

    # -----------------------------
    # Saving Results
    # -----------------------------
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame({
        "step": Ts,
        "lr": lrs.numpy(),
        "total_risk": total.numpy(),
        "bias": bias_part.numpy(),
        "variance": var_part.numpy(),
    })

    base = (
        f"alg_{algorithm}"
        f"_alpha_{alpha:g}"
        f"_betaMul_{beta_mul}"
        f"_eta_{eta:g}"
        f"_sigma2_{sigma2:g}"
        f"_seed_{seed}"
        f"_d_{d}"
    )
    if algorithm == "wsd":
        tag = f"{base}_p_{wsd_p:g}_T_{int(wsd_T)}"
    else:
        tag = f"{base}_a_{a:g}_ema_{ema:g}_N_{N}"

    csv_path = os.path.join(output_dir, f"{tag}.csv")
    df.to_csv(csv_path, index=False)

    print(f"[DONE in {wall:.2f}s] {tag}", flush=True)


def main():
    print("STARTING JOB", flush=True)
    parser = argparse.ArgumentParser(description="Closed-form SGD moment recursion sweep (Torch/GPU)")

    # Sweepable args (lists)
    parser.add_argument("--alpha", type=float, nargs="+", default=[1.1, 1.5, 1.9])
    parser.add_argument("--beta_mul", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    parser.add_argument("--sigma2", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1])
    parser.add_argument("--eta", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0, 1.25, 1.5, 1.9])
    parser.add_argument("--sqrt_coeff", type=float, nargs="+", default=[400, 800, 1600, 3200, 6400, 12800, 25600])
    parser.add_argument("--algorithm", type=str, nargs="+", default=["constant", "sqrt", "wsd"], choices=["constant", "sqrt", "wsd"])
    parser.add_argument("--ema", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])

    # WSD sweep args (durations you requested)
    parser.add_argument("--wsd_T", type=int, nargs="+",
                        default=[1000, 2000, 3000, 5000, 8000, 10000, 20000, 50000])
    parser.add_argument("--wsd_p", type=float, nargs="+",
                        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Chunking / job splitting
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_idx", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10,
                        help="Print progress every N configs within this job slice")

    # Non-swept args
    parser.add_argument("--output_dir", type=str, default="./results_closed_form")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32")

    # New non-swept knobs
    parser.add_argument("--d", type=int, default=200_000)
    parser.add_argument("--N", type=int, default=50_000, help="Used when algorithm != wsd")

    args = parser.parse_args()

    # Device / dtype
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    # Build configs:
    # - If algorithm != wsd: sweep sqrt_coeff and ema normally
    # - If algorithm == wsd: DO NOT sweep ema; DO NOT sweep sqrt_coeff; force ema=0.0 and sqrt_coeff=1.0
    configs = []

    # Non-WSD grid
    non_wsd_algs = [str(a) for a in args.algorithm if str(a) != "wsd"]
    if len(non_wsd_algs) > 0:
        non_wsd_grid = list(itertools.product(
            args.alpha,
            args.beta_mul,
            args.sigma2,
            args.eta,
            args.sqrt_coeff,
            non_wsd_algs,
        ))
        for alpha, beta_mul, sigma2, eta, sqrt_coeff, algorithm in non_wsd_grid:
            for ema in args.ema:
                configs.append((alpha, beta_mul, sigma2, eta, sqrt_coeff, algorithm, float(ema), None, None))

    # WSD grid (sqrt_coeff fixed to 1.0; ema fixed to 0.0)
    if "wsd" in [str(a) for a in args.algorithm]:
        wsd_grid = list(itertools.product(
            args.alpha,
            args.beta_mul,
            args.sigma2,
            args.eta,
            args.wsd_T,
            args.wsd_p,
        ))
        for alpha, beta_mul, sigma2, eta, wsd_T, wsd_p in wsd_grid:
            configs.append((alpha, beta_mul, sigma2, eta, 1.0, "wsd", 0.0, float(wsd_p), int(wsd_T)))

    total_cfgs = len(configs)

    # Chunk across jobs
    num_jobs = int(args.num_jobs)
    job_idx = int(args.job_idx)
    assert 0 <= job_idx < num_jobs, f"job_idx must be in [0, {num_jobs-1}]"

    chunk = (total_cfgs + num_jobs - 1) // num_jobs
    start = job_idx * chunk
    end = min(total_cfgs, start + chunk)
    slice_len = max(0, end - start)

    print(
        f"Total configs={total_cfgs}. "
        f"Job {job_idx}/{num_jobs} runs indices [{start}, {end}) = {slice_len} configs "
        f"on device={device}, dtype={dtype}, d={args.d}, N={args.N}",
        flush=True
    )

    if slice_len == 0:
        print("Nothing to run for this job slice; exiting.", flush=True)
        return

    job_start_time = time.time()
    for local_idx, combo_global_idx in enumerate(range(start, end), start=1):
        alpha, beta_mul, sigma2, eta, sqrt_coeff, algorithm, ema, wsd_p, wsd_T = configs[combo_global_idx]

        if args.log_every > 0 and (local_idx == 1 or local_idx % args.log_every == 0 or combo_global_idx == end - 1):
            elapsed = time.time() - job_start_time
            rate = local_idx / max(elapsed, 1e-9)
            remaining = (slice_len - local_idx) / max(rate, 1e-9)

            extra = f" p={wsd_p} T={wsd_T}" if str(algorithm) == "wsd" else f" ema={ema}"
            extra2 = "" if str(algorithm) == "wsd" else f" a={sqrt_coeff}"

            print(
                f"[PROGRESS] job={job_idx}/{num_jobs} "
                f"{local_idx}/{slice_len} (global_idx={combo_global_idx}) "
                f"elapsed={elapsed:.1f}s eta={eta} alpha={alpha} beta_mul={beta_mul} sigma2={sigma2} "
                f"alg={algorithm}{extra}{extra2} "
                f"~{remaining/60:.1f} min left",
                flush=True
            )

        run_one_closed_form(
            alpha=float(alpha),
            beta_mul=int(beta_mul),
            sigma2=float(sigma2),
            eta=float(eta),
            sqrt_coeff=float(sqrt_coeff),
            algorithm=str(algorithm),
            ema=float(ema),
            seed=int(args.seed),
            device=device,
            dtype=dtype,
            output_dir=args.output_dir,
            d=int(args.d),
            N_default=int(args.N),
            wsd_p=(None if wsd_p is None else float(wsd_p)),
            wsd_T=(None if wsd_T is None else int(wsd_T)),
        )

    total_elapsed = time.time() - job_start_time
    print(f"[JOB DONE] job={job_idx}/{num_jobs} ran {slice_len} configs in {total_elapsed/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
