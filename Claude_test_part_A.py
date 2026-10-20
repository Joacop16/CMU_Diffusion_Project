"""
midterm_checks.py -- one script that checks everything Person A built:
DDPMScheduler, DDIMScheduler and the sampling pipeline (plain, unconditional path).

Run from the repo root (restart the Jupyter kernel first if you edited files,
otherwise stale imports can give misleading results):

    python midterm_checks.py          # or in a notebook cell:  %run midterm_checks.py

Every check prints PASS/FAIL with the numbers. At the end it raises if anything failed.
Where possible the expected value is re-derived by hand from the formulas, so the check
does not depend on the code it is testing.

Note: it is named midterm_checks.py (not test_*.py) so `unittest discover` ignores it.
"""
import numpy as np
import torch

from models import UNet
from pipelines import DDPMPipeline
from schedulers import DDIMScheduler, DDPMScheduler

T, BETA_START, BETA_END = 1000, 1e-4, 0.02
results = []


def check(name, ok, detail=""):
    ok = bool(ok)
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def close(a, b, atol=1e-4, rtol=1e-4):
    return torch.allclose(a, b, atol=atol, rtol=rtol)


def gen(seed):
    return torch.Generator().manual_seed(seed)


def ddpm(**kw):
    return DDPMScheduler(num_train_timesteps=T, beta_start=BETA_START, beta_end=BETA_END, **kw)


def ddim(steps, **kw):
    return DDIMScheduler(num_train_timesteps=T, num_inference_steps=steps,
                         beta_start=BETA_START, beta_end=BETA_END, **kw)


def expected_variance(sch, t):
    """Posterior variance beta_tilde, re-derived by hand from alphas_cumprod."""
    prev = sch.previous_timestep(t)
    a = sch.alphas_cumprod[t]
    a_prev = sch.alphas_cumprod[prev] if prev >= 0 else torch.tensor(1.0)
    return (1 - a_prev) / (1 - a) * (1 - a / a_prev)


def main():
    torch.manual_seed(0)

    # ------------------------------------------------------------------ 1. schedule
    s = ddpm()
    ac = s.alphas_cumprod
    check("1a alphas_cumprod shape/dtype", ac.shape == (T,) and ac.dtype == torch.float32,
          f"{tuple(ac.shape)} {ac.dtype}")
    check("1b endpoints ~0.9999 and ~4e-5",
          abs(ac[0].item() - 0.9999) < 1e-6 and 3.5e-5 < ac[-1].item() < 4.5e-5,
          f"{ac[0].item():.6f}, {ac[-1].item():.3e}")
    check("1c strictly decreasing", (ac[1:] < ac[:-1]).all())
    check("1d alphas = 1 - betas, alphas_cumprod = cumprod(alphas)",
          close(s.alphas, 1 - s.betas, atol=1e-7)
          and torch.allclose(ac, torch.cumprod(s.alphas.double(), 0).float(), rtol=1e-3, atol=1e-8))
    check("1e default timesteps 999..0, int64",
          s.timesteps[0].item() == T - 1 and s.timesteps[-1].item() == 0
          and len(s.timesteps) == T and s.timesteps.dtype == torch.int64)
    check("1f state_dict has exactly betas/alphas/alphas_cumprod (checkpoint-safe)",
          set(s.state_dict().keys()) == {"betas", "alphas", "alphas_cumprod"},
          str(list(s.state_dict().keys())))
    try:
        DDPMScheduler(beta_schedule="nonexistent_schedule")
        raised = False
    except NotImplementedError:
        raised = True
    check("1g unknown beta_schedule raises NotImplementedError", raised)

    # ------------------------------------------------------------------ 2. timesteps
    s = ddpm()
    check("2a previous_timestep is t-1 before set_timesteps", s.previous_timestep(500) == 499)
    s.set_timesteps(200)
    check("2b set_timesteps(200): 995,990,985 ... 0 (int64, len 200)",
          s.timesteps[:3].tolist() == [995, 990, 985] and s.timesteps[-1].item() == 0
          and len(s.timesteps) == 200 and s.timesteps.dtype == torch.int64)
    check("2c previous_timestep uses the stride (500->495, 0->-5)",
          s.previous_timestep(500) == 495 and s.previous_timestep(0) == -5)
    check("2d set_timesteps leaves state_dict unchanged",
          set(s.state_dict().keys()) == {"betas", "alphas", "alphas_cumprod"})

    # ------------------------------------------------------------------ 3. add_noise
    s = ddpm()
    x0 = torch.randn(4, 3, 8, 8)
    eps = torch.randn_like(x0)
    t = torch.tensor([0, 10, 500, 999])
    xt = s.add_noise(x0, eps, t)
    ab = s.alphas_cumprod[t].view(-1, 1, 1, 1)
    expected = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    err = (xt - expected).abs().max().item()
    check("3a add_noise == sqrt(abar)*x0 + sqrt(1-abar)*eps (hand-computed)",
          xt.shape == x0.shape and err < 1e-5, f"max err {err:.2e}")
    d0 = (xt[0] - x0[0]).abs().max().item()
    d999 = (xt[3] - eps[3]).abs().max().item()
    check("3b t=0 is nearly the clean image", d0 < 0.1, f"max diff {d0:.4f}")
    check("3c t=999 is nearly pure noise", d999 < 0.1, f"max diff {d999:.4f}")
    big, e = torch.randn(64, 3, 8, 8), torch.randn(64, 3, 8, 8)
    stds = [s.add_noise(big, e, torch.full((64,), tt)).std().item() for tt in (0, 250, 500, 999)]
    check("3d unit variance preserved at every t (catches wrong coefficients)",
          all(abs(v - 1) < 0.05 for v in stds), str([round(v, 3) for v in stds]))

    # ------------------------------------------------------------------ 4. variance
    s = ddpm()
    ok = True
    for tt in (0, 1, 500, 999):
        got = s._get_variance(torch.tensor(tt))
        want = expected_variance(s, tt).clamp(min=1e-20)
        ok &= torch.allclose(got, want, rtol=1e-4, atol=1e-12)
    check("4a DDPM fixed_small == posterior variance (t=0,1,500,999)", ok)
    check("4b t=0 returns the 1e-20 clamp, not 0 / NaN",
          s._get_variance(torch.tensor(0)).item() == torch.tensor(1e-20).item())
    s200 = ddpm()
    s200.set_timesteps(200)
    check("4c stride-5 variance matches hand formula at t=995",
          torch.allclose(s200._get_variance(torch.tensor(995)), expected_variance(s200, 995),
                         rtol=1e-4, atol=1e-12))
    sl = ddpm(variance_type="fixed_large")
    check("4d fixed_large: ~beta_999 at t=999, tiny at t=0",
          torch.isclose(sl._get_variance(torch.tensor(999)), sl.betas[999], rtol=1e-3)
          and sl._get_variance(torch.tensor(0)).item() <= 1e-19)
    d = ddim(200)
    check("4e DDIM variance: exactly 0 at last step, equals DDPM's formula elsewhere",
          d._get_variance(torch.tensor(0)).item() == 0.0
          and torch.isclose(d._get_variance(torch.tensor(995)), s200._get_variance(torch.tensor(995)),
                            rtol=1e-4))

    # ------------------------------------------------------------------ 5. DDPM step
    s = ddpm(clip_sample=False)  # clipping would hide errors with randn x0
    x0 = torch.randn(2, 3, 8, 8)
    eps = torch.randn_like(x0)
    xt0 = s.add_noise(x0, eps, torch.tensor([0, 0]))
    err = (s.step(eps, torch.tensor(0), xt0) - x0).abs().max().item()
    check("5a last step with true noise returns x0", err < 1e-4, f"max err {err:.2e}")
    t_mid = 500
    xt = s.add_noise(x0, eps, torch.tensor([t_mid, t_mid]))
    a_t, a_p = s.alphas_cumprod[t_mid], s.alphas_cumprod[t_mid - 1]
    alpha_t = a_t / a_p
    beta_t = 1 - alpha_t
    mu = (a_p.sqrt() * beta_t / (1 - a_t)) * x0 + (alpha_t.sqrt() * (1 - a_p) / (1 - a_t)) * xt
    var = (1 - a_p) / (1 - a_t) * beta_t
    want = mu + var.sqrt() * torch.randn(x0.shape, generator=gen(0))
    got = s.step(eps, torch.tensor(t_mid), xt, generator=gen(0))
    err = (got - want).abs().max().item()
    check("5b step == hand-derived posterior mean (Eq. 7) + sigma*z", err < 1e-4, f"max err {err:.2e}")
    check("5c step is finite at t=999",
          torch.isfinite(s.step(eps, torch.tensor(999), torch.randn_like(x0))).all())

    # ------------------------------------------------------------------ 6. DDIM step
    d = ddim(200, clip_sample=False)
    xt = d.add_noise(x0, eps, torch.tensor([500, 500]))
    want = d.add_noise(x0, eps, torch.tensor([495, 495]))
    err = (d.step(eps, torch.tensor(500), xt) - want).abs().max().item()
    check("6a eta=0 + true noise lands exactly on the forward marginal at t-5", err < 1e-4,
          f"max err {err:.2e}")
    xt0 = d.add_noise(x0, eps, torch.tensor([0, 0]))
    err = (d.step(eps, torch.tensor(0), xt0) - x0).abs().max().item()
    check("6b last step with true noise returns x0", err < 1e-4, f"max err {err:.2e}")
    check("6c eta=0 is deterministic",
          torch.equal(d.step(eps, torch.tensor(500), xt), d.step(eps, torch.tensor(500), xt)))
    r1 = d.step(eps, torch.tensor(500), xt, generator=gen(0), eta=1.0)
    r2 = d.step(eps, torch.tensor(500), xt, generator=gen(0), eta=1.0)
    r3 = d.step(eps, torch.tensor(500), xt, generator=gen(1), eta=1.0)
    check("6d eta=1: same seed reproduces, different seed differs",
          torch.equal(r1, r2) and not torch.equal(r1, r3))
    xr, er = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
    a = ddim(1000, clip_sample=False).step(er, torch.tensor(500), xr, generator=gen(7), eta=1.0)
    b = ddpm(clip_sample=False).step(er, torch.tensor(500), xr, generator=gen(7))
    err = (a - b).abs().max().item()
    check("6e DDIM with eta=1 and 1000 steps == DDPM step (the link between them)", err < 1e-4,
          f"max err {err:.2e}")
    try:
        DDIMScheduler(num_train_timesteps=T, beta_start=BETA_START, beta_end=BETA_END)
        raised = False
    except AssertionError:
        raised = True
    check("6f DDIM without num_inference_steps asserts", raised)

    # ------------------------------------------------------------------ 7. pipeline smoke test
    unet = UNet(input_size=8, input_ch=3, T=10, ch=32, ch_mult=[1, 2], attn=[1], num_res_blocks=1).eval()
    schedulers = {
        "DDPM": DDPMScheduler(num_train_timesteps=10, beta_start=BETA_START, beta_end=BETA_END),
        "DDIM": DDIMScheduler(num_train_timesteps=10, num_inference_steps=5,
                              beta_start=BETA_START, beta_end=BETA_END),
    }
    for name, sched in schedulers.items():
        pipe = DDPMPipeline(unet, sched)
        imgs = pipe(batch_size=2, num_inference_steps=5, generator=gen(0), device="cpu")
        imgs_again = pipe(batch_size=2, num_inference_steps=5, generator=gen(0), device="cpu")
        check(f"7 pipeline + {name}: 2 images of size (8, 8)",
              len(imgs) == 2 and imgs[0].size == (8, 8))
        check(f"7 pipeline + {name}: timesteps [8,6,4,2,0]", sched.timesteps.tolist() == [8, 6, 4, 2, 0])
        check(f"7 pipeline + {name}: same seed gives identical images",
              all(np.array_equal(np.array(p), np.array(q)) for p, q in zip(imgs, imgs_again)))

    # ------------------------------------------------------------------ summary
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        raise AssertionError("Failed checks: " + "; ".join(failed))
    print("All checks passed.")


if __name__ == "__main__":
    main()
