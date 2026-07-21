"""Gated auto-launch of the 48-config minibatch array.

Waits for the two validation jobs to finish, checks each against explicit pass criteria, and ONLY
if both pass submits the staged 48-config array. If anything fails or is ambiguous it logs why and
does NOT launch -- a failed criterion means the settings are not ready, not that they should be
forced through.

  real 70% run (19286831)  -> real70/result.json
     PASS if: run completed, psi_location_err < 0.06 (psi at the observed rate),
              psi_contrast > 0.04 (programs differ, above the collapse floor),
              phi_perplexity < 0.9*K (cells commit), final_loss finite (no un-recovered blow-up),
              n_finite_steps == total_steps (clean, no NaN truncation)
  sim minibatch-vs-fullbatch (19286867) -> the CHECKS block in mb_vs_full.log
     PASS if all four are True: subsampling engaged, recovery within 0.02 of full-batch,
              beta_slope in [0.9,1.1], converged (late psi drift < 0.01)
"""
import json, os, re, subprocess, time, sys

BASE = "/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch"
REAL_JOB, SIM_JOB = "19286831", "19286867"
REAL_JSON = f"{BASE}/real70/result.json"
SIM_LOG = f"{BASE}/mb_vs_full.log"
ARRAY_SH = ("/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/"
            "LEAFLETFA_MODELS/2026-07-18/leaflet_minibatch_array.sh")
LOG = f"{BASE}/gated_launch.log"


def say(msg):
    line = f"[{time.strftime('%m-%d %H:%M')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def state(job):
    out = subprocess.run(["sacct", "-j", job, "-n", "-X", "--format=State"],
                         capture_output=True, text=True).stdout
    st = [s.strip() for s in out.splitlines() if s.strip()]
    return st[0] if st else "UNKNOWN"


def done(job):
    return state(job).split()[0] in {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED",
                                     "OUT_OF_MEMORY", "NODE_FAIL"}


def check_real():
    if not os.path.exists(REAL_JSON):
        return False, "result.json missing (run did not reach the end)"
    r = json.load(open(REAL_JSON))
    K = r.get("K", 20)
    reasons, ok = [], True
    def req(cond, msg):
        nonlocal ok
        reasons.append(("PASS" if cond else "FAIL") + " " + msg)
        ok = ok and cond
    le = r.get("psi_location_err")
    req(le is not None and le < 0.06, f"psi_location_err={le} < 0.06")
    req((r.get("psi_contrast") or 0) > 0.04, f"psi_contrast={r.get('psi_contrast')} > 0.04")
    req((r.get("phi_perplexity") or 1e9) < 0.9 * K,
        f"phi_perplexity={r.get('phi_perplexity')} < {0.9*K:.1f}")
    req(r.get("final_loss") is not None, f"final_loss finite ({r.get('final_loss')})")
    req(r.get("n_finite_steps") == r.get("total_steps"),
        f"n_finite_steps={r.get('n_finite_steps')} == total_steps={r.get('total_steps')}")
    return ok, "; ".join(reasons)


def check_sim():
    if not os.path.exists(SIM_LOG):
        return False, "sim log missing"
    txt = open(SIM_LOG).read()
    pats = {
        "subsampling engaged": r"subsampling engaged for every minibatch run:\s*(True|False)",
        "recovery within 0.02": r"minibatch recovery within 0\.02 of full-batch:\s*(True|False)",
        "beta_slope 0.9-1.1": r"minibatch beta_slope in \[0\.9, 1\.1\][^:]*:\s*(True|False)",
        "converged": r"minibatch converged \(late psi drift < 0\.01\):\s*(True|False)",
    }
    reasons, ok = [], True
    for label, pat in pats.items():
        mobj = re.search(pat, txt)
        val = (mobj.group(1) == "True") if mobj else False
        reasons.append(("PASS" if val else "FAIL") + " " + label + ("" if mobj else " (not found)"))
        ok = ok and val
    return ok, "; ".join(reasons)


def already_launched():
    out = subprocess.run(["squeue", "-u", "kisaev", "-h", "-n", "lfa_mb", "-o", "%i"],
                         capture_output=True, text=True).stdout
    return bool(out.strip())


def main():
    say(f"=== gated launcher started; waiting on real={REAL_JOB} sim={SIM_JOB} ===")
    while not (done(REAL_JOB) and done(SIM_JOB)):
        time.sleep(300)
    say(f"both finished: real={state(REAL_JOB)}  sim={state(SIM_JOB)}")

    if already_launched():
        say("array already running (lfa_mb present) -- not launching again. Exit."); return

    real_ok, real_why = check_real()
    sim_ok, sim_why = check_sim()
    say(f"REAL  {'PASS' if real_ok else 'FAIL'}: {real_why}")
    say(f"SIM   {'PASS' if sim_ok else 'FAIL'}: {sim_why}")

    if real_ok and sim_ok:
        jid = subprocess.run(["sbatch", "--parsable", ARRAY_SH],
                             capture_output=True, text=True).stdout.strip()
        say(f"ALL PASS -> launched 48-config minibatch array as job {jid}")
    else:
        say("NOT launching: at least one gate failed. Settings need review before the array runs.")


if __name__ == "__main__":
    main()
