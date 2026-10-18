# -*- coding: utf-8 -*-
"""阶段 1 出口清单 runner（2026-10-03，接 082 收口后的头名任务）。

五项出口与 boot 分配：
  1 identity canary   E1（082 臂默认 kvmem_e1a）：canary 50K 请求（L<=B=97,888
                      ⇒ 视窗不激活、连接器全机制在场）greedy 输出须与生产基准
                      逐 token 一致。生产基准 =
                      prod029_logs/kvmem_e1_canary/vp_canary_prod.json
                      （48,564 tok，text "\\n\\n77349"，finish=stop）。
  4 多轨迹/串台       同 E1 boot：probe A(e1tra) -> probe B(e1trb) -> serve A
                      重发（stages=serve）。判 A 重发 HIT 且与首遍逐位一致
                      （B/C/D 轨迹入库后 A 的检索与答案不被污染）。工作区
                      204 槽会被四轨迹挤满（072 已预告），dropped 如实记录；
                      丢页属容量紧处，不算串台。
  2 紧预算 needle     E2（S079_PAGES=0 kvmem_e1b）：视窗 = sink 1,456 + recent
                      16,384 = 17,840、检索 0 页。d0.55 必须 MISS
                      （recency-only 必败对照；retrieval 必成一侧已由 082-B
                      8/8 HIT 覆盖，此处补三角的第三边）。
  5 1456 装配回归     E3（load1456 S082_LOAD=1 kvmem_e1c）+ E4（S082_LOAD=0
                      kvmem_e1d）双 boot：assembly probe 200K d0.50，
                      compare --sliding-window 131072 --block-size 1456。
                      页长 1456 上首次实测装配链（含 mamba 页边界快照），
                      也是装配 x 投机组合首次实测（异常则退 S079_SPEC=0 复测
                      并如实记录）。
  3 冷 262K           E5（cold262k kvmem_e1e，L=262,144 无投机）：probe
                      262,000 全三段不崩不 OOM。needle 召回只记录不判
                      （200K-262K = 已知风险带）。

每个 boot 的 dump 目录全新（必守 16-8）；探针判据只认 recent_tokens 指纹；
失败如实记录 INCOMPLETE，不调参凑通过。速度只记录、不判（阶段 1 无性能判据）。

用法（venv python、cwd 不在记录仓）：
  cd G:\\qwen3.8model
  G:\\qwen3.8model\\vllm-win029\\Scripts\\python.exe ^
      G:\\qwen3.8model\\_tmp_line_b\\roundE1_082.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

TOOLS = r"G:\qwen3.8model\vllm-030win-git\tools"
LOGS = r"G:\qwen3.8model\prod029_logs"
SCRATCH = r"G:\qwen3.8model\_tmp_line_b"
PY = r"G:\qwen3.8model\vllm-win029\Scripts\python.exe"
BOOT_KEEP = os.path.join(SCRATCH, "boot_keep.py")
ARM082 = os.path.join(TOOLS, "serve_gsq_kvmem_viewport082_spec.cmd")
ARMLOAD = os.path.join(TOOLS, "serve_gsq_kvmem_load1456.cmd")
ARMCOLD = os.path.join(TOOLS, "serve_gsq_kvmem_cold262k.cmd")
VPROBE = os.path.join(TOOLS, "kvmem_viewport_probe.py")
ASMPROBE = os.path.join(TOOLS, "kvmem_assembly_probe.py")
STATE = os.path.join(LOGS, "stepe1_boots.json")
CANARY_PROD = os.path.join(LOGS, "kvmem_e1_canary", "vp_canary_prod.json")
# Same override as tools/step080_boot.py: Orca legs set KVMEM_RUNNER_BASE to
# http://127.0.0.1:8001 (098 tooling debt; h1b's health() was verified by hand).
BASE = os.environ.get("KVMEM_RUNNER_BASE", "http://127.0.0.1:8080")
TMO_VP = 7200
TMO_BOOT = 420

sys.path.insert(0, TOOLS)
import step080_boot as s80  # noqa: E402  (down/verify/health reuse)


def say(msg: str) -> None:
    print(":: " + msg, flush=True)


def health() -> bool:
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def load_state() -> list:
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as fh:
            return json.load(fh)
    return []


def save_state(rows: list) -> None:
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)


def merge_state(rec: dict) -> None:
    rows = [r for r in load_state() if r.get("boot") != rec["boot"]]
    rows.append(rec)
    save_state(rows)


def boot_arm(arm: str, boot: str, tag: str, extra: dict | None = None,
             timeout: float = TMO_BOOT) -> dict:
    dump_dir = os.path.join(LOGS, tag)
    if os.path.exists(dump_dir):
        say("REFUSING: dump dir %s already exists (iron rule 16-viii)" % dump_dir)
        return {"boot": boot, "tag": tag, "refused": True, "health": False}
    out_log = os.path.join(LOGS, "stepe1_%s_arm.out.log" % boot)
    err_log = os.path.join(LOGS, "stepe1_%s_arm.err.log" % boot)
    env = dict(os.environ)
    env["S079_TAG"] = tag
    env["S074_TAG"] = tag
    for key, val in (extra or {}).items():
        env[key] = str(val)
    rec = {"boot": boot, "tag": tag, "arm": os.path.basename(arm),
           "extra_env": extra or {},
           "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "out_log": out_log, "err_log": err_log}
    say("up %s/%s arm=%s extra=%s" % (boot, tag, rec["arm"], extra))
    with open(out_log, "w", encoding="utf-8", errors="replace") as fo, \
            open(err_log, "w", encoding="utf-8", errors="replace") as fe:
        subprocess.Popen([PY, BOOT_KEEP, arm, out_log, err_log], cwd=SCRATCH,
                         env=env)
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(5)
            if health():
                break
            blob = "\n".join(
                ln for ln in (s80.read_tail(err_log, 200_000)
                              + s80.read_tail(out_log, 400_000)).splitlines()
                if "[rank0]:W" not in ln)
            if ("EngineCore failed to start" in blob
                    or "Traceback (most recent call" in blob):
                rec["fatal"] = True
                break
        rec["boot_s"] = round(time.time() - t0, 1)
        rec["health"] = health()
    rec.update(s80.verify(out_log, err_log))
    say("up %s: boot_s=%s health=%s err=%s tb=%s" % (
        boot, rec["boot_s"], rec["health"], rec.get("error_lines"),
        rec.get("traceback_lines")))
    return rec


def stop_arm(label: str) -> None:
    s80.down(800.0, 240, "e1-" + label)
    say("down (%s), waiting for <800 MiB done" % label)


def run_probe(script: str, args: list, timeout: float) -> dict:
    cmd = [PY, script, "run"] + args
    say("probe: %s" % " ".join(os.path.basename(script) + " " + a
                               if i == 0 else a
                               for i, a in enumerate(args)))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=SCRATCH, timeout=timeout)
    except subprocess.TimeoutExpired:
        say("probe TIMEOUT after %ss" % timeout)
        return {"rc": -1, "tail": "TIMEOUT"}
    tail = (proc.stdout or "")[-1200:] + (proc.stderr or "")[-400:]
    print(tail, flush=True)
    return {"rc": proc.returncode, "tail": tail}


def load_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        say("cannot read %s: %r" % (path, exc))
        return None


def main() -> int:
    rows = load_state()
    rows.append({"boot": "START", "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "note": "stage-1 exit list round E1"})
    save_state(rows)

    # 生产下电（跑臂与生产抢显存，一次只能一个；用户 10-02 令：不必询问）。
    stop_arm("pre")
    summary: dict = {}

    # ---------------- E1: canary + 多轨迹/串台（082 臂默认） ----------------
    rec = boot_arm(ARM082, "e1a", "kvmem_e1a")
    if rec.get("health"):
        d1 = os.path.join(LOGS, "kvmem_e1a")
        # (1) identity canary：臂侧 50K serve（L=48,564 <= B=97,888 ⇒ 不改写）
        canary_out = os.path.join(d1, "vp_e1canary.json")
        run_probe(VPROBE, ["--tokens", "50000", "--depth", "0.50",
                           "--nonce", "e1canary", "--question-style", "focused",
                           "--stages", "serve", "--max-tokens", "32",
                           "--tag", "e1canary_arm", "--out", canary_out],
                  timeout=2400)
        arm_c = load_json(canary_out)
        prod_c = load_json(CANARY_PROD)
        if arm_c and prod_c:
            ta = (arm_c["serve"].get("text") or arm_c["serve"].get("sample") or "")
            tb = (prod_c["serve"].get("text") or prod_c["serve"].get("sample") or "")
            canary = {
                "prod_text": tb, "arm_text": ta,
                "text_equal": ta == tb,
                "finish_equal": (arm_c["serve"].get("finish_reason")
                                 == prod_c["serve"].get("finish_reason")),
                "tokens_prod": prod_c["tokens_ingest"],
                "tokens_arm": arm_c["tokens_ingest"],
                "needle_hit_arm": arm_c["serve"].get("needle_hit"),
                "fingerprint_note": "check kvmem_retrieval json recent_tokens "
                                    "== 32768 (native) in the dump dir",
            }
            summary["canary"] = canary
            say("CANARY: text_equal=%s finish_equal=%s prod=%r arm=%r" % (
                canary["text_equal"], canary["finish_equal"], tb[:40], ta[:40]))
        # (4a) probe A 基线
        a_out = os.path.join(d1, "vp_e1tra.json")
        run_probe(VPROBE, ["--tokens", "200000", "--depth", "0.55",
                           "--nonce", "e1tra", "--question-style", "focused",
                           "--tag", "e1tra", "--out", a_out], timeout=TMO_VP)
        # (4b) probe B（第二轨迹入库）
        b_out = os.path.join(d1, "vp_e1trb.json")
        run_probe(VPROBE, ["--tokens", "200000", "--depth", "0.55",
                           "--nonce", "e1trb", "--question-style", "focused",
                           "--tag", "e1trb", "--out", b_out], timeout=TMO_VP)
        # (4c) serve A 重发（B/C/D 入库后 A 是否被污染）
        a2_out = os.path.join(d1, "vp_e1tra_resend.json")
        run_probe(VPROBE, ["--tokens", "200000", "--depth", "0.55",
                           "--nonce", "e1tra", "--question-style", "focused",
                           "--stages", "serve", "--tag", "e1tra_resend",
                           "--out", a2_out], timeout=TMO_VP)
        ra, rb, ra2 = load_json(a_out), load_json(b_out), load_json(a2_out)
        if ra and ra2:
            ta = (ra["serve"].get("text") or ra["serve"].get("sample") or "")
            ta2 = (ra2["serve"].get("text") or ra2["serve"].get("sample") or "")
            summary["cross_traj"] = {
                "A_first": {"hit": ra["serve"].get("needle_hit"),
                            "text": ta, "finish": ra["serve"].get("finish_reason")},
                "B": {"hit": (rb or {}).get("serve", {}).get("needle_hit"),
                      "text": (rb or {}).get("serve", {}).get("text", ""),
                      "finish": (rb or {}).get("serve", {}).get("finish_reason")},
                "A_resend": {"hit": ra2["serve"].get("needle_hit"),
                             "text": ta2, "finish": ra2["serve"].get("finish_reason")},
                "A_resend_text_equal": ta == ta2,
                "A_resend_hit": ra2["serve"].get("needle_hit"),
            }
            say("CROSS-TRAJ: A first=%r resend=%r equal=%s B=%r" % (
                ta[:40], ta2[:40], ta == ta2,
                summary["cross_traj"]["B"]["text"][:40]))
    rec["summary"] = {k: summary.get(k) for k in ("canary", "cross_traj")
                      if k in summary}
    merge_state(rec)
    stop_arm("e1a")

    # ---------------- E2: 紧预算 recency-only（S079_PAGES=0） ----------------
    rec = boot_arm(ARM082, "e1b", "kvmem_e1b", extra={"S079_PAGES": "0"})
    if rec.get("health"):
        d2 = os.path.join(LOGS, "kvmem_e1b")
        r0_out = os.path.join(d2, "vp_e1rec0.json")
        run_probe(VPROBE, ["--tokens", "200000", "--depth", "0.55",
                           "--nonce", "e1rec0", "--question-style", "focused",
                           "--tag", "e1rec0", "--out", r0_out], timeout=TMO_VP)
        r0 = load_json(r0_out)
        if r0:
            summary["recency_only"] = {
                "hit": r0["serve"].get("needle_hit"),
                "text": r0["serve"].get("text", ""),
                "finish": r0["serve"].get("finish_reason"),
                "expected": "MISS (needle at ~109k, window = sink+recent 17,840)",
            }
            say("RECENCY-ONLY: hit=%s text=%r (expected MISS)" % (
                summary["recency_only"]["hit"],
                summary["recency_only"]["text"][:60]))
    rec["summary"] = {"recency_only": summary.get("recency_only")}
    merge_state(rec)
    stop_arm("e1b")

    # ---------------- E3/E4: 1456 装配回归（LOAD=1 / LOAD=0） ----------------
    asm_outs = {}
    for boot, tag, load in (("e1c", "kvmem_e1c", "1"), ("e1d", "kvmem_e1d", "0")):
        rec = boot_arm(ARMLOAD, boot, tag, extra={"S082_LOAD": load})
        ok = rec.get("health")
        if ok:
            out = os.path.join(LOGS, tag, "asm_%s.json" % boot)
            r = run_probe(ASMPROBE, ["--tokens", "200000", "--depth", "0.50",
                                     "--nonce", "e1asml", "--ingest-max-tokens",
                                     "16", "--max-tokens", "32", "--tag", boot,
                                     "--out", out], timeout=TMO_VP)
            asm_outs[load] = out if r.get("rc") == 0 else None
        rec["summary"] = {"asm_out": asm_outs.get(load)}
        merge_state(rec)
        stop_arm(boot)
    if asm_outs.get("1") and asm_outs.get("0"):
        cmd = [PY, ASMPROBE, "compare", "--a", asm_outs["1"], "--b",
               asm_outs["0"], "--sliding-window", "131072",
               "--block-size", "1456"]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=SCRATCH, timeout=600)
        print((proc.stdout or "")[-1500:], flush=True)
        summary["assembly_compare_rc"] = proc.returncode
        try:
            summary["assembly_compare"] = json.loads(
                (proc.stdout or "").strip().splitlines()[-1]
                if "ASSEMBLY VERDICT" not in (proc.stdout or "")
                else (proc.stdout or "").strip().splitlines()[-2])
        except Exception:
            pass

    # ---------------- E5: 冷 262K transcript（L=262,144 无投机） ----------------
    rec = boot_arm(ARMCOLD, "e1e", "kvmem_e1e")
    if rec.get("health"):
        d5 = os.path.join(LOGS, "kvmem_e1e")
        c_out = os.path.join(d5, "vp_e1c262.json")
        run_probe(VPROBE, ["--tokens", "262000", "--depth", "0.65",
                           "--nonce", "e1c262", "--question-style", "focused",
                           "--tag", "e1c262", "--out", c_out], timeout=TMO_VP)
        rc262 = load_json(c_out)
        if rc262:
            summary["cold262k"] = {
                "ingest_ok": rc262.get("ingest", {}).get("ok"),
                "flush_ok": rc262.get("flush", {}).get("ok"),
                "serve_ok": rc262.get("serve", {}).get("ok"),
                "tokens": rc262.get("tokens_ingest"),
                "needle_hit": rc262.get("serve", {}).get("needle_hit"),
                "finish": rc262.get("serve", {}).get("finish_reason"),
                "record_only": "200K-262K is the known-risk band (record, not gate)",
            }
            say("COLD-262K: ingest=%s flush=%s serve=%s hit=%s" % (
                summary["cold262k"]["ingest_ok"],
                summary["cold262k"]["flush_ok"],
                summary["cold262k"]["serve_ok"],
                summary["cold262k"]["needle_hit"]))
    rec["summary"] = {"cold262k": summary.get("cold262k")}
    merge_state(rec)
    stop_arm("e1e")

    rows = load_state()
    rows.append({"boot": "SUMMARY", "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "summary": summary})
    save_state(rows)
    say("ROUND-E1 DONE")
    say(json.dumps(summary, ensure_ascii=False, indent=2)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
