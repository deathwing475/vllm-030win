#!/bin/bash
# run_anchor_arm.sh <launcher.cmd> <out.json> <arm> <lengths...>
LAUNCHER="$1"; OUT="$2"; ARM="$3"; shift 3
export TMP="G:/qwen3.8model/_tmp_anchors" TEMP="G:/qwen3.8model/_tmp_anchors"
PWS="powershell -NoProfile -ExecutionPolicy Bypass -File G:\qwen3.8model\vllm-030win-git\tools\kill_vllm_orphans.ps1"
echo "--- pre-clean"
$PWS
cmd //c "$LAUNCHER" > "G:/qwen3.8model/_tmp_anchors/${ARM}_serve.log" 2>&1 &
SERVE_WRAPPER=$!
echo "serve starting: $LAUNCHER"
HEALTHY=0
for i in $(seq 1 120); do
  if python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/v1/models',timeout=5)" 2>/dev/null; then
    echo "healthy after ${i}x5s"; HEALTHY=1; break
  fi
  if ! kill -0 $SERVE_WRAPPER 2>/dev/null; then echo "serve exited early"; break; fi
  sleep 5
done
RC=1
if [ "$HEALTHY" = "1" ]; then
  python "G:/qwen3.8model/vllm-030win-git/tools/anchor_longctx.py" --lengths "$@" --out "$OUT" --arm "$ARM"
  RC=$?
fi
echo "--- post-clean"
$PWS
wait $SERVE_WRAPPER 2>/dev/null
echo "ARM_DONE rc=$RC"
