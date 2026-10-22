#!/usr/bin/env bash
# Run the eval (baseline + agent) on a rented vast.ai GPU instance.
#
# Sets up the same stack as the owner's laptop, natively (instances are
# containers without Docker): Elasticsearch 8.15.0, Ollama 0.34.0 with the
# host-RAM prompt cache disabled, the pinned Python deps, the indexes built on
# the GPU. Then runs both modes on the SAME machine so they are comparable.
#
# Assumptions: run as root, from a clone of this repo. Python is the active
# venv's `python`, else /venv/main/bin/python (vast.ai images activate that
# venv only in interactive shells, so `ssh host cmd` and nohup don't see it).
#
# Every step is idempotent: re-running after a dropped SSH session skips what
# is done, including evals whose results/<tag>_*.json already exists.
#
# Usage (from the repo root):
#   bash scripts/vast_eval.sh
# If the SSH connection might drop, detach it from the session instead:
#   nohup bash scripts/vast_eval.sh > vast_eval.log 2>&1 &
#   tail -f vast_eval.log
#
# Settings (environment variables):
#   MODELS        Ollama models to evaluate, in order (default qwen2.5:7b-instruct)
#   MODES         modes per model, in order          (default "baseline agent")
#   RUN_ID        tag part; results are <mode>_<RUN_ID>_<model>[_ctx...] (default v4_vast)
#   EVAL_SETS     --sets for the eval runner (default: the runner's default)
#   NUM_CTX       override llm.num_ctx (Ollama context window, tokens)
#   CONTEXT_CHARS override agent.context_chars (per-document chars in prompts)
#   MIN_FREE_GB   free disk required before starting           (default 60)
#   ALLOW_MODEL_MISMATCH=1  continue even if the 7B's id differs from the laptop's
#   ALLOW_CPU_OFFLOAD=1     continue even if a model does not fit fully in VRAM

set -euo pipefail

ES_VERSION="8.15.0"
OLLAMA_VERSION="0.34.0"
# The laptop's model: runs of it must use the same weights to be comparable.
LAPTOP_MODEL="qwen2.5:7b-instruct"
LAPTOP_MODEL_ID="845dbda0ea48"
MODELS="${MODELS:-$LAPTOP_MODEL}"
RUN_ID="${RUN_ID:-v4_vast}"
MODES="${MODES:-baseline agent}"
EVAL_SETS="${EVAL_SETS:-}"
NUM_CTX="${NUM_CTX:-}"
CONTEXT_CHARS="${CONTEXT_CHARS:-}"
MIN_FREE_GB="${MIN_FREE_GB:-60}"
ES_HOME="/opt/elasticsearch-${ES_VERSION}"
LOG_DIR="${HOME}/vast_logs"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"

log() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

wait_for() {  # wait_for <description> <timeout_s> <command...>
    local what="$1" timeout="$2"; shift 2
    local waited=0
    until "$@" >/dev/null 2>&1; do
        (( waited >= timeout )) && die "$what did not come up within ${timeout}s"
        sleep 3; waited=$(( waited + 3 ))
    done
}

es_healthy() {
    curl -fs 'http://127.0.0.1:9200/_cluster/health?wait_for_status=yellow&timeout=2s'
}

ollama_up() { curl -fs http://127.0.0.1:11434/api/version; }

# --------------------------------------------------------------- preflight
log "Preflight checks"
[[ $EUID -eq 0 ]] || die "run as root (vast.ai instances log you in as root)"
[[ -f requirements.txt && -d agent ]] || die "not a clone of agentic-rag: $REPO_ROOT"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: is this a GPU instance?"

# Older drivers print "CUDA Version: 13.0", newer ones "CUDA UMD Version: 13.4".
cuda_version="$(nvidia-smi | sed -nE 's/.*CUDA (UMD )?Version: ([0-9]+\.[0-9]+).*/\2/p' | head -1)"
[[ -n "$cuda_version" ]] || die "could not read the driver's CUDA version from nvidia-smi"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) | driver CUDA $cuda_version"

# torch only runs the query/document encoders (Ollama ships its own CUDA
# runtime), so the torch build follows the driver, as the Dockerfile's
# TORCH_VARIANT does: the cu130 build pinned in requirements.txt on a CUDA 13
# driver, the same torch version built for cu126 on a CUDA 12 driver.
TORCH_VERSION="$(sed -nE 's/^torch==([0-9.]+)\+cu[0-9]+.*/\1/p' requirements.txt)"
[[ -n "$TORCH_VERSION" ]] || die "could not read the torch pin from requirements.txt"
case "${cuda_version%%.*}" in
    1[3-9]) TORCH_VARIANT=cu130 ;;
    12)     TORCH_VARIANT=cu126 ;;
    *)      die "driver supports CUDA $cuda_version; this needs >= 12 (filter on 'Max CUDA' when renting)" ;;
esac
echo "torch: $TORCH_VERSION+$TORCH_VARIANT"

# CUDA 13 dropped GPUs below compute capability 7.5 (Pascal, e.g. Tesla P40).
compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
if [[ "$TORCH_VARIANT" == cu130 ]]; then
    awk -v c="$compute_cap" 'BEGIN { exit !(c >= 7.5) }' \
        || die "GPU compute capability $compute_cap is below 7.5, which CUDA 13 torch cannot use (rent an RTX 20-series or newer)"
fi

free_gb="$(df -BG --output=avail "$REPO_ROOT" | tail -1 | tr -dc '0-9')"
(( free_gb >= MIN_FREE_GB )) \
    || die "only ${free_gb} GB free disk; need ${MIN_FREE_GB} GB (torch, models, indexes). The disk size is fixed at rent time: rent with more"
echo "Disk: ${free_gb} GB free"

if command -v python >/dev/null; then
    PY="$(command -v python)"
elif [[ -x /venv/main/bin/python ]]; then
    PY=/venv/main/bin/python
else
    die "no python found: activate a venv or use a vast.ai image with /venv/main"
fi
py_version="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
    || die "python $py_version is too old; this project needs >= 3.11 (asyncio.timeout)"
[[ "$py_version" == "3.12" ]] \
    || echo "WARNING: python $py_version; the pinned dependencies were resolved and tested on 3.12"
"$PY" -c 'import sys; sys.exit(sys.prefix == sys.base_prefix)' \
    || echo "WARNING: $PY is not a venv; installing into it anyway"
echo "Python: $PY ($py_version)"

# ----------------------------------------------------------- system packages
missing=()
for tool in curl zstd; do command -v "$tool" >/dev/null || missing+=("$tool"); done
if (( ${#missing[@]} )); then
    log "Installing system packages: ${missing[*]}"
    apt-get update -qq && apt-get install -y -qq "${missing[@]}"
fi

# --------------------------------------------------------- Python dependencies
log "Installing Python dependencies (torch $TORCH_VERSION+$TORCH_VARIANT, then core + ml)"
"$PY" -m pip install -q --upgrade pip
# torch first, from its CUDA index, so nothing below pulls a PyPI build.
"$PY" -m pip install -q "torch==${TORCH_VERSION}+${TORCH_VARIANT}" \
    --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}"
"$PY" -m pip install -q -r requirements-core.txt -r requirements-ml.txt
# Images with torch preinstalled (e.g. vast.ai PyTorch) also ship torchvision /
# torchaudio built for THEIR torch. This project uses neither, but transformers
# imports torchvision whenever it is installed, and a mismatched one crashes
# that import ("operator torchvision::nms does not exist").
"$PY" -m pip uninstall -y -q torchvision torchaudio 2>/dev/null || true
"$PY" - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU"
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
EOF
log "Unit tests"
"$PY" -m unittest discover -s tests -t . 2>&1 | tail -3

# ----------------------------------------------------------- Elasticsearch
if es_healthy >/dev/null 2>&1; then
    log "Elasticsearch already running"
else
    if [[ ! -x "$ES_HOME/bin/elasticsearch" ]]; then
        log "Installing Elasticsearch $ES_VERSION"
        curl -fsSL "https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-${ES_VERSION}-linux-x86_64.tar.gz" \
            | tar -xz -C /opt
        # Same settings as the laptop container. allow_mmap=false replaces
        # vm.max_map_count, which a container cannot raise.
        cat >> "$ES_HOME/config/elasticsearch.yml" <<'EOF'

# --- agentic-rag (scripts/vast_eval.sh) ---
discovery.type: single-node
network.host: 127.0.0.1
xpack.security.enabled: false
node.store.allow_mmap: false
EOF
    fi
    id es >/dev/null 2>&1 || useradd -m es   # Elasticsearch refuses to run as root
    chown -R es:es "$ES_HOME"
    log "Starting Elasticsearch"
    su es -c "ES_JAVA_OPTS='-Xms1g -Xmx1g' $ES_HOME/bin/elasticsearch -d -p /tmp/es.pid"
    wait_for "Elasticsearch" 180 es_healthy
fi
echo "Elasticsearch $(curl -fs http://127.0.0.1:9200 | grep -oP '"number" : "\K[^"]+')"

# ------------------------------------------------------------------ Ollama
installed_ollama="$(ollama --version 2>/dev/null | grep -oP '[0-9]+\.[0-9]+\.[0-9]+' | tail -1 || true)"
if [[ "$installed_ollama" != "$OLLAMA_VERSION" ]]; then
    log "Installing Ollama $OLLAMA_VERSION (a 'systemd not found' warning is expected)"
    curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="$OLLAMA_VERSION" sh
fi

if ollama_up >/dev/null 2>&1; then
    # A server we did not start may have the prompt cache on, which OOM-kills
    # the model mid-eval. Only accept it if its environment disables the cache.
    pid="$(pgrep -xo ollama || true)"
    if [[ -z "$pid" ]] || ! tr '\0' '\n' < "/proc/$pid/environ" | grep -qx 'LLAMA_ARG_CACHE_RAM=0'; then
        die "an Ollama server is already running without LLAMA_ARG_CACHE_RAM=0; stop it (pkill -x ollama) and re-run"
    fi
    log "Ollama already running (prompt cache disabled)"
else
    log "Starting Ollama (localhost only, prompt cache disabled)"
    LLAMA_ARG_CACHE_RAM=0 OLLAMA_HOST=127.0.0.1:11434 \
        nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
    wait_for "Ollama" 60 ollama_up
fi

for model in $MODELS; do
    log "Pulling $model"
    ollama pull "$model"
    model_id="$(ollama list | awk -v m="$model" '$1 == m {print $2}')"
    echo "Model $model id: $model_id"
    if [[ "$model" == "$LAPTOP_MODEL" && "$model_id" != "$LAPTOP_MODEL_ID" ]]; then
        msg="$model id $model_id differs from the laptop's $LAPTOP_MODEL_ID: the tag now points at different weights, so results are not comparable with laptop runs"
        [[ "${ALLOW_MODEL_MISMATCH:-0}" == "1" ]] && echo "WARNING: $msg" || die "$msg (set ALLOW_MODEL_MISMATCH=1 to continue anyway)"
    fi
done

# ----------------------------------------------------------------- indexes
log "Building indexes (idempotent; downloads SciFact if needed)"
"$PY" -m scripts.build_indexes

# ------------------------------------------------------------------- evals
# Config overrides for every eval run (configs/loader.py env overrides).
ctx_suffix=""
if [[ -n "$NUM_CTX" ]]; then
    export AGENTIC_RAG__LLM__NUM_CTX="$NUM_CTX"; ctx_suffix+="_ctx${NUM_CTX}"
fi
if [[ -n "$CONTEXT_CHARS" ]]; then
    export AGENTIC_RAG__AGENT__CONTEXT_CHARS="$CONTEXT_CHARS"; ctx_suffix+="_chars${CONTEXT_CHARS}"
fi

for model in $MODELS; do
    export LLM_MODEL="$model"
    slug="$(tr ':/' '--' <<< "$model")"

    # Load the model the way the eval will (same num_ctx) and check it fits
    # in VRAM: a partly CPU-offloaded model runs several times slower and
    # makes latency numbers meaningless.
    num_ctx="$("$PY" -c 'from configs.loader import load_config; print(load_config()["llm"]["num_ctx"])')"
    log "Loading $model (num_ctx=$num_ctx) to check it fits in VRAM"
    curl -fs http://127.0.0.1:11434/api/generate \
        -d "{\"model\": \"$model\", \"prompt\": \"\", \"options\": {\"num_ctx\": $num_ctx}}" >/dev/null
    placement="$(ollama ps | awk -v m="$model" '$1 == m {for (i = 1; i <= NF; i++) if ($i ~ /%/) {print $i, $(i+1); exit}}')"
    echo "$model placement: ${placement:-unknown}"
    if [[ "$placement" != "100% GPU" ]]; then
        msg="$model is not fully on the GPU ($placement) at num_ctx=$num_ctx"
        [[ "${ALLOW_CPU_OFFLOAD:-0}" == "1" ]] && echo "WARNING: $msg" || die "$msg; use a smaller model, a smaller NUM_CTX or a bigger GPU (or ALLOW_CPU_OFFLOAD=1)"
    fi

    for mode in $MODES; do
        tag="${mode}_${RUN_ID}_${slug}${ctx_suffix}"
        if compgen -G "results/${tag}_*.json" >/dev/null; then
            log "Skipping $tag: $(ls results/"${tag}"_*.json | tail -1) already exists"
            continue
        fi
        log "Evaluating model=$model mode=$mode tag=$tag (log: $LOG_DIR/$tag.log)"
        args=(--mode "$mode" --tag "$tag")
        [[ -n "$EVAL_SETS" ]] && args+=(--sets "$EVAL_SETS")
        # JSON logs go to the log file only; the summary table goes to both.
        "$PY" -m evaluation.evaluate_rag "${args[@]}" 2>>"$LOG_DIR/$tag.log" | tee -a "$LOG_DIR/$tag.log"
    done
    # Free the VRAM before the next model loads next to the encoders.
    ollama stop "$model" || true
done

# ------------------------------------------------------------------- done
log "Done. Results:"
ls -1 results/*_"${RUN_ID}"_*.json
cat <<EOF

Copy them to the laptop (host and port are on the instance's Connect button):
  scp -P <PORT> 'root@<HOST>:${REPO_ROOT}/results/*_${RUN_ID}_*.json' ~/agentic-rag/results/
Then DESTROY the instance in the vast.ai console (a stopped instance still bills for disk).
EOF
