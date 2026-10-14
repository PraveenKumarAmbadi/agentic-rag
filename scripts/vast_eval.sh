#!/usr/bin/env bash
# Run the eval (baseline + agent) on a rented vast.ai GPU instance.
#
# Sets up the same stack as the owner's laptop, natively (instances are
# containers without Docker): Elasticsearch 8.15.0, Ollama 0.34.0 with the
# host-RAM prompt cache disabled, the pinned Python deps, the indexes built on
# the GPU. Then runs both modes on the SAME machine so they are comparable.
#
# Assumptions: run as root, from a clone of this repo, with the template's
# Python venv already active (the script installs into whatever `python` is).
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
#   RUN_ID        tag suffix; results are <mode>_<RUN_ID>   (default v4_vast)
#   MODES         modes to evaluate, in order        (default "baseline agent")
#   EVAL_SETS     --sets for the eval runner (default: the runner's default)
#   EXPECTED_MODEL_ID  Ollama model id the laptop runs  (default 845dbda0ea48)
#   ALLOW_MODEL_MISMATCH=1  continue even if the pulled model id differs

set -euo pipefail

ES_VERSION="8.15.0"
OLLAMA_VERSION="0.34.0"
MODEL="qwen2.5:7b-instruct"
EXPECTED_MODEL_ID="${EXPECTED_MODEL_ID:-845dbda0ea48}"
RUN_ID="${RUN_ID:-v4_vast}"
MODES="${MODES:-baseline agent}"
EVAL_SETS="${EVAL_SETS:-}"
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
if [[ -z "$cuda_version" ]]; then
    echo "WARNING: could not read the driver's CUDA version; relying on the torch GPU check below"
else
    # requirements.txt pins torch==...+cu130, which needs a CUDA 13 driver.
    (( ${cuda_version%%.*} >= 13 )) \
        || die "driver supports CUDA $cuda_version; this needs >= 13.0 (filter on 'Max CUDA' when renting)"
fi
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) | driver CUDA ${cuda_version:-unknown}"

command -v python >/dev/null || die "no 'python' on PATH: activate the template's venv"
py_version="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
python -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
    || die "python $py_version is too old; this project needs >= 3.11 (asyncio.timeout)"
[[ "$py_version" == "3.12" ]] \
    || echo "WARNING: python $py_version; the pinned dependencies were resolved and tested on 3.12"
python -c 'import sys; sys.exit(sys.prefix == sys.base_prefix)' \
    || echo "WARNING: python is not a venv ($(command -v python)); installing into it anyway"
echo "Python: $(command -v python) ($py_version)"

# ----------------------------------------------------------- system packages
missing=()
for tool in curl zstd; do command -v "$tool" >/dev/null || missing+=("$tool"); done
if (( ${#missing[@]} )); then
    log "Installing system packages: ${missing[*]}"
    apt-get update -qq && apt-get install -y -qq "${missing[@]}"
fi

# --------------------------------------------------------- Python dependencies
log "Installing Python dependencies (requirements.txt)"
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU"
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
EOF
log "Unit tests"
python -m unittest discover -s tests -t . 2>&1 | tail -3

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

log "Pulling $MODEL"
ollama pull "$MODEL"
model_id="$(ollama list | awk -v m="$MODEL" '$1 == m {print $2}')"
echo "Model $MODEL id: $model_id (laptop: $EXPECTED_MODEL_ID)"
if [[ "$model_id" != "$EXPECTED_MODEL_ID" ]]; then
    msg="model id $model_id differs from the laptop's $EXPECTED_MODEL_ID: the tag now points at different weights, so results are not comparable with laptop runs"
    [[ "${ALLOW_MODEL_MISMATCH:-0}" == "1" ]] && echo "WARNING: $msg" || die "$msg (set ALLOW_MODEL_MISMATCH=1 to continue anyway)"
fi

# ----------------------------------------------------------------- indexes
log "Building indexes (idempotent; downloads SciFact if needed)"
python -m scripts.build_indexes

# ------------------------------------------------------------------- evals
for mode in $MODES; do
    tag="${mode}_${RUN_ID}"
    if compgen -G "results/${tag}_*.json" >/dev/null; then
        log "Skipping $mode: $(ls results/${tag}_*.json | tail -1) already exists"
        continue
    fi
    log "Evaluating mode=$mode tag=$tag (log: $LOG_DIR/$tag.log)"
    args=(--mode "$mode" --tag "$tag")
    [[ -n "$EVAL_SETS" ]] && args+=(--sets "$EVAL_SETS")
    # JSON logs go to the log file only; the summary table goes to both.
    python -m evaluation.evaluate_rag "${args[@]}" 2>>"$LOG_DIR/$tag.log" | tee -a "$LOG_DIR/$tag.log"
done

# ------------------------------------------------------------------- done
log "Done. Results:"
ls -1 results/*_"${RUN_ID}"_*.json
cat <<EOF

Copy them to the laptop (host and port are on the instance's Connect button):
  scp -P <PORT> 'root@<HOST>:${REPO_ROOT}/results/*_${RUN_ID}_*.json' ~/agentic-rag/results/
Then DESTROY the instance in the vast.ai console (a stopped instance still bills for disk).
EOF
