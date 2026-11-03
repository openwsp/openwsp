"""Modal implementation. The live editor never receives this module or evaluation data."""
import json
import os
import time
from pathlib import Path

import modal

from openwsp.backends.modal_runtime import configure_tls

configure_tls()
EDITOR_TRANSPORT = os.environ.get("OPENWSP_EDITOR_TRANSPORT", "openai")
if EDITOR_TRANSPORT not in {"openai", "codex"}:
    raise ValueError("Unsupported deployed editor transport")
REPO = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/opt/openwsp")
SOURCE = REPO / "src/openwsp" if modal.is_local() else REPO / "openwsp"
app = modal.App("openwsp-exp8")
start_gate_store = modal.Dict.from_name("openwsp-exp8-start-gates", create_if_missing=True)
data_volume = modal.Volume.from_name("openwsp-exp8-data", create_if_missing=True, version=2)
model_volume = modal.Volume.from_name("openwsp-exp8-model", create_if_missing=True, version=2)
practice_volume = modal.Volume.from_name("openwsp-exp8-practice", create_if_missing=True, version=2)
work_volume = modal.Volume.from_name("openwsp-exp8-work", create_if_missing=True, version=2)
kernel_cache_volume = modal.Volume.from_name("openwsp-exp8-kernel-cache", create_if_missing=True, version=2)
common = ["numpy", "safetensors", "transformers==5.17.0", "math-verify==0.9.0", "datasets",
          "absl-py", "langdetect", "nltk", "immutabledict",
          "scipy", "scikit-learn", "pandas", "sympy", "networkx", "openai==3.42.1", "modal==1.6.3"]
cpu_dependencies = (modal.Image.debian_slim(python_version="3.12")
             .uv_pip_install("torch==2.12.0", index_url="https://download.pytorch.org/whl/cpu")
             .uv_pip_install(*common)
             .env({"PYTHONPATH": "/opt/openwsp", "HF_HUB_DISABLE_XET": "1",
                   "OPENWSP_EDITOR_TRANSPORT": EDITOR_TRANSPORT}))
cpu_image = cpu_dependencies.add_local_dir(SOURCE, "/opt/openwsp/openwsp", copy=True)
sandbox_image = cpu_dependencies
for filename in ("__init__.py", "core.py", "edits.py", "kernel.py", "backends/__init__.py", "backends/sandbox_worker.py"):
    sandbox_image = sandbox_image.add_local_file(SOURCE / filename, "/opt/openwsp/openwsp/" + filename, copy=True)
driver_image = (cpu_image.apt_install("git")
                .run_commands("python -m nltk.downloader -d /opt/nltk_data punkt punkt_tab")
                .env({"NLTK_DATA": "/opt/nltk_data"})
                .add_local_dir(REPO / "replications", "/opt/openwsp/replications", copy=True))
gpu_image = (modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
             .apt_install("build-essential", "ninja-build")
             .uv_pip_install("vllm==0.33.0", *common)
             .add_local_dir(SOURCE, "/opt/openwsp/openwsp", copy=True)
             .add_local_dir(REPO / "replications", "/opt/openwsp/replications", copy=True)
             .env({"PYTHONPATH": "/opt/openwsp", "HF_HUB_DISABLE_XET": "1",
                   "OPENWSP_EDITOR_TRANSPORT": EDITOR_TRANSPORT,
                   "VLLM_CACHE_ROOT": "/kernel-cache/vllm", "TRITON_CACHE_DIR": "/kernel-cache/triton",
                   "FLASHINFER_WORKSPACE_BASE": "/kernel-cache/flashinfer"}))


@app.function(image=driver_image, timeout=1800, cpu=(2, 2), memory=(8192, 8192),
              volumes={"/data": data_volume, "/model": model_volume, "/practice": practice_volume},
              secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])])
def prepare():
    from replications.exp8.runtime import atomic_write
    from replications.exp8.runtime import prepare as prepare_data
    start = time.monotonic()
    result = prepare_data()
    result["seconds"] = time.monotonic() - start
    atomic_write("/data/prepare.json", result)
    for volume in (data_volume, model_volume, practice_volume):
        volume.commit()
    return result


@app.cls(image=gpu_image, gpu="H100", cpu=(4, 4), memory=(16384, 16384), timeout=1800, startup_timeout=600,
         scaledown_window=2, max_containers=2, volumes={"/data": data_volume, "/kernel-cache": kernel_cache_volume})
class Generator:
    checkpoint: str = modal.parameter(default="/data/original")
    slot: str = modal.parameter(default="generation")
    job_key: str = modal.parameter(default="")
    gpu_kind: str = modal.parameter(default="H100")

    @modal.enter()
    def load(self):
        self.started = time.monotonic()
        self.load_error = None
        try:
            if self.job_key:
                import uuid

                from openwsp.backends.start_gate import StartGate
                decision = StartGate(start_gate_store).try_start(self.job_key, owner_id=uuid.uuid4().hex)
                if not decision.may_start:
                    self.load_error = "Queued attempt did not win its start gate: " + decision.state
                    return
            self._load()
        except Exception as exc:  # noqa: BLE001 - deliver startup failures once to the caller
            # Returning from enter prevents Modal's startup retry loop from
            # repeatedly charging for a deterministic image/configuration error.
            self.load_error = f"{type(exc).__name__}: {exc}"

    def _load(self):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.engine = LLM(model=self.checkpoint, dtype="bfloat16", language_model_only=True,
                          max_model_len=34816, gpu_memory_utilization=0.85, seed=0,
                          max_num_seqs=128, enable_prefix_caching=False, disable_log_stats=False)
        self.last_accounted = self.started
        kernel_cache_volume.commit()

    def _seconds(self):
        now = time.monotonic()
        seconds = now - self.last_accounted
        self.last_accounted = now
        return seconds

    @modal.method()
    def reference(self, probe_token_ids, edit_id):
        if self.load_error:
            raise RuntimeError(self.load_error)
        from openwsp.backends.vllm_weights import capture_reference
        return {"reference": capture_reference(self.engine, probe_token_ids, edit_id),
                "container_seconds": self._seconds()}

    @modal.method()
    def audit_swap(self, edits, edit_id, probe_token_ids, reference):
        result = self._swap(edits, edit_id, probe_token_ids, reference)
        result["container_seconds"] = self._seconds()
        return result

    def _swap(self, edits, edit_id, probe_token_ids, reference):
        if self.load_error:
            raise RuntimeError(self.load_error)
        from openwsp.backends.vllm_weights import FreshEngineRequired, VLLMWeightSwapper
        from openwsp.core import WeightStore
        try:
            if not hasattr(self, "swapper"):
                self.swapper = VLLMWeightSwapper(self.engine, WeightStore("/data/original"))
            receipt = self.swapper.apply(edits, edit_id=edit_id, probe_token_ids=probe_token_ids, reference=reference)
            result = {"ok": True, "receipt": receipt}
        except FreshEngineRequired as exc:
            result = {"ok": False, "fallback": "fresh_engine_per_pass", "error": str(exc), "evidence": exc.evidence}
        return result

    @modal.method()
    def generate(self, rows, sampling, thinking=True, seed_by_id=False,
                 edits=None, edit_id=None, probe_token_ids=None, reference=None):
        if self.load_error:
            raise RuntimeError(self.load_error)
        import hashlib

        from vllm import SamplingParams

        from replications.exp8.runtime import INSTRUCTION
        start = time.monotonic()
        swap = self._swap(edits, edit_id, probe_token_ids, reference) if edits is not None else None
        if swap and not swap["ok"]:
            return {"needs_fresh_engine": True, "swap_failure": swap, "container_seconds": self._seconds()}
        prompts = [self.tokenizer.apply_chat_template(
            [{"role": "user", "content": row.get("prompt", row.get("problem", "") + "\n" + INSTRUCTION)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking) for row in rows]
        params = SamplingParams(**sampling)
        if seed_by_id:
            params = [SamplingParams(**{**sampling, "seed": int(hashlib.sha256(str(row["id"]).encode()).hexdigest()[:8], 16) % (2**31)})
                      for row in rows]
        generated = self.engine.generate(prompts, params, use_tqdm=False)
        outputs = [{"id": row["id"], "samples": [
            {"text": sample.text, "token_count": len(sample.token_ids), "finish_reason": sample.finish_reason}
            for sample in output.outputs]} for row, output in zip(rows, generated, strict=True)]
        return {"outputs": outputs, "generation_seconds": time.monotonic()-start,
                "container_seconds": self._seconds(),
                "gpu": self.gpu_kind, "engine": "vllm==0.33.0", "transformers": "5.17.0", "swap": swap,
                "engine_settings": {"dtype": "bfloat16", "language_model_only": True, "max_model_len": 34816,
                                    "max_num_seqs": 128, "gpu_memory_utilization": 0.85, "prefix_caching": False}}


@app.cls(image=gpu_image, gpu="H100", cpu=(4, 4), memory=(16384, 16384), timeout=1800, startup_timeout=600,
         scaledown_window=2, max_containers=2,
         volumes={"/data": data_volume, "/work": work_volume, "/kernel-cache": kernel_cache_volume})
class TransformerReader:
    checkpoint: str = modal.parameter(default="/data/original")
    eager: bool = modal.parameter(default=False)
    job_key: str = modal.parameter(default="")
    gpu_kind: str = modal.parameter(default="H100")

    @modal.enter()
    def load(self):
        self.started = time.monotonic()
        self.load_error = None
        try:
            if self.job_key:
                import uuid

                from openwsp.backends.start_gate import StartGate
                decision = StartGate(start_gate_store).try_start(self.job_key, owner_id=uuid.uuid4().hex)
                if not decision.may_start:
                    self.load_error = "Queued attempt did not win its start gate: " + decision.state
                    return
            self._load()
        except Exception as exc:  # noqa: BLE001 - prevent repeated failed container startups
            self.load_error = f"{type(exc).__name__}: {exc}"

    def _load(self):
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.checkpoint, dtype=torch.bfloat16,
            attn_implementation="eager" if self.eager else "sdpa").to("cuda").eval()
        self.last_accounted = self.started

    def _seconds(self):
        now = time.monotonic()
        seconds = now - self.last_accounted
        self.last_accounted = now
        return seconds

    @modal.method()
    def generate(self, rows, sampling, thinking=True):
        """Private second-engine control, with the same rendered user prompts."""
        if self.load_error:
            raise RuntimeError(self.load_error)
        from openwsp.backends.hf_generation import generate
        from replications.exp8.runtime import INSTRUCTION

        start = time.monotonic()
        prompts = [self.tokenizer.apply_chat_template(
            [{"role": "user", "content": row.get("prompt", row.get("problem", "") + "\n" + INSTRUCTION)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking) for row in rows]
        result = generate(self.model, self.tokenizer, prompts, **sampling)
        return {"outputs": [{"id": row["id"], "samples": [sample]}
                            for row, sample in zip(rows, result["outputs"], strict=True)],
                "generation_seconds": time.monotonic() - start, "container_seconds": self._seconds(),
                "gpu": self.gpu_kind, "engine": "transformers==" + result["transformers_version"],
                "transformers": result["transformers_version"],
                "engine_settings": {"dtype": "bfloat16", "attention": "sdpa", "text_inputs_only": True,
                                    "presence_penalty_scope": "generated tokens only",
                                    "eos_token_ids": result["eos_token_ids"],
                                    "sampling": result["sampling"]}}

    @modal.method()
    def read(self, request, directory):
        if self.load_error:
            raise RuntimeError(self.load_error)
        from replications.exp8.internals import capture
        start = time.monotonic()
        result = capture(self.model, self.tokenizer, request, directory)
        result.update(s=time.monotonic()-start, container_seconds=self._seconds())
        (data_volume if Path(directory).is_relative_to("/data") else work_volume).commit()
        return result

    @modal.method()
    def generation_activations(self, rows, records, layers, mlp, directory):
        if self.load_error:
            raise RuntimeError(self.load_error)
        import numpy as np

        from replications.exp8.internals import generation_summary
        from replications.exp8.runtime import INSTRUCTION, atomic_write
        start = time.monotonic()
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        mlp_layers = [layer for layer in layers if layer < 24] if mlp else []
        samples = [(row, record, sample_index, sample) for row, record in zip(rows, records, strict=True)
                   for sample_index, sample in enumerate(record["samples"])]
        arrays = {"hidden_last_prompt.npy": np.zeros((len(samples), len(layers), 1024), dtype=np.float32),
                  "hidden_mean_answer.npy": np.zeros((len(samples), len(layers), 1024), dtype=np.float32)}
        if mlp:
            arrays.update({"mlp_last_prompt.npy": np.zeros((len(samples), len(mlp_layers), 3584), dtype=np.float32),
                           "mlp_mean_answer.npy": np.zeros((len(samples), len(mlp_layers), 3584), dtype=np.float32)})
        index = []
        for i, (row, record, sample_index, sample) in enumerate(samples):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": row.get("prompt", row.get("problem", "") + "\n" + INSTRUCTION)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
            pids = self.tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = self.tokenizer.encode(sample["text"], add_special_tokens=False)[:8192]
            # Concatenation follows the tool's raw-text teacher-forcing convention.
            text = prompt + self.tokenizer.decode(answer_ids, skip_special_tokens=False)
            summary = generation_summary(self.model, self.tokenizer, text, len(pids), layers, mlp_layers)
            for name, value in summary.items():
                arrays[name][i] = value
            index.append({"item": row["id"], "sample": sample_index, "correct": sample.get("correct"),
                          "tokens": len(answer_ids)})
        for name, array in arrays.items():
            np.save(output / name, array)
        meta = {"layers": layers, "hidden_layers": layers, "mlp_layers": mlp_layers, "samples": index,
                "files": {name: list(array.shape) for name, array in arrays.items()}}
        atomic_write(output / "index.json", meta)
        work_volume.commit()
        return {"dir": directory, "files": meta["files"], "s": time.monotonic()-start,
                "container_seconds": self._seconds()}

    @modal.method()
    def mean_mlp_outputs(self, examples, layers=(15, 19)):
        """Private control measurement; never exposed as an editor tool."""
        if self.load_error:
            raise RuntimeError(self.load_error)
        import torch

        start = time.monotonic()
        sums = {group: {layer: torch.zeros(1024, dtype=torch.float64) for layer in layers}
                for group in ("looping", "finished")}
        counts = {"looping": 0, "finished": 0}
        hooks, current = [], {}
        for layer in layers:
            def hook(module, arguments, result, layer=layer):
                values = result[0, current["prompt_length"]:]
                if not len(values):
                    raise ValueError("Control example has no answer tokens")
                sums[current["group"]][layer] += values.float().mean(0).double().cpu()
            hooks.append(self.model.model.language_model.layers[layer].mlp.down_proj.register_forward_hook(hook))
        try:
            with torch.inference_mode():
                for example in examples:
                    prompt = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": example["prompt"]}], tokenize=False,
                        add_generation_prompt=True, enable_thinking=True)
                    prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                    answer_ids = self.tokenizer.encode(example["text"], add_special_tokens=False)[:8192]
                    current.update(group=example["group"], prompt_length=len(prompt_ids))
                    tokens = torch.tensor([prompt_ids + answer_ids], device="cuda")
                    self.model.model.language_model(input_ids=tokens, use_cache=False)
                    counts[example["group"]] += 1
        finally:
            for hook in hooks:
                hook.remove()
        return {"sums": {group: {str(layer): value.tolist() for layer, value in values.items()}
                         for group, values in sums.items()}, "counts": counts,
                "measurement_seconds": time.monotonic()-start, "container_seconds": self._seconds()}


@app.function(image=driver_image, timeout=86400, startup_timeout=120, scaledown_window=5, retries=0,
              cpu=(1, 1), memory=(4096, 4096),
              volumes={"/data": data_volume, "/model": model_volume, "/practice": practice_volume,
                       "/work": work_volume},
              secrets=([modal.Secret.from_name("openai", required_keys=["OPENAI_API_KEY"])]
                       if EDITOR_TRANSPORT == "openai" else [])
                      + [modal.Secret.from_name("github", required_keys=["GITHUB_TOKEN"])],
              max_containers=1)
def pipeline(stage="all", config=None, driver_ticket=None):
    from replications.exp8.pipeline import run
    if (config or {}).get("editor", "openai") != EDITOR_TRANSPORT:
        raise ValueError("Requested editor differs from the deployed authentication configuration")
    return run(stage, config or {}, app=app, driver_ticket=driver_ticket)


@app.function(image=driver_image, timeout=120, startup_timeout=60, scaledown_window=2,
              cpu=(0.25, 0.25), memory=(1024, 1024), max_containers=1,
              volumes={"/data": data_volume})
@modal.concurrent(max_inputs=1)
def launch(stage="all", config=None, recover=False):
    from openwsp.backends.coordinator import DriverCoordinator
    from openwsp.budget import BudgetLedger
    def ledger_owner():
        ledger = BudgetLedger("/data/budget.json")
        if "infrastructure-headroom" not in ledger.snapshot()["reservations"]:
            ledger.reserve("modal", 2, "infrastructure", reservation_id="infrastructure-headroom",
                           metadata={"basis": "conservative headroom for builds, status, coordinator, watchdog, volume and network; reconcile against provider billing"})
            data_volume.commit()
        return ledger
    coordinator = DriverCoordinator(ledger_owner, "/data/coordinator.json",
                                    state_path="/data/state.json", commit=data_volume.commit,
                                    reload=data_volume.reload)
    settings = dict(config or {})
    resume_reason = settings.pop("resume_reason", None)
    spawn = lambda next_stage, settings, ticket: pipeline.spawn(next_stage, settings, ticket)
    return coordinator.recover(spawn) if recover else coordinator.launch(stage, settings, spawn, resume_reason=resume_reason)


@app.function(image=cpu_image, timeout=150, cpu=(0.25, 0.25), memory=(512, 512),
              schedule=modal.Period(minutes=15))
def watchdog():
    return launch.remote(recover=True)


@app.function(image=driver_image, timeout=3600, startup_timeout=120, scaledown_window=5,
              cpu=(1, 1), memory=(4096, 4096), retries=0, max_containers=1,
              volumes={"/data": data_volume})
@modal.concurrent(max_inputs=1)
def public_forward(payload):
    from openwsp.backends.public_forward import execute
    return execute(payload)


@app.function(image=driver_image, timeout=60, volumes={"/data": data_volume})
def status():
    path = Path("/data/state.json")
    return json.loads(path.read_text()) if path.exists() else {"status": "not_started"}


@app.function(image=driver_image, timeout=120, cpu=(0.25, 0.25), memory=(1024, 1024),
              scaledown_window=2, max_containers=1, volumes={"/data": data_volume})
@modal.concurrent(max_inputs=1)
def editor_exchange(operation, args, kwargs):
    from openwsp.backends.editor_bridge import exchange
    return exchange(operation, args, kwargs, directory="/data/editor-bridge",
                    commit=data_volume.commit, reload=data_volume.reload)


@app.function(image=sandbox_image, timeout=60)
def sandbox_image_ready():
    return True
