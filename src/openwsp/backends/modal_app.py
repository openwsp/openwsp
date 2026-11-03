"""Modal implementation. The live editor never receives this module or evaluation data."""
import json
import time
from pathlib import Path

import modal

from openwsp.backends.modal_runtime import configure_tls

configure_tls()
REPO = Path(__file__).resolve().parents[3]
app = modal.App("openwsp-exp8")
data_volume = modal.Volume.from_name("openwsp-exp8-data", create_if_missing=True, version=2)
model_volume = modal.Volume.from_name("openwsp-exp8-model", create_if_missing=True, version=2)
practice_volume = modal.Volume.from_name("openwsp-exp8-practice", create_if_missing=True, version=2)
work_volume = modal.Volume.from_name("openwsp-exp8-work", create_if_missing=True, version=2)
common = ["numpy", "safetensors", "transformers==5.17.0", "math-verify==0.9.0", "datasets",
          "scipy", "scikit-learn", "pandas", "sympy", "networkx", "openai==3.42.1", "modal==1.6.3"]
cpu_image = (modal.Image.debian_slim(python_version="3.12")
             .uv_pip_install("torch==2.12.0", index_url="https://download.pytorch.org/whl/cpu")
             .uv_pip_install(*common)
             .add_local_dir(REPO / "src/openwsp", "/opt/openwsp/openwsp", copy=True)
             .env({"PYTHONPATH": "/opt/openwsp", "HF_HUB_DISABLE_XET": "1"}))
driver_image = cpu_image.add_local_dir(REPO / "replications", "/opt/openwsp/replications", copy=True)
gpu_image = (modal.Image.debian_slim(python_version="3.12")
             .uv_pip_install("vllm==0.33.0", *common)
             .add_local_dir(REPO / "src/openwsp", "/opt/openwsp/openwsp", copy=True)
             .add_local_dir(REPO / "replications", "/opt/openwsp/replications", copy=True)
             .env({"PYTHONPATH": "/opt/openwsp", "HF_HUB_DISABLE_XET": "1"}))


@app.function(image=driver_image, timeout=1800, cpu=2, memory=8192,
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


@app.cls(image=gpu_image, gpu="H100", cpu=4, memory=16384, timeout=1800,
         scaledown_window=2, max_containers=2, volumes={"/data": data_volume})
class Generator:
    checkpoint: str = modal.parameter(default="/data/original")

    @modal.enter()
    def load(self):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.started = time.monotonic()
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.engine = LLM(model=self.checkpoint, dtype="bfloat16", language_model_only=True,
                          max_model_len=34816, gpu_memory_utilization=0.85, seed=0,
                          enable_prefix_caching=False)

    @modal.method()
    def generate(self, rows, sampling, thinking=True):
        from vllm import SamplingParams

        from replications.exp8.runtime import INSTRUCTION
        start = time.monotonic()
        prompts = [self.tokenizer.apply_chat_template(
            [{"role": "user", "content": row.get("prompt", row.get("problem", "") + "\n" + INSTRUCTION)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking) for row in rows]
        generated = self.engine.generate(prompts, SamplingParams(**sampling), use_tqdm=False)
        outputs = [{"id": row["id"], "samples": [
            {"text": sample.text, "token_count": len(sample.token_ids), "finish_reason": sample.finish_reason}
            for sample in output.outputs]} for row, output in zip(rows, generated, strict=True)]
        return {"outputs": outputs, "generation_seconds": time.monotonic()-start,
                "container_seconds": time.monotonic()-self.started,
                "gpu": "H100", "engine": "vllm==0.33.0", "transformers": "5.17.0"}


@app.function(image=driver_image, timeout=86400, cpu=1, memory=4096,
              volumes={"/data": data_volume, "/model": model_volume, "/practice": practice_volume,
                       "/work": work_volume},
              secrets=[modal.Secret.from_name("openai", required_keys=["OPENAI_API_KEY"]),
                       modal.Secret.from_name("github", required_keys=["GITHUB_TOKEN"])],
              max_containers=1)
def pipeline(stage="all", config=None):
    from replications.exp8.pipeline import run
    return run(stage, config or {}, app=app)


@app.function(image=driver_image, timeout=60, volumes={"/data": data_volume})
def status():
    path = Path("/data/state.json")
    return json.loads(path.read_text()) if path.exists() else {"status": "not_started"}
