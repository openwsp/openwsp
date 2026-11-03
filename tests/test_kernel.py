import pytest
import torch

from openwsp.core import WeightStore
from openwsp.kernel import Kernel


@pytest.fixture
def kernel(tmp_path):
    store = WeightStore({"layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}, tmp_path / "edited")
    return Kernel(store, tmp_path / "work", tmp_path / "practice")


def test_persistent_namespace_last_expression_and_gradients(kernel):
    assert kernel.run("a = 7\na + 1")["output"] == "8\n"
    result = kernel.run("assert not torch.is_grad_enabled()\na *= 3\na")
    assert result["ok"] and result["output"] == "21\n"
    assert result["runs"] == 2


def test_save_and_inspect(kernel):
    result = kernel.run("w = load(NAMES[0])\nw[0, 0] += 2\nsave_edit(NAMES[0], w)\nedited()")
    assert result["ok"]
    assert result["output"] == "['layer.weight']\n"
    assert kernel.store.load("layer.weight")[0, 0] == 3
    assert kernel.store.load_original("layer.weight")[0, 0] == 1


@pytest.mark.parametrize(
    "code",
    [
        "torch.set_grad_enabled(True)",
        "from torch import set_grad_enabled as grad; grad(True)",
        "from torch import set_grad_enabled; set_grad_enabled(True)",
        "from torch.nn.functional import linear; linear(torch.ones(2), torch.ones(2, 2))",
        "from torch.nn.functional import linear as dense; dense(torch.ones(2), torch.ones(2, 2))",
        "from torch import *",
        "from torch.nn.functional import *",
        "torch.autograd.grad([], [])",
        "w = torch.ones(2, requires_grad=True)",
        "load(NAMES[0]).backward()",
        "from torch.optim import Adam",
        "from torch import optim",
        "getattr(torch, 'autograd')",
        "import transformers",
        "torch.nn.Linear(2, 2)",
        "import socket",
        "import os\nos.system('true')",
        "exec('1 + 2')",
    ],
)
def test_protocol_refusals(kernel, code):
    result = kernel.run(code)
    assert not result["ok"]
    assert "ProtocolRefused" in result["error"]
    assert result["runs"] == 0


def test_error_does_not_lose_namespace(kernel):
    result = kernel.run("a = 12\nraise ValueError('failure')")
    assert not result["ok"] and "failure" in result["error"]
    assert kernel.run("a")["output"] == "12\n"


def test_weight_analysis_normalization_is_allowed(kernel):
    result = kernel.run(
        "from torch.nn import functional as F\nv = F.normalize(load(NAMES[0]), dim=1)\nv.shape"
    )
    assert result["ok"]
    assert result["output"] == "torch.Size([2, 2])\n"


def test_imported_callable_aliases_are_checked_across_cells(kernel):
    assert kernel.run("from torch import set_grad_enabled as grad")["ok"]
    refused = kernel.run("grad(True)")
    assert not refused["ok"] and "ProtocolRefused" in refused["error"]
    assert refused["runs"] == 1
    disabled = kernel.run("grad(False)\ntorch.is_grad_enabled()")
    assert disabled["ok"] and disabled["output"] == "False\n"
    assert kernel.run("from torch.nn.functional import linear as dense")["ok"]
    refused = kernel.run("dense(load(NAMES[0]), load(NAMES[0]))")
    assert not refused["ok"] and "ProtocolRefused" in refused["error"]
    assert refused["runs"] == 3


def test_output_bound_and_ascii(kernel):
    kernel.output_limit = 100
    result = kernel.run("print('é' * 10_000)")
    assert result["ok"]
    assert len(result["output"]) == 100
    assert result["output"].isascii()
    assert result["output"].endswith("[output truncated]")


def test_exception_text_is_bounded_and_kernel_remains_usable(kernel):
    result = kernel.run("raise ValueError('é' * 1_100_000)")
    assert not result["ok"]
    assert len(result["error"]) <= kernel.output_limit
    assert result["error"].isascii()
    assert kernel.run("21 * 2")["output"] == "42\n"


def test_timeout_then_reuse(kernel):
    kernel.timeout = 0.05
    result = kernel.run("while True: pass")
    assert not result["ok"] and "KernelTimeout" in result["error"]
    assert kernel.run("6 * 7")["output"] == "42\n"


def test_budget_exhaustion(kernel):
    kernel.max_runs = 1
    assert kernel.run("1")["ok"]
    result = kernel.run("2")
    assert not result["ok"] and "budget" in result["error"]
    assert result["runs"] == 1
