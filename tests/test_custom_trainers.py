"""Unit tests for the FedProx / SCAFFOLD gradient-injection logic
(hafl/custom_trainers.py), using a plain nn.Linear instead of a real YOLO
model + dataset so these run in seconds with no GPU or data needed. This is
the part most at risk of silently doing nothing if the ultralytics
`optimizer_step` hook point ever changes, so it is checked against a
hand-computed expected gradient."""

import torch
import torch.nn as nn

from hafl.custom_trainers import inject_fedprox_gradient, inject_scaffold_gradient


def _tiny_model_with_grad():
    torch.manual_seed(0)
    m = nn.Linear(2, 1, bias=False)
    m.weight.data = torch.tensor([[1.0, 2.0]])
    loss = (m(torch.tensor([1.0, 1.0])) ** 2).sum()
    loss.backward()
    return m


def test_inject_fedprox_gradient_adds_expected_term():
    m = _tiny_model_with_grad()
    grad_before = m.weight.grad.clone()
    global_weights = {"weight": torch.tensor([[0.0, 0.0]])}
    inject_fedprox_gradient(m, mu=0.5, global_weights=global_weights)
    expected_addition = 0.5 * (m.weight.data - global_weights["weight"])
    assert torch.allclose(m.weight.grad, grad_before + expected_addition)


def test_inject_fedprox_gradient_noop_when_mu_zero():
    m = _tiny_model_with_grad()
    grad_before = m.weight.grad.clone()
    inject_fedprox_gradient(m, mu=0.0, global_weights={"weight": torch.zeros(1, 2)})
    assert torch.allclose(m.weight.grad, grad_before)


def test_inject_fedprox_gradient_noop_when_no_global_weights():
    m = _tiny_model_with_grad()
    grad_before = m.weight.grad.clone()
    inject_fedprox_gradient(m, mu=1.0, global_weights=None)
    assert torch.allclose(m.weight.grad, grad_before)


def test_inject_scaffold_gradient_adds_correction_directly():
    m = _tiny_model_with_grad()
    grad_before = m.weight.grad.clone()
    correction = {"weight": torch.tensor([[0.1, -0.2]])}
    inject_scaffold_gradient(m, correction)
    assert torch.allclose(m.weight.grad, grad_before + correction["weight"])


def test_inject_scaffold_gradient_noop_when_correction_empty():
    m = _tiny_model_with_grad()
    grad_before = m.weight.grad.clone()
    inject_scaffold_gradient(m, {})
    assert torch.allclose(m.weight.grad, grad_before)


def test_inject_functions_skip_unmatched_parameter_names():
    m = _tiny_model_with_grad()
    grad_before = m.weight.grad.clone()
    inject_fedprox_gradient(m, mu=1.0, global_weights={"some_other_param": torch.zeros(1, 2)})
    assert torch.allclose(m.weight.grad, grad_before)
