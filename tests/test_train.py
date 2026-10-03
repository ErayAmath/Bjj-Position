"""Tests for owner task B (`train_step`). They fail until the function is implemented."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bjj.train import PositionMLP, train_step  # noqa: E402


@pytest.fixture
def tiny_problem():
    torch.manual_seed(0)
    x = torch.randn(32, 8)
    y = (x[:, 0] > 0).long()          # trivially learnable rule
    model = PositionMLP(8, 2, hidden=32, dropout=0.0)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    return model, x, y, loss_fn, optimizer


def test_train_step_returns_a_float_loss(tiny_problem):
    model, x, y, loss_fn, optimizer = tiny_problem
    loss = train_step(model, x, y, loss_fn, optimizer)
    assert isinstance(loss, float)
    assert loss > 0


def test_model_overfits_a_tiny_batch(tiny_problem):
    """The classic sanity check: a model that cannot memorise 32 examples has a bug."""
    model, x, y, loss_fn, optimizer = tiny_problem
    model.train()
    losses = [train_step(model, x, y, loss_fn, optimizer) for _ in range(150)]
    assert losses[-1] < 0.05, f"loss did not go down: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_parameters_actually_change(tiny_problem):
    model, x, y, loss_fn, optimizer = tiny_problem
    before = [p.detach().clone() for p in model.parameters()]
    train_step(model, x, y, loss_fn, optimizer)
    assert any(not torch.allclose(a, b) for a, b in zip(before, model.parameters()))


def test_gradients_are_cleared_between_steps(tiny_problem):
    """If zero_grad() is missing, gradients accumulate and keep growing step after step."""
    model, x, y, loss_fn, optimizer = tiny_problem
    model.train()
    train_step(model, x, y, loss_fn, optimizer)
    first = model.net[0].weight.grad.abs().sum().item()
    for _ in range(5):
        train_step(model, x, y, loss_fn, optimizer)
    later = model.net[0].weight.grad.abs().sum().item()
    assert later < first * 3, "gradients look accumulated — is zero_grad() missing?"


def test_no_grad_leaks_into_the_returned_loss(tiny_problem):
    """Returning the tensor instead of a float keeps the whole graph alive (memory leak)."""
    model, x, y, loss_fn, optimizer = tiny_problem
    assert not isinstance(train_step(model, x, y, loss_fn, optimizer), torch.Tensor)


def test_np_import_is_available():
    assert np.array([1]).sum() == 1


def test_extend_output_layer_keeps_the_trained_classes():
    from bjj.train import extend_output_layer
    torch.manual_seed(0)
    model = PositionMLP(8, 3, hidden=32, dropout=0.0).eval()
    x = torch.randn(5, 8)
    before = model(x)
    wider = extend_output_layer(model, 5).eval()
    after = wider(x)
    assert after.shape == (5, 5)
    torch.testing.assert_close(after[:, :3], before)       # old outputs unchanged
    assert (after[:, 3:] < after[:, :3].min()).all()        # new classes start out unlikely


def test_extend_output_layer_is_a_no_op_when_wide_enough():
    from bjj.train import extend_output_layer
    model = PositionMLP(8, 4, hidden=16)
    assert extend_output_layer(model, 4).net[-1].out_features == 4
