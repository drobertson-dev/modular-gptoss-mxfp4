import pytest
from max.nn.kernels import mxfp4_grouped_matmul_ragged


class _FakeTensor:
    def __init__(self, rank: int, shape: tuple[int, ...]) -> None:
        self.rank = rank
        self.shape = shape
        # Keep a device attribute to match real TensorValue API when building messages.
        self.device = "fake"


def test_mxfp4_grouped_matmul_ragged_raises_on_bad_weight_ranks() -> None:
    q = _FakeTensor(rank=2, shape=(1, 1))  # wrong rank
    e = _FakeTensor(rank=3, shape=(1, 1, 1))
    hidden = _FakeTensor(rank=2, shape=(1, 2))
    start = _FakeTensor(rank=1, shape=(1,))
    ids = _FakeTensor(rank=1, shape=(1,))
    stats = (_FakeTensor(rank=1, shape=(1,)), _FakeTensor(rank=1, shape=(1,)))

    with pytest.raises(ValueError):
        mxfp4_grouped_matmul_ragged(hidden, q, e, start, ids, stats)


def test_mxfp4_grouped_matmul_ragged_raises_on_bad_hidden_rank() -> None:
    q = _FakeTensor(rank=3, shape=(1, 1, 1))
    e = _FakeTensor(rank=3, shape=(1, 1, 1))
    hidden = _FakeTensor(rank=3, shape=(1, 1, 1))  # wrong rank
    start = _FakeTensor(rank=1, shape=(1,))
    ids = _FakeTensor(rank=1, shape=(1,))
    stats = (_FakeTensor(rank=1, shape=(1,)), _FakeTensor(rank=1, shape=(1,)))

    with pytest.raises(ValueError):
        mxfp4_grouped_matmul_ragged(hidden, q, e, start, ids, stats)


def test_mxfp4_grouped_matmul_ragged_raises_on_hidden_width_mismatch() -> None:
    q = _FakeTensor(rank=3, shape=(1, 1, 1))
    e = _FakeTensor(rank=3, shape=(1, 1, 1))
    hidden = _FakeTensor(rank=2, shape=(1, 4))  # width 4, but q last dim => 2
    start = _FakeTensor(rank=1, shape=(1,))
    ids = _FakeTensor(rank=1, shape=(1,))
    stats = (_FakeTensor(rank=1, shape=(1,)), _FakeTensor(rank=1, shape=(1,)))

    with pytest.raises(ValueError):
        mxfp4_grouped_matmul_ragged(hidden, q, e, start, ids, stats)


def test_mxfp4_grouped_matmul_ragged_raises_on_q_e_shape_mismatch() -> None:
    q = _FakeTensor(rank=3, shape=(2, 1, 1))  # expert dim 2
    e = _FakeTensor(rank=3, shape=(1, 1, 1))  # expert dim 1
    hidden = _FakeTensor(rank=2, shape=(1, 2))  # width matches q last dim * 2
    start = _FakeTensor(rank=1, shape=(1,))
    ids = _FakeTensor(rank=1, shape=(1,))
    stats = (_FakeTensor(rank=1, shape=(1,)), _FakeTensor(rank=1, shape=(1,)))

    with pytest.raises(ValueError):
        mxfp4_grouped_matmul_ragged(hidden, q, e, start, ids, stats)
