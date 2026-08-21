import pytest

from sampler import ADC_FULL_SCALE, Ring, counts_to_volts


def test_counts_to_volts_spans_the_reference():
    assert counts_to_volts(0, 3.3) == 0
    assert counts_to_volts(ADC_FULL_SCALE, 3.3) == pytest.approx(3.3)
    assert counts_to_volts(ADC_FULL_SCALE // 2, 3.3) == pytest.approx(1.65, abs=1e-4)


def test_latest_returns_oldest_first():
    ring = Ring(8)
    for value in range(5):
        ring.push(value)
    assert ring.latest() == [0, 1, 2, 3, 4]
    assert ring.latest(3) == [2, 3, 4]
    assert ring.latest(99) == [0, 1, 2, 3, 4]


def test_wrapping_keeps_the_newest_capacity_samples():
    ring = Ring(4)
    for value in range(10):
        ring.push(value)
    assert ring.available() == 4
    assert ring.latest() == [6, 7, 8, 9]
    assert ring.total == 10


def test_since_streams_each_sample_exactly_once():
    ring = Ring(64)
    mark, seen = 0, []
    for batch in range(5):
        for value in range(10):
            ring.push(batch * 10 + value)
        mark, dropped, chunk = ring.since(mark)
        assert dropped == 0
        seen.extend(chunk)
    assert seen == list(range(50))
    assert ring.since(mark)[2] == []


def test_since_reports_samples_the_reader_missed():
    ring = Ring(4)
    mark = ring.since(0)[0]
    for value in range(10):
        ring.push(value)
    mark, dropped, chunk = ring.since(mark)
    assert dropped == 6           # 10 pushed, only the last 4 retained
    assert chunk == [6, 7, 8, 9]


def test_since_limit_skips_ahead_instead_of_backlogging():
    ring = Ring(64)
    for value in range(40):
        ring.push(value)
    mark, dropped, chunk = ring.since(0, limit=10)
    assert chunk == list(range(30, 40))   # newest, not oldest
    assert dropped == 30
    assert mark == 40


def test_empty_ring_is_safe_to_read():
    ring = Ring(8)
    assert ring.latest() == []
    assert ring.since(0) == (0, 0, [])
    assert ring.available() == 0


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        Ring(0)
