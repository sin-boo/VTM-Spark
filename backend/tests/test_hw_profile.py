from pathlib import Path

from backend.hw_profile import (
    CALL_MAX_S,
    keys_per_s,
    load_failed,
    load_rates,
    plan_batch,
    predict_call_s,
    profile_key,
    save_failed,
    save_rate,
)

# Seconds per call measured on an RTX 5060 Ti (speed boost on, 1 step).
TI_5060 = {1: 0.102, 2: 0.142, 3: 0.175, 4: 0.227}


def test_line_through_two_sizes_predicts_the_rest() -> None:
    two = {1: TI_5060[1], 2: TI_5060[2]}
    for b in (3, 4):
        guess = predict_call_s(two, b)
        assert guess is not None
        assert abs(guess - TI_5060[b]) / TI_5060[b] < 0.1
    assert predict_call_s({1: 0.1}, 3) is None


def test_fast_card_stays_at_batch_1() -> None:
    # 25 keys/s at x1: reaches 10/s using 40 % of the GPU — bigger only lags.
    assert plan_batch({1: 0.04}, 10.0, 4) == (1, None)


def test_first_run_times_x1_then_x2() -> None:
    assert plan_batch({}, 10.0, 4) == (1, 1)
    # x1 alone cannot hold 10/s under 95 % busy: time x2 before deciding.
    assert plan_batch({1: 0.102}, 10.0, 4) == (1, 2)


def test_5060ti_takes_x2_for_ten_keys() -> None:
    # x1 would be ~100 % busy for 9.8 keys/s; x2 does 10/s at ~71 %.
    assert plan_batch({1: 0.102, 2: 0.142}, 10.0, 4) == (2, None)


def test_5060ti_stays_x1_when_just_under_max_duty() -> None:
    # x1 at 0.0906 s is ~91 % busy for 10/s: under MAX_DUTY, so skip x2's lag.
    assert plan_batch({1: 0.0906, 2: 0.142}, 10.0, 4) == (1, None)


def test_slow_card_stays_at_batch_1_for_latency() -> None:
    """x3 at 0.6 s a call bought ~40 % more keys, and showed every pose over
    a second late. A card already slow at x1 is not timed any bigger."""
    assert plan_batch({1: 0.30}, 10.0, 4) == (1, None)
    assert plan_batch({1: 0.30, 2: 0.45, 3: 0.60}, 10.0, 4) == (1, None)


def test_mid_card_steps_up_only_within_the_call_cap() -> None:
    # x2 reaches 10/s in a 0.17 s call: worth it.
    assert plan_batch({1: 0.12, 2: 0.17}, 10.0, 4) == (2, None)
    # x2 would take 0.22 s a call: stay at x1 and live with fewer keys.
    assert plan_batch({1: 0.16, 2: 0.22}, 10.0, 4) == (1, None)
    # Stepping up still stops at the first size past the cap.
    rates = {1: 0.10, 2: 0.14, 3: 0.18}
    assert CALL_MAX_S < predict_call_s(rates, 4)
    assert plan_batch(rates, 20.0, 4)[0] == 3


def test_sizes_between_measured_ones_are_read_between_them() -> None:
    """A live stream at a fixed x4 records x4 alone. x2 used to be floored at
    x4's time, so with the call cap Auto never even tried it."""
    gap = {1: TI_5060[1], 4: TI_5060[4]}
    guess = predict_call_s(gap, 2)
    assert guess is not None and abs(guess - TI_5060[2]) / TI_5060[2] < 0.05
    assert plan_batch(gap, 10.0, 4) == (2, 2)


def test_out_of_memory_sizes_are_skipped() -> None:
    rates = {1: 0.10, 2: 0.14, 3: 0.18}
    assert plan_batch(rates, 20.0, 4, failed={3})[0] == 2


def test_profile_round_trip_and_blend(tmp_path: Path) -> None:
    path = tmp_path / "hw.json"
    key = profile_key("RTX 5060 Ti", "VTM-1.5.1.pt", 1, True)
    assert load_rates(key, path) == {}
    save_rate(key, 1, 0.10, path=path)
    save_rate(key, 2, 0.14, path=path)
    # A live stream nudges instead of replacing.
    rates = save_rate(key, 1, 0.20, weight=0.3, path=path)
    assert abs(rates[1] - 0.13) < 1e-6
    assert abs(keys_per_s(rates, 2) - 2 / 0.14) < 1e-6
    save_failed(key, 4, path=path)
    assert load_failed(key, path) == {4}
    # Another card / model is its own entry.
    assert load_rates(profile_key("RTX 3060", "VTM-1.5.1.pt", 1, True), path) == {}
