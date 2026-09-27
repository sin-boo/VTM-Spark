from pathlib import Path

from backend.load_timing import StageClock, StageMeter, StagePlan, stage_fill


def test_stage_fills_to_ninety_percent_on_time_then_eases() -> None:
    assert stage_fill(0.0, 10.0) == 0.0
    assert abs(stage_fill(5.0, 10.0) - 0.45) < 1e-9
    assert abs(stage_fill(10.0, 10.0) - 0.9) < 1e-9
    late = stage_fill(30.0, 10.0)
    assert 0.9 < late < 0.99


def test_plan_weights_stages_by_expected_seconds() -> None:
    plan = StagePlan([("fast", 1.0), ("slow", 9.0)])
    assert abs(plan.done_through("fast") - 0.1) < 1e-9
    # Halfway through the slow stage: fast done + 45 % of slow's slice.
    assert abs(plan.fraction("slow", 4.5) - (1.0 + 0.45 * 9.0) / 10.0) < 1e-9
    # A real measure (bytes read) wins over the timer when it is ahead.
    assert plan.fraction("fast", 0.0, inner=0.8) > plan.fraction("fast", 0.0)


def test_clock_remembers_measured_times(tmp_path: Path) -> None:
    path = tmp_path / "timings.json"
    clock = StageClock(path)
    clock.record("decode", 6.0)
    clock.record("dit", 4.0, gb=2.0)  # stored per GB
    again = StageClock(path)
    assert abs(again.expected("decode") - 6.0) < 1e-9
    assert abs(again.expected("dit", gb=1.0) - 2.0) < 1e-9
    again.record("decode", 8.0)
    assert 6.0 < again.expected("decode") < 8.0


def test_meter_records_each_finished_stage(tmp_path: Path) -> None:
    clock = StageClock(tmp_path / "timings.json")
    seen: list[float] = []
    with StageMeter(clock, ["a", "b"], lambda v, _label: seen.append(v), tick_s=10.0) as meter:
        meter.stage("a", "A")
        meter.stage("b", "B")
        meter.stage("done", "Ready")
    saved = StageClock(tmp_path / "timings.json")
    assert saved.expected("a") > 0 and saved.expected("b") > 0
    assert seen == sorted(seen)  # never steps back
