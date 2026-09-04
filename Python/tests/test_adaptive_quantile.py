from math import isnan, nan

import pytest

from src.adaptive_quantile import (
    AdaptiveQuantileState,
)


def snapshot(
    state: AdaptiveQuantileState,
) -> tuple[float, float, float]:

    return (
        state.value,
        state.probability,
        state.scale,
    )


def run_state(
    sample: list[float],
    probability: float,
    alpha: float,
) -> AdaptiveQuantileState:

    state = AdaptiveQuantileState()

    for observation in sample:
        state.update(
            observation,
            probability,
            alpha,
        )

    return state


def test_first_valid_observation_initializes_state():
    state = AdaptiveQuantileState()

    state.update(
        10.0,
        0.5,
        0.2,
    )

    assert state.value == 10.0
    assert state.probability == 0.5
    assert state.scale == 0.0


def test_second_observation_builds_scale_without_quantile_move():
    state = AdaptiveQuantileState()

    state.update(
        10.0,
        0.5,
        0.2,
    )

    state.update(
        14.0,
        0.5,
        0.2,
    )

    # Previous scale was zero, so the second observation
    # cannot move the quantile.
    assert state.value == 10.0

    # scale =
    #     0 + 0.2 * |14 - 10|
    #     0.8
    assert state.scale == pytest.approx(
        0.8,
        abs=1e-12,
    )


def test_third_observation_uses_previous_scale():
    state = AdaptiveQuantileState()

    state.update(
        10.0,
        0.5,
        0.2,
    )

    state.update(
        14.0,
        0.5,
        0.2,
    )

    state.update(
        14.0,
        0.5,
        0.2,
    )

    # step =
    #     0.2 * 0.8
    #     0.16
    #
    # median score = 0.5
    #
    # q =
    #     10 + 0.16 * 0.5
    #     10.08
    assert state.value == pytest.approx(
        10.08,
        abs=1e-12,
    )

    # scale =
    #     0.8
    #     + 0.2 * (4.0 - 0.8)
    #     1.44
    assert state.scale == pytest.approx(
        1.44,
        abs=1e-12,
    )


def test_asymmetric_quantile_score():
    upper = AdaptiveQuantileState()

    upper.update(
        10.0,
        0.95,
        0.5,
    )

    upper.update(
        12.0,
        0.95,
        0.5,
    )

    upper.update(
        12.0,
        0.95,
        0.5,
    )

    # Previous scale = 1.
    #
    # step =
    #     0.5 * 1
    #
    # upward score =
    #     0.95
    #
    # correction =
    #     0.475
    assert upper.value == pytest.approx(
        10.475,
        abs=1e-12,
    )

    lower = AdaptiveQuantileState()

    lower.update(
        10.0,
        0.95,
        0.5,
    )

    lower.update(
        8.0,
        0.95,
        0.5,
    )

    lower.update(
        8.0,
        0.95,
        0.5,
    )

    # downward score =
    #     0.95 - 1
    #     -0.05
    #
    # correction =
    #     0.5 * 1 * -0.05
    #     -0.025
    assert lower.value == pytest.approx(
        9.975,
        abs=1e-12,
    )


def test_exact_tie_does_not_move_quantile():
    state = AdaptiveQuantileState()

    state.update(
        10.0,
        0.5,
        0.5,
    )

    state.update(
        12.0,
        0.5,
        0.5,
    )

    assert state.value == 10.0
    assert state.scale == 1.0

    state.update(
        10.0,
        0.5,
        0.5,
    )

    # Exact tie gives score zero.
    assert state.value == 10.0

    # Scale still adapts because the current absolute innovation is zero.
    assert state.scale == pytest.approx(
        0.5,
        abs=1e-12,
    )


def test_zero_alpha_holds_active_state():
    state = AdaptiveQuantileState()

    state.update(
        10.0,
        0.5,
        0.5,
    )

    state.update(
        14.0,
        0.5,
        0.5,
    )

    previous = snapshot(
        state
    )

    state.update(
        100.0,
        0.5,
        0.0,
    )

    assert snapshot(
        state
    ) == previous


def test_missing_input_preserves_state():
    state = AdaptiveQuantileState()

    state.update(
        10.0,
        0.5,
        0.5,
    )

    state.update(
        14.0,
        0.5,
        0.5,
    )

    previous = snapshot(
        state
    )

    state.update(
        nan,
        0.5,
        0.5,
    )

    assert snapshot(
        state
    ) == previous


def test_missing_alpha_preserves_state():
    state = AdaptiveQuantileState()

    state.update(
        10.0,
        0.5,
        0.5,
    )

    state.update(
        14.0,
        0.5,
        0.5,
    )

    previous = snapshot(
        state
    )

    state.update(
        100.0,
        0.5,
        nan,
    )

    assert snapshot(
        state
    ) == previous


def test_missing_alpha_does_not_initialize_state():
    state = AdaptiveQuantileState()

    state.update(
        10.0,
        0.5,
        nan,
    )

    assert isnan(
        state.value
    )

    assert isnan(
        state.probability
    )

    assert isnan(
        state.scale
    )


def test_reset_discards_state():
    state = AdaptiveQuantileState()

    state.update(
        10.0,
        0.5,
        0.5,
    )

    state.update(
        14.0,
        0.5,
        0.5,
    )

    state.reset()

    assert isnan(
        state.value
    )

    assert isnan(
        state.probability
    )

    assert isnan(
        state.scale
    )


def test_reset_allows_new_probability():
    state = AdaptiveQuantileState()

    state.update(
        10.0,
        0.5,
        0.5,
    )

    state.reset()

    state.update(
        20.0,
        0.95,
        0.5,
    )

    assert state.value == 20.0
    assert state.probability == 0.95
    assert state.scale == 0.0


def test_probability_cannot_change_inside_active_state():
    state = AdaptiveQuantileState()

    state.update(
        10.0,
        0.5,
        0.5,
    )

    with pytest.raises(
        ValueError,
        match="cannot change",
    ):
        state.update(
            20.0,
            0.95,
            0.5,
        )


@pytest.mark.parametrize(
    "probability",
    [
        nan,
        0.0,
        -0.01,
        1.0,
        1.01,
    ],
)
def test_invalid_probability_is_rejected(
    probability,
):
    state = AdaptiveQuantileState()

    with pytest.raises(
        ValueError,
        match="strictly between zero and one",
    ):
        state.update(
            10.0,
            probability,
            0.5,
        )


def test_negative_alpha_clamps_to_zero():
    state = AdaptiveQuantileState()

    state.update(
        10.0,
        0.5,
        0.5,
    )

    state.update(
        14.0,
        0.5,
        0.5,
    )

    previous = snapshot(
        state
    )

    state.update(
        100.0,
        0.5,
        -1.0,
    )

    assert snapshot(
        state
    ) == previous


def test_alpha_above_one_clamps_to_one():
    state = AdaptiveQuantileState()

    state.update(
        10.0,
        0.5,
        0.5,
    )

    state.update(
        14.0,
        0.5,
        0.5,
    )

    assert state.value == 10.0
    assert state.scale == 2.0

    state.update(
        14.0,
        0.5,
        2.0,
    )

    # alpha clamps to 1.
    #
    # q =
    #     10
    #     + 1 * 2 * 0.5
    #     11
    assert state.value == pytest.approx(
        11.0,
        abs=1e-12,
    )

    # scale =
    #     2 + 1 * (4 - 2)
    #     4
    assert state.scale == pytest.approx(
        4.0,
        abs=1e-12,
    )


def test_translation_equivariance():
    sample = [
        10.0,
        12.0,
        8.0,
        15.0,
        9.0,
        11.0,
    ]

    offset = 137.0

    original = run_state(
        sample,
        0.95,
        0.2,
    )

    translated = run_state(
        [
            observation + offset
            for observation in sample
        ],
        0.95,
        0.2,
    )

    assert translated.value == pytest.approx(
        original.value + offset,
        abs=1e-12,
    )

    assert translated.scale == pytest.approx(
        original.scale,
        abs=1e-12,
    )


def test_positive_scale_equivariance():
    sample = [
        10.0,
        12.0,
        8.0,
        15.0,
        9.0,
        11.0,
    ]

    multiplier = 1000.0

    original = run_state(
        sample,
        0.05,
        0.2,
    )

    scaled = run_state(
        [
            multiplier
            * observation
            for observation in sample
        ],
        0.05,
        0.2,
    )

    assert scaled.value == pytest.approx(
        multiplier
        * original.value,
        abs=1e-9,
    )

    assert scaled.scale == pytest.approx(
        multiplier
        * original.scale,
        abs=1e-9,
    )
