from math import isnan

import pytest

from src.adaptive_expectile import (
    AdaptiveExpectileState,
)


def clamp(
    value: float,
    lower: float,
    upper: float,
) -> float:

    return max(
        lower,
        min(
            upper,
            value,
        ),
    )


def reference_iir(
    sample: list[float],
    coefficients: list[float],
) -> list[float]:

    output: list[float] = []

    state: float | None = None

    for observation, raw_coefficient in zip(
        sample,
        coefficients,
        strict=True,
    ):

        coefficient = clamp(
            raw_coefficient,
            0.0,
            1.0,
        )

        if state is None:
            state = observation

        else:
            state += (
                coefficient
                * (
                    observation
                    - state
                )
            )

        output.append(
            state
        )

    return output


def test_first_valid_observation_initializes_state():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.95,
        0.20,
    )

    assert state.value == pytest.approx(
        10.0
    )

    assert state.level == pytest.approx(
        0.95
    )


def test_hand_calculated_recursion():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.80,
        0.25,
    )

    assert state.value == pytest.approx(
        10.0
    )

    state.update(
        14.0,
        0.80,
        0.25,
    )

    # innovation = 14 - 10 = 4
    # correction = 0.25 * 0.80 * 4 = 0.8
    assert state.value == pytest.approx(
        10.8
    )

    state.update(
        6.0,
        0.80,
        0.25,
    )

    # innovation = 6 - 10.8 = -4.8
    # correction = 0.25 * 0.20 * -4.8 = -0.24
    assert state.value == pytest.approx(
        10.56
    )


def test_level_095_positive_residual_uses_level_weight():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.95,
        0.20,
    )

    state.update(
        14.0,
        0.95,
        0.20,
    )

    expected = (
        10.0
        + 0.20
        * 0.95
        * (
            14.0
            - 10.0
        )
    )

    assert state.value == pytest.approx(
        expected
    )


def test_level_095_negative_residual_uses_complement_weight():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.95,
        0.20,
    )

    state.update(
        6.0,
        0.95,
        0.20,
    )

    expected = (
        10.0
        + 0.20
        * 0.05
        * (
            6.0
            - 10.0
        )
    )

    assert state.value == pytest.approx(
        expected
    )


def test_level_005_positive_residual_uses_level_weight():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.05,
        0.20,
    )

    state.update(
        14.0,
        0.05,
        0.20,
    )

    expected = (
        10.0
        + 0.20
        * 0.05
        * (
            14.0
            - 10.0
        )
    )

    assert state.value == pytest.approx(
        expected
    )


def test_level_005_negative_residual_uses_complement_weight():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.05,
        0.20,
    )

    state.update(
        6.0,
        0.05,
        0.20,
    )

    expected = (
        10.0
        + 0.20
        * 0.95
        * (
            6.0
            - 10.0
        )
    )

    assert state.value == pytest.approx(
        expected
    )


def test_exact_tie_does_not_move():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.95,
        0.20,
    )

    state.update(
        10.0,
        0.95,
        1.00,
    )

    assert state.value == pytest.approx(
        10.0
    )


def test_alpha_zero_initializes_empty_state():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.95,
        0.0,
    )

    assert state.value == pytest.approx(
        10.0
    )

    assert state.level == pytest.approx(
        0.95
    )


def test_alpha_zero_holds_active_state():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.95,
        0.20,
    )

    state.update(
        100.0,
        0.95,
        0.0,
    )

    assert state.value == pytest.approx(
        10.0
    )


def test_alpha_one_at_center_uses_half_effective_coefficient():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.50,
        0.20,
    )

    state.update(
        25.0,
        0.50,
        1.0,
    )

    # 10 + 1.0 * 0.5 * (25 - 10)
    # = 17.5
    assert state.value == pytest.approx(
        17.5
    )


def test_negative_alpha_clamps_to_zero():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.80,
        0.20,
    )

    state.update(
        20.0,
        0.80,
        -1.0,
    )

    assert state.value == pytest.approx(
        10.0
    )


def test_alpha_above_one_clamps_to_one():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.80,
        0.20,
    )

    state.update(
        20.0,
        0.80,
        2.0,
    )

    # alpha clamps to 1:
    #
    # 10 + 1.0 * 0.8 * (20 - 10)
    # = 18
    assert state.value == pytest.approx(
        18.0
    )


def test_missing_input_preserves_state():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.95,
        0.20,
    )

    previous_value = state.value
    previous_level = state.level

    state.update(
        None,
        0.95,
        0.20,
    )

    assert state.value == previous_value
    assert state.level == previous_level


def test_missing_alpha_preserves_state():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.95,
        0.20,
    )

    previous_value = state.value
    previous_level = state.level

    state.update(
        100.0,
        0.95,
        None,
    )

    assert state.value == previous_value
    assert state.level == previous_level


def test_missing_alpha_does_not_initialize():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.95,
        None,
    )

    assert isnan(
        state.value
    )

    assert isnan(
        state.level
    )


def test_reset_discards_state():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.95,
        0.20,
    )

    state.reset()

    assert isnan(
        state.value
    )

    assert isnan(
        state.level
    )


def test_reset_allows_new_level():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.95,
        0.20,
    )

    state.reset()

    state.update(
        20.0,
        0.05,
        0.20,
    )

    assert state.value == pytest.approx(
        20.0
    )

    assert state.level == pytest.approx(
        0.05
    )


def test_level_cannot_change_inside_active_state():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.95,
        0.20,
    )

    with pytest.raises(
        ValueError
    ):
        state.update(
            11.0,
            0.05,
            0.20,
        )


def test_level_change_is_rejected_even_when_input_is_missing():

    state = AdaptiveExpectileState()

    state.update(
        10.0,
        0.95,
        0.20,
    )

    with pytest.raises(
        ValueError
    ):
        state.update(
            None,
            0.05,
            0.20,
        )


@pytest.mark.parametrize(
    "level",
    [
        float("nan"),
        0.0,
        -0.01,
        1.0,
        1.01,
    ],
)
def test_invalid_level_rejected(
    level: float,
):

    state = AdaptiveExpectileState()

    with pytest.raises(
        ValueError
    ):
        state.update(
            10.0,
            level,
            0.20,
        )


def test_translation_equivariance():

    sample = [
        -2.0,
        1.0,
        4.0,
        -1.0,
        8.0,
        3.0,
    ]

    offset = 17.0

    original = AdaptiveExpectileState()
    translated = AdaptiveExpectileState()

    for observation in sample:

        original.update(
            observation,
            0.95,
            0.20,
        )

        translated.update(
            observation
            + offset,
            0.95,
            0.20,
        )

    assert translated.value == pytest.approx(
        original.value
        + offset
    )


def test_positive_scale_equivariance():

    sample = [
        -2.0,
        1.0,
        4.0,
        -1.0,
        8.0,
        3.0,
    ]

    factor = 7.0

    original = AdaptiveExpectileState()
    scaled = AdaptiveExpectileState()

    for observation in sample:

        original.update(
            observation,
            0.95,
            0.20,
        )

        scaled.update(
            factor
            * observation,
            0.95,
            0.20,
        )

    assert scaled.value == pytest.approx(
        factor
        * original.value
    )


def test_sign_and_complement_level_symmetry():

    sample = [
        -3.0,
        2.0,
        7.0,
        -1.0,
        4.0,
        -8.0,
        6.0,
    ]

    upper = AdaptiveExpectileState()
    reflected_lower = AdaptiveExpectileState()

    for observation in sample:

        upper.update(
            observation,
            0.95,
            0.20,
        )

        reflected_lower.update(
            -observation,
            0.05,
            0.20,
        )

        assert reflected_lower.value == pytest.approx(
            -upper.value
        )


def test_level_half_matches_iir_fixed_alpha_over_full_path():

    sample = [
        10.0,
        14.0,
        8.0,
        20.0,
        -2.0,
        5.0,
        12.0,
    ]

    alphas = [
        0.20,
    ] * len(
        sample
    )

    effective_coefficients = [
        clamp(
            alpha,
            0.0,
            1.0,
        )
        / 2.0
        for alpha in alphas
    ]

    expected = reference_iir(
        sample,
        effective_coefficients,
    )

    state = AdaptiveExpectileState()

    actual: list[float] = []

    for observation, alpha in zip(
        sample,
        alphas,
        strict=True,
    ):

        state.update(
            observation,
            0.50,
            alpha,
        )

        actual.append(
            state.value
        )

    assert actual == pytest.approx(
        expected
    )


def test_level_half_matches_iir_variable_alpha_over_full_path():

    sample = [
        10.0,
        14.0,
        8.0,
        20.0,
        -2.0,
        5.0,
        12.0,
    ]

    alphas = [
        0.00,
        0.05,
        0.20,
        0.80,
        1.00,
        0.40,
        0.10,
    ]

    effective_coefficients = [
        clamp(
            alpha,
            0.0,
            1.0,
        )
        / 2.0
        for alpha in alphas
    ]

    expected = reference_iir(
        sample,
        effective_coefficients,
    )

    state = AdaptiveExpectileState()

    actual: list[float] = []

    for observation, alpha in zip(
        sample,
        alphas,
        strict=True,
    ):

        state.update(
            observation,
            0.50,
            alpha,
        )

        actual.append(
            state.value
        )

    assert actual == pytest.approx(
        expected
    )


def test_directional_correction_ratio_matches_level_odds():

    upward = AdaptiveExpectileState()
    downward = AdaptiveExpectileState()

    upward.update(
        10.0,
        0.90,
        0.50,
    )

    downward.update(
        10.0,
        0.90,
        0.50,
    )

    upward.update(
        14.0,
        0.90,
        0.50,
    )

    downward.update(
        6.0,
        0.90,
        0.50,
    )

    upward_correction = (
        upward.value
        - 10.0
    )

    downward_correction = (
        10.0
        - downward.value
    )

    assert (
        upward_correction
        / downward_correction
    ) == pytest.approx(
        9.0
    )


def test_no_overshoot_across_levels_and_alphas():

    sample = [
        0.0,
        100.0,
        -100.0,
        50.0,
        -25.0,
        200.0,
        -300.0,
        17.5,
    ]

    levels = [
        0.01,
        0.05,
        0.10,
        0.50,
        0.90,
        0.95,
        0.99,
    ]

    alphas = [
        -1.0,
        0.0,
        0.01,
        0.20,
        0.50,
        1.0,
        2.0,
    ]

    tolerance = 1e-12

    for level in levels:

        for alpha in alphas:

            state = AdaptiveExpectileState()

            for observation in sample:

                previous_value = state.value

                state.update(
                    observation,
                    level,
                    alpha,
                )

                if isnan(
                    previous_value
                ):
                    assert state.value == pytest.approx(
                        observation
                    )

                    continue

                lower_bound = min(
                    previous_value,
                    observation,
                )

                upper_bound = max(
                    previous_value,
                    observation,
                )

                assert (
                    state.value
                    >= lower_bound
                    - tolerance
                )

                assert (
                    state.value
                    <= upper_bound
                    + tolerance
                )
