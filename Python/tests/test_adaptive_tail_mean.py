from math import isnan

import pytest

from src.adaptive_tail_mean import (
    AdaptiveTailMeanState,
)


def test_first_lower_tail_observation_initializes_normalized_mean():

    state = AdaptiveTailMeanState()

    state.update(
        -2.0,
        -1.0,
        0.20,
        "lower",
    )

    assert state.value == pytest.approx(
        -2.0
    )

    assert state.weighted_tail == pytest.approx(
        -0.4
    )

    assert state.tail_mass == pytest.approx(
        0.20
    )

    assert state.side == "lower"


def test_first_upper_tail_observation_initializes_normalized_mean():

    state = AdaptiveTailMeanState()

    state.update(
        3.0,
        1.0,
        0.25,
        "upper",
    )

    assert state.value == pytest.approx(
        3.0
    )

    assert state.weighted_tail == pytest.approx(
        0.75
    )

    assert state.tail_mass == pytest.approx(
        0.25
    )

    assert state.side == "upper"


def test_positive_weight_non_tail_observation_binds_side_but_not_value():

    state = AdaptiveTailMeanState()

    state.update(
        2.0,
        -1.0,
        0.20,
        "lower",
    )

    assert isnan(
        state.value
    )

    assert state.weighted_tail == pytest.approx(
        0.0
    )

    assert state.tail_mass == pytest.approx(
        0.0
    )

    assert state.side == "lower"


def test_alpha_zero_does_not_initialize_empty_state():

    state = AdaptiveTailMeanState()

    state.update(
        -2.0,
        -1.0,
        0.0,
        "lower",
    )

    assert isnan(
        state.value
    )

    assert state.weighted_tail == pytest.approx(
        0.0
    )

    assert state.tail_mass == pytest.approx(
        0.0
    )

    assert state.side is None


def test_negative_alpha_clamps_to_zero():

    state = AdaptiveTailMeanState()

    state.update(
        -2.0,
        -1.0,
        -1.0,
        "lower",
    )

    assert isnan(
        state.value
    )

    assert state.weighted_tail == pytest.approx(
        0.0
    )

    assert state.tail_mass == pytest.approx(
        0.0
    )

    assert state.side is None


def test_non_tail_observation_preserves_defined_tail_mean():

    state = AdaptiveTailMeanState()

    state.update(
        -2.0,
        -1.0,
        0.20,
        "lower",
    )

    previous_value = state.value

    state.update(
        5.0,
        -1.0,
        0.20,
        "lower",
    )

    assert state.value == pytest.approx(
        previous_value
    )

    assert state.weighted_tail == pytest.approx(
        -0.32
    )

    assert state.tail_mass == pytest.approx(
        0.16
    )


def test_hand_calculated_tail_update():

    state = AdaptiveTailMeanState()

    state.update(
        -2.0,
        -1.0,
        0.20,
        "lower",
    )

    state.update(
        -4.0,
        -1.0,
        0.20,
        "lower",
    )

    expected_weighted_tail = (
        0.80
        * -0.40
        + 0.20
        * -4.0
    )

    expected_tail_mass = (
        0.80
        * 0.20
        + 0.20
    )

    expected_value = (
        expected_weighted_tail
        / expected_tail_mass
    )

    assert state.weighted_tail == pytest.approx(
        expected_weighted_tail
    )

    assert state.tail_mass == pytest.approx(
        expected_tail_mass
    )

    assert state.value == pytest.approx(
        expected_value
    )


def test_tail_observation_moves_value_between_previous_mean_and_observation():

    state = AdaptiveTailMeanState()

    state.update(
        -2.0,
        -1.0,
        0.10,
        "lower",
    )

    previous_value = state.value

    observation = -8.0

    state.update(
        observation,
        -1.0,
        0.10,
        "lower",
    )

    assert (
        observation
        <= state.value
        <= previous_value
    )


def test_upper_tail_observation_moves_value_between_previous_mean_and_observation():

    state = AdaptiveTailMeanState()

    state.update(
        2.0,
        1.0,
        0.10,
        "upper",
    )

    previous_value = state.value

    observation = 8.0

    state.update(
        observation,
        1.0,
        0.10,
        "upper",
    )

    assert (
        previous_value
        <= state.value
        <= observation
    )


def test_alpha_above_one_clamps_to_one_for_tail_observation():

    state = AdaptiveTailMeanState()

    state.update(
        -2.0,
        -1.0,
        0.20,
        "lower",
    )

    state.update(
        -7.0,
        -1.0,
        2.0,
        "lower",
    )

    assert state.value == pytest.approx(
        -7.0
    )

    assert state.weighted_tail == pytest.approx(
        -7.0
    )

    assert state.tail_mass == pytest.approx(
        1.0
    )


def test_alpha_one_non_tail_observation_clears_effective_tail_population():

    state = AdaptiveTailMeanState()

    state.update(
        -2.0,
        -1.0,
        0.20,
        "lower",
    )

    state.update(
        10.0,
        -1.0,
        1.0,
        "lower",
    )

    assert isnan(
        state.value
    )

    assert state.weighted_tail == pytest.approx(
        0.0
    )

    assert state.tail_mass == pytest.approx(
        0.0
    )

    assert state.side == "lower"


def test_lower_tail_includes_observation_equal_to_threshold():

    state = AdaptiveTailMeanState()

    state.update(
        -1.0,
        -1.0,
        0.20,
        "lower",
    )

    assert state.value == pytest.approx(
        -1.0
    )

    assert state.tail_mass == pytest.approx(
        0.20
    )


def test_upper_tail_includes_observation_equal_to_threshold():

    state = AdaptiveTailMeanState()

    state.update(
        1.0,
        1.0,
        0.20,
        "upper",
    )

    assert state.value == pytest.approx(
        1.0
    )

    assert state.tail_mass == pytest.approx(
        0.20
    )


def test_missing_input_preserves_complete_state():

    state = AdaptiveTailMeanState()

    state.update(
        -2.0,
        -1.0,
        0.20,
        "lower",
    )

    previous = (
        state.value,
        state.weighted_tail,
        state.tail_mass,
        state.side,
    )

    state.update(
        None,
        -1.0,
        0.20,
        "lower",
    )

    assert (
        state.value,
        state.weighted_tail,
        state.tail_mass,
        state.side,
    ) == previous


def test_missing_threshold_preserves_complete_state():

    state = AdaptiveTailMeanState()

    state.update(
        -2.0,
        -1.0,
        0.20,
        "lower",
    )

    previous = (
        state.value,
        state.weighted_tail,
        state.tail_mass,
        state.side,
    )

    state.update(
        -4.0,
        None,
        0.20,
        "lower",
    )

    assert (
        state.value,
        state.weighted_tail,
        state.tail_mass,
        state.side,
    ) == previous


def test_missing_alpha_preserves_complete_state():

    state = AdaptiveTailMeanState()

    state.update(
        -2.0,
        -1.0,
        0.20,
        "lower",
    )

    previous = (
        state.value,
        state.weighted_tail,
        state.tail_mass,
        state.side,
    )

    state.update(
        -4.0,
        -1.0,
        None,
        "lower",
    )

    assert (
        state.value,
        state.weighted_tail,
        state.tail_mass,
        state.side,
    ) == previous


def test_reset_discards_complete_state():

    state = AdaptiveTailMeanState()

    state.update(
        -2.0,
        -1.0,
        0.20,
        "lower",
    )

    state.reset()

    assert isnan(
        state.value
    )

    assert state.weighted_tail == pytest.approx(
        0.0
    )

    assert state.tail_mass == pytest.approx(
        0.0
    )

    assert state.side is None


def test_reset_allows_new_side():

    state = AdaptiveTailMeanState()

    state.update(
        -2.0,
        -1.0,
        0.20,
        "lower",
    )

    state.reset()

    state.update(
        2.0,
        1.0,
        0.20,
        "upper",
    )

    assert state.side == "upper"

    assert state.value == pytest.approx(
        2.0
    )


def test_side_cannot_change_inside_active_state():

    state = AdaptiveTailMeanState()

    state.update(
        -2.0,
        -1.0,
        0.20,
        "lower",
    )

    with pytest.raises(
        ValueError
    ):
        state.update(
            2.0,
            1.0,
            0.20,
            "upper",
        )


def test_side_change_is_rejected_even_when_observation_is_missing():

    state = AdaptiveTailMeanState()

    state.update(
        -2.0,
        -1.0,
        0.20,
        "lower",
    )

    with pytest.raises(
        ValueError
    ):
        state.update(
            None,
            1.0,
            0.20,
            "upper",
        )


@pytest.mark.parametrize(
    "side",
    [
        "",
        "Lower",
        "Upper",
        "left",
        "right",
    ],
)
def test_invalid_side_rejected(
    side: str,
):

    state = AdaptiveTailMeanState()

    with pytest.raises(
        ValueError
    ):
        state.update(
            -2.0,
            -1.0,
            0.20,
            side,
        )


def test_reflection_symmetry():

    sample = [
        -3.0,
        2.0,
        7.0,
        -1.0,
        4.0,
        -8.0,
        6.0,
        0.5,
    ]

    thresholds = [
        -1.0,
        -0.8,
        -0.4,
        -0.6,
        -0.2,
        -1.2,
        -0.7,
        -0.5,
    ]

    lower = AdaptiveTailMeanState()
    mirrored_upper = AdaptiveTailMeanState()

    for observation, threshold in zip(
        sample,
        thresholds,
        strict=True,
    ):

        lower.update(
            observation,
            threshold,
            0.05,
            "lower",
        )

        mirrored_upper.update(
            -observation,
            -threshold,
            0.05,
            "upper",
        )

        if isnan(
            lower.value
        ):
            assert isnan(
                mirrored_upper.value
            )

        else:
            assert mirrored_upper.value == pytest.approx(
                -lower.value
            )

        assert mirrored_upper.tail_mass == pytest.approx(
            lower.tail_mass
        )

        assert mirrored_upper.weighted_tail == pytest.approx(
            -lower.weighted_tail
        )


def test_translation_equivariance():

    sample = [
        -3.0,
        2.0,
        -1.0,
        -8.0,
        0.5,
        -4.0,
    ]

    thresholds = [
        -1.0,
        -0.5,
        -0.7,
        -1.2,
        -0.3,
        -0.8,
    ]

    offset = 17.0

    original = AdaptiveTailMeanState()
    translated = AdaptiveTailMeanState()

    for observation, threshold in zip(
        sample,
        thresholds,
        strict=True,
    ):

        original.update(
            observation,
            threshold,
            0.05,
            "lower",
        )

        translated.update(
            observation
            + offset,
            threshold
            + offset,
            0.05,
            "lower",
        )

    assert translated.value == pytest.approx(
        original.value
        + offset
    )

    assert translated.tail_mass == pytest.approx(
        original.tail_mass
    )


def test_positive_scale_equivariance():

    sample = [
        -3.0,
        2.0,
        -1.0,
        -8.0,
        0.5,
        -4.0,
    ]

    thresholds = [
        -1.0,
        -0.5,
        -0.7,
        -1.2,
        -0.3,
        -0.8,
    ]

    factor = 7.0

    original = AdaptiveTailMeanState()
    scaled = AdaptiveTailMeanState()

    for observation, threshold in zip(
        sample,
        thresholds,
        strict=True,
    ):

        original.update(
            observation,
            threshold,
            0.05,
            "lower",
        )

        scaled.update(
            factor
            * observation,
            factor
            * threshold,
            0.05,
            "lower",
        )

    assert scaled.value == pytest.approx(
        factor
        * original.value
    )

    assert scaled.weighted_tail == pytest.approx(
        factor
        * original.weighted_tail
    )

    assert scaled.tail_mass == pytest.approx(
        original.tail_mass
    )


def test_tail_mass_always_remains_in_unit_interval():

    sample = [
        -4.0,
        3.0,
        -2.0,
        5.0,
        -8.0,
        1.0,
        -1.0,
        -10.0,
    ]

    alphas = [
        -1.0,
        0.0,
        0.05,
        0.20,
        0.50,
        1.0,
        2.0,
        0.10,
    ]

    state = AdaptiveTailMeanState()

    for observation, alpha in zip(
        sample,
        alphas,
        strict=True,
    ):

        state.update(
            observation,
            -1.0,
            alpha,
            "lower",
        )

        assert (
            0.0
            <= state.tail_mass
            <= 1.0
        )
