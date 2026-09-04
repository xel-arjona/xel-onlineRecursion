from math import isnan

import pytest

from src.adaptive_huber import (
    AdaptiveHuberState,
)


def test_first_valid_observation_initializes_directly():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        2.0,
        1.5,
        0.20,
    )

    assert state.value == pytest.approx(
        10.0
    )

    assert state.tuning == pytest.approx(
        1.5
    )


def test_inner_region_exactly_matches_iir():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        1.0,
        1.5,
        0.20,
    )

    state.update(
        11.0,
        1.0,
        1.5,
        0.20,
    )

    assert state.value == pytest.approx(
        10.2
    )


def test_positive_outlier_correction_is_clipped():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        2.0,
        1.5,
        0.20,
    )

    state.update(
        1_000.0,
        2.0,
        1.5,
        0.20,
    )

    assert state.value == pytest.approx(
        10.6
    )


def test_negative_outlier_correction_is_clipped():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        2.0,
        1.5,
        0.20,
    )

    state.update(
        -1_000.0,
        2.0,
        1.5,
        0.20,
    )

    assert state.value == pytest.approx(
        9.4
    )


def test_alpha_zero_initializes_empty_state():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        2.0,
        1.5,
        0.0,
    )

    assert state.value == pytest.approx(
        10.0
    )


def test_alpha_zero_freezes_active_state():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        2.0,
        1.5,
        0.20,
    )

    state.update(
        100.0,
        2.0,
        1.5,
        0.0,
    )

    assert state.value == pytest.approx(
        10.0
    )


def test_negative_alpha_clamps_to_zero():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        2.0,
        1.5,
        0.20,
    )

    state.update(
        100.0,
        2.0,
        1.5,
        -1.0,
    )

    assert state.value == pytest.approx(
        10.0
    )


def test_alpha_above_one_clamps_to_one_inside_region():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        10.0,
        1.5,
        0.20,
    )

    state.update(
        12.0,
        10.0,
        1.5,
        2.0,
    )

    assert state.value == pytest.approx(
        12.0
    )


def test_alpha_above_one_remains_clipped_outside_region():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        1.0,
        1.5,
        0.20,
    )

    state.update(
        100.0,
        1.0,
        1.5,
        2.0,
    )

    assert state.value == pytest.approx(
        11.5
    )


def test_zero_scale_initializes_empty_state():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        0.0,
        1.5,
        0.20,
    )

    assert state.value == pytest.approx(
        10.0
    )


def test_zero_scale_freezes_active_state():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        2.0,
        1.5,
        0.20,
    )

    state.update(
        100.0,
        0.0,
        1.5,
        0.20,
    )

    assert state.value == pytest.approx(
        10.0
    )


def test_external_scale_may_change():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        1.0,
        1.5,
        0.20,
    )

    state.update(
        14.0,
        1.0,
        1.5,
        0.20,
    )

    assert state.value == pytest.approx(
        10.3
    )

    state.update(
        14.0,
        10.0,
        1.5,
        0.20,
    )

    assert state.value == pytest.approx(
        11.04
    )


def test_missing_input_preserves_state():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        2.0,
        1.5,
        0.20,
    )

    previous = (
        state.value,
        state.tuning,
    )

    state.update(
        None,
        2.0,
        1.5,
        0.20,
    )

    assert (
        state.value,
        state.tuning,
    ) == previous


def test_missing_scale_preserves_state():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        2.0,
        1.5,
        0.20,
    )

    previous = (
        state.value,
        state.tuning,
    )

    state.update(
        20.0,
        None,
        1.5,
        0.20,
    )

    assert (
        state.value,
        state.tuning,
    ) == previous


def test_missing_alpha_preserves_state():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        2.0,
        1.5,
        0.20,
    )

    previous = (
        state.value,
        state.tuning,
    )

    state.update(
        20.0,
        2.0,
        1.5,
        None,
    )

    assert (
        state.value,
        state.tuning,
    ) == previous


@pytest.mark.parametrize(
    "tuning",
    [
        0.0,
        -1.0,
        float(
            "nan"
        ),
    ],
)
def test_invalid_tuning_rejected(
    tuning: float,
):

    state = AdaptiveHuberState()

    with pytest.raises(
        ValueError
    ):
        state.update(
            10.0,
            1.0,
            tuning,
            0.20,
        )


def test_negative_scale_rejected():

    state = AdaptiveHuberState()

    with pytest.raises(
        ValueError
    ):
        state.update(
            10.0,
            -1.0,
            1.5,
            0.20,
        )


def test_tuning_cannot_change_inside_active_state():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        1.0,
        1.5,
        0.20,
    )

    with pytest.raises(
        ValueError
    ):
        state.update(
            11.0,
            1.0,
            2.0,
            0.20,
        )


def test_tuning_change_rejected_even_when_input_missing():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        1.0,
        1.5,
        0.20,
    )

    with pytest.raises(
        ValueError
    ):
        state.update(
            None,
            1.0,
            2.0,
            0.20,
        )


def test_negative_scale_rejected_even_when_input_missing():

    state = AdaptiveHuberState()

    with pytest.raises(
        ValueError
    ):
        state.update(
            None,
            -1.0,
            1.5,
            0.20,
        )


def test_reset_clears_complete_state():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        1.0,
        1.5,
        0.20,
    )

    state.reset()

    assert isnan(
        state.value
    )

    assert isnan(
        state.tuning
    )


def test_reset_allows_new_tuning():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        1.0,
        1.5,
        0.20,
    )

    state.reset()

    state.update(
        20.0,
        1.0,
        2.0,
        0.20,
    )

    assert state.value == pytest.approx(
        20.0
    )

    assert state.tuning == pytest.approx(
        2.0
    )


def test_reflection_symmetry():

    sample = [
        3.0,
        -7.0,
        2.0,
        20.0,
        -1.0,
        -30.0,
        4.0,
    ]

    scales = [
        1.0,
        2.0,
        1.5,
        3.0,
        0.5,
        4.0,
        2.0,
    ]

    positive = AdaptiveHuberState()
    negative = AdaptiveHuberState()

    for observation, scale in zip(
        sample,
        scales,
        strict=True,
    ):

        positive.update(
            observation,
            scale,
            1.5,
            0.10,
        )

        negative.update(
            -observation,
            scale,
            1.5,
            0.10,
        )

        assert negative.value == pytest.approx(
            -positive.value
        )


def test_translation_equivariance():

    sample = [
        3.0,
        -7.0,
        2.0,
        20.0,
        -1.0,
    ]

    offset = 100.0

    original = AdaptiveHuberState()
    translated = AdaptiveHuberState()

    for observation in sample:

        original.update(
            observation,
            2.0,
            1.5,
            0.10,
        )

        translated.update(
            observation
            + offset,
            2.0,
            1.5,
            0.10,
        )

    assert translated.value == pytest.approx(
        original.value
        + offset
    )


def test_positive_scale_equivariance():

    sample = [
        3.0,
        -7.0,
        2.0,
        20.0,
        -1.0,
    ]

    factor = 7.0

    original = AdaptiveHuberState()
    scaled = AdaptiveHuberState()

    for observation in sample:

        original.update(
            observation,
            2.0,
            1.5,
            0.10,
        )

        scaled.update(
            factor
            * observation,
            factor
            * 2.0,
            1.5,
            0.10,
        )

    assert scaled.value == pytest.approx(
        factor
        * original.value
    )


def test_one_observation_delta_is_bounded():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        3.0,
        1.5,
        0.20,
    )

    previous = state.value

    state.update(
        1_000_000.0,
        3.0,
        1.5,
        0.20,
    )

    maximum_delta = (
        0.20
        * 1.5
        * 3.0
    )

    assert abs(
        state.value
        - previous
    ) <= pytest.approx(
        maximum_delta
    )

def test_one_observation_delta_is_bounded():

    state = AdaptiveHuberState()

    state.update(
        10.0,
        3.0,
        1.5,
        0.20,
    )

    previous = state.value

    state.update(
        1_000_000.0,
        3.0,
        1.5,
        0.20,
    )

    maximum_delta = (
        0.20
        * 1.5
        * 3.0
    )

    actual_delta = abs(
        state.value
        - previous
    )

    assert actual_delta <= (
        maximum_delta
        + 1e-12
    )
