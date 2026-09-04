from math import isnan, nan

import pytest

from src.p2 import P2QuantileState, round_half_to_even


def test_round_half_to_even():
    assert round_half_to_even(0.5) == 0
    assert round_half_to_even(1.5) == 2
    assert round_half_to_even(2.5) == 2
    assert round_half_to_even(3.5) == 4

    assert round_half_to_even(1.4) == 1
    assert round_half_to_even(1.6) == 2


def test_bootstrap_is_undefined_before_five_observations():
    state = P2QuantileState()

    for observation in [1.0, 2.0, 3.0, 4.0]:
        state.update(
            observation,
            0.5,
        )

        assert isnan(state.value)

    assert state.count == 4


def test_fifth_observation_initializes_median():
    state = P2QuantileState()

    for observation in [1.0, 2.0, 3.0, 4.0, 5.0]:
        state.update(
            observation,
            0.5,
        )

    assert state.count == 5
    assert state.value == pytest.approx(
        3.0,
        abs=1e-12,
    )


def test_ascending_median_sequence():
    state = P2QuantileState()

    expected = {
        4: 3.0,
        5: 3.0,
        6: 3.0,
        7: 4.0,
        8: 5.0,
        9: 5.0,
        10: 6.0,
        11: 6.0,
        12: 7.0,
        13: 7.0,
        14: 8.0,
        15: 8.0,
        16: 9.0,
        17: 9.0,
        18: 10.0,
        19: 10.0,
    }

    for index in range(20):
        state.update(
            float(index + 1),
            0.5,
        )

        if index < 4:
            assert isnan(state.value)

        else:
            assert state.value == pytest.approx(
                expected[index],
                abs=1e-12,
            )


def test_lower_tail_ascending_sequence():
    state = P2QuantileState()

    expected = {
        4: 1.0,
        9: 1.0,
        19: 1.0,
        24: 2.0,
        39: 2.0,
        49: 3.0,
        74: 4.0,
        99: 5.0,
    }

    for index in range(100):
        state.update(
            float(index + 1),
            0.05,
        )

        if index in expected:
            assert state.value == pytest.approx(
                expected[index],
                abs=1e-12,
            )


def test_upper_tail_ascending_sequence():
    state = P2QuantileState()

    expected = {
        4: 5.0,
        9: 7.0,
        19: 17.0,
        24: 22.0,
        39: 37.0,
        49: 46.0,
        74: 71.0,
        99: 95.0,
    }

    for index in range(100):
        state.update(
            float(index + 1),
            0.95,
        )

        if index in expected:
            assert state.value == pytest.approx(
                expected[index],
                abs=1e-12,
            )


def test_descending_median_sequence():
    state = P2QuantileState()

    expected = {
        4: 98.0,
        9: 96.0,
        19: 91.0,
        24: 88.0,
        39: 81.0,
        49: 76.0,
        74: 63.0,
        99: 51.0,
    }

    for index in range(100):
        state.update(
            100.0 - float(index),
            0.5,
        )

        if index in expected:
            assert state.value == pytest.approx(
                expected[index],
                abs=1e-12,
            )


def test_missing_observation_preserves_state():
    state = P2QuantileState()

    for observation in [1.0, 2.0, 3.0, 4.0, 5.0]:
        state.update(
            observation,
            0.5,
        )

    previous_count = state.count
    previous_value = state.value
    previous_probability = state.probability

    previous_heights = state.marker_heights.copy()
    previous_positions = state.marker_positions.copy()
    previous_desired = state.desired_positions.copy()

    state.update(
        nan,
        0.5,
    )

    assert state.count == previous_count
    assert state.value == previous_value
    assert state.probability == previous_probability

    assert state.marker_heights == previous_heights
    assert state.marker_positions == previous_positions
    assert state.desired_positions == previous_desired


def test_reset_discards_population_and_reuses_storage():
    state = P2QuantileState()

    for observation in [1.0, 2.0, 3.0, 4.0, 5.0]:
        state.update(
            observation,
            0.5,
        )

    heights_id = id(state.marker_heights)
    positions_id = id(state.marker_positions)
    desired_id = id(state.desired_positions)

    state.reset()

    assert state.count == 0
    assert isnan(state.value)
    assert isnan(state.probability)

    assert id(state.marker_heights) == heights_id
    assert id(state.marker_positions) == positions_id
    assert id(state.desired_positions) == desired_id

    assert all(
        isnan(value)
        for value in state.marker_heights
    )

    assert state.marker_positions == [
        None,
        None,
        None,
        None,
        None,
    ]

    assert all(
        isnan(value)
        for value in state.desired_positions
    )


def test_reset_allows_new_probability_and_rebootstrap():
    state = P2QuantileState()

    for observation in [1.0, 2.0, 3.0, 4.0, 5.0]:
        state.update(
            observation,
            0.5,
        )

    state.reset()

    for observation in [101.0, 102.0, 103.0, 104.0, 105.0]:
        state.update(
            observation,
            0.95,
        )

    assert state.count == 5
    assert state.probability == 0.95

    # Adaptive initialization at p = 0.95 selects the upper
    # bootstrap extreme as q2.
    assert state.value == pytest.approx(
        105.0,
        abs=1e-12,
    )


def test_probability_cannot_change_inside_population():
    state = P2QuantileState()

    state.update(
        1.0,
        0.5,
    )

    with pytest.raises(
        ValueError,
        match="cannot change",
    ):
        state.update(
            2.0,
            0.95,
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
    state = P2QuantileState()

    with pytest.raises(
        ValueError,
        match="strictly between zero and one",
    ):
        state.update(
            1.0,
            probability,
        )
