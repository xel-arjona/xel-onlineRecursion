from math import isnan, sqrt

import numpy as np
import pytest

from src.adaptive_moments import AdaptiveMomentsState


DERIVED_PROPERTIES = (
    "variance",
    "correction_denominator",
    "corrected_variance",
    "sigma",
    "corrected_sigma",
    "skewness",
    "kurtosis",
    "excess_kurtosis",
    "effective_sample_size",
)


def retained_state(
    state: AdaptiveMomentsState,
) -> tuple[float, float, float, float, float]:
    return (
        state.mean,
        state.m2,
        state.m3,
        state.m4,
        state.weight_square_sum,
    )


def heavy_tail_oracle_update(
    state: tuple[float, float, float, float],
    input: float | None,
    alpha: float | None,
) -> tuple[float, float, float, float]:
    """Literal moment engine from onlineRecursion.pine lines 2848-2906."""

    old_mean, old_m2, old_m3, old_m4 = state

    if input is None or alpha is None:
        return state

    observation = float(input)
    feedback_factor = float(alpha)

    if isnan(observation) or isnan(feedback_factor):
        return state

    coefficient = max(
        0.0,
        min(
            1.0,
            feedback_factor,
        ),
    )

    # Historical HeavyTail does not initialize an empty state at alpha zero.
    if coefficient <= 0.0:
        return state

    retention = 1.0 - coefficient

    if isnan(old_mean) or coefficient >= 1.0:
        return (
            observation,
            0.0,
            0.0,
            0.0,
        )

    delta = observation - old_mean
    delta2 = delta * delta
    delta3 = delta2 * delta
    delta4 = delta2 * delta2

    return (
        old_mean + coefficient * delta,
        retention * old_m2 + coefficient * retention * delta2,
        (
            retention * old_m3
            - 3.0 * coefficient * retention * delta * old_m2
            + coefficient
            * retention
            * (retention - coefficient)
            * delta3
        ),
        (
            retention * old_m4
            - 4.0 * coefficient * retention * delta * old_m3
            + 6.0
            * coefficient
            * coefficient
            * retention
            * delta2
            * old_m2
            + coefficient
            * retention
            * (1.0 - 3.0 * coefficient * retention)
            * delta4
        ),
    )


def explicit_weighted_moments(
    values: list[float],
    alphas: list[float],
) -> tuple[float, float, float, float, float]:
    weights = np.array(
        [
            1.0,
        ],
        dtype=float,
    )

    for alpha in alphas[1:]:
        coefficient = max(
            0.0,
            min(
                1.0,
                alpha,
            ),
        )
        weights *= 1.0 - coefficient
        weights = np.append(
            weights,
            coefficient,
        )

    observations = np.asarray(
        values,
        dtype=float,
    )
    mean = float(
        np.sum(
            weights * observations
        )
    )
    centered = observations - mean

    return (
        mean,
        float(np.sum(weights * centered**2)),
        float(np.sum(weights * centered**3)),
        float(np.sum(weights * centered**4)),
        float(np.sum(weights * weights)),
    )


def assert_moment_parity(
    sample: list[tuple[float | None, float | None]],
) -> None:
    state = AdaptiveMomentsState()
    oracle = (
        float("nan"),
        0.0,
        0.0,
        0.0,
    )

    for input, alpha in sample:
        state.update(
            input,
            alpha,
        )
        oracle = heavy_tail_oracle_update(
            oracle,
            input,
            alpha,
        )

        assert state.mean == pytest.approx(oracle[0], abs=1e-14)
        assert state.m2 == pytest.approx(oracle[1], abs=1e-14)
        assert state.m3 == pytest.approx(oracle[2], abs=1e-14)
        assert state.m4 == pytest.approx(oracle[3], abs=1e-14)


def test_initial_state_is_empty_and_all_views_are_undefined():
    state = AdaptiveMomentsState()

    assert isnan(state.mean)
    assert state.m2 == 0.0
    assert state.m3 == 0.0
    assert state.m4 == 0.0
    assert state.weight_square_sum == 1.0

    for property_name in DERIVED_PROPERTIES:
        assert isnan(
            getattr(
                state,
                property_name,
            )
        )


def test_first_valid_observation_initializes_singleton():
    state = AdaptiveMomentsState().update(
        7.5,
        0.2,
    )

    assert retained_state(state) == (
        7.5,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def test_empty_alpha_zero_initializes_singleton_unlike_heavy_tail():
    state = AdaptiveMomentsState().update(
        4.0,
        0.0,
    )
    oracle = heavy_tail_oracle_update(
        (float("nan"), 0.0, 0.0, 0.0),
        4.0,
        0.0,
    )

    assert retained_state(state) == (
        4.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    assert isnan(oracle[0])


def test_negative_alpha_clamps_to_zero_during_initialization():
    state = AdaptiveMomentsState().update(
        -2.0,
        -10.0,
    )

    assert retained_state(state) == (
        -2.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def test_active_alpha_zero_freezes_complete_state_exactly():
    state = AdaptiveMomentsState()
    state.update(1.0, 0.4).update(5.0, 0.4)
    previous = retained_state(state)

    state.update(
        100.0,
        0.0,
    )

    assert retained_state(state) == previous


@pytest.mark.parametrize(
    "alpha",
    [
        1.0,
        5.0,
    ],
)
def test_alpha_one_or_above_replaces_with_singleton(alpha):
    state = AdaptiveMomentsState()
    state.update(1.0, 0.3).update(9.0, 0.3)

    state.update(
        -6.0,
        alpha,
    )

    assert retained_state(state) == (
        -6.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )


@pytest.mark.parametrize(
    ("input", "alpha"),
    [
        (None, 0.2),
        (float("nan"), 0.2),
        (4.0, None),
        (4.0, float("nan")),
    ],
)
def test_missing_observation_or_alpha_preserves_active_state(input, alpha):
    state = AdaptiveMomentsState()
    state.update(1.0, 0.2).update(3.0, 0.2)
    previous = retained_state(state)

    state.update(
        input,
        alpha,
    )

    assert retained_state(state) == previous


@pytest.mark.parametrize(
    ("input", "alpha"),
    [
        (None, 0.2),
        (float("nan"), 0.2),
        (4.0, None),
        (4.0, float("nan")),
    ],
)
def test_missing_observation_or_alpha_preserves_empty_state(input, alpha):
    state = AdaptiveMomentsState()

    state.update(
        input,
        alpha,
    )

    assert isnan(state.mean)
    assert state.m2 == 0.0
    assert state.m3 == 0.0
    assert state.m4 == 0.0
    assert state.weight_square_sum == 1.0


def test_reset_restores_empty_representation_and_undefined_views():
    state = AdaptiveMomentsState()
    state.update(1.0, 0.2).update(3.0, 0.2)

    returned = state.reset()

    assert returned is state
    assert isnan(state.mean)
    assert state.m2 == 0.0
    assert state.m3 == 0.0
    assert state.m4 == 0.0
    assert state.weight_square_sum == 1.0
    assert all(
        isnan(getattr(state, name))
        for name in DERIVED_PROPERTIES
    )


def test_reset_then_alpha_zero_initializes_new_singleton():
    state = AdaptiveMomentsState()
    state.update(1.0, 0.2).update(3.0, 0.2).reset().update(11.0, 0.0)

    assert retained_state(state) == (
        11.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def test_fixed_alpha_matches_canonical_heavy_tail_recurrence():
    assert_moment_parity(
        [
            (2.0, 0.25),
            (-1.0, 0.25),
            (7.0, 0.25),
            (3.0, 0.25),
            (12.0, 0.25),
        ]
    )


def test_variable_alpha_and_boundaries_match_heavy_tail_recurrence():
    assert_moment_parity(
        [
            (2.0, 0.2),
            (-1.0, 0.7),
            (20.0, 0.0),
            (None, 0.3),
            (5.0, None),
            (float("nan"), 0.4),
            (8.0, -2.0),
            (4.0, 0.1),
            (10.0, 1.0),
            (-3.0, 2.0),
            (6.0, 0.35),
        ]
    )


def test_two_point_half_alpha_hand_calculation_and_derived_views():
    state = AdaptiveMomentsState()
    state.update(0.0, 0.5).update(2.0, 0.5)

    assert retained_state(state) == pytest.approx(
        (
            1.0,
            1.0,
            0.0,
            1.0,
            0.5,
        )
    )
    assert state.variance == pytest.approx(1.0)
    assert state.sigma == pytest.approx(1.0)
    assert state.correction_denominator == pytest.approx(0.5)
    assert state.corrected_variance == pytest.approx(2.0)
    assert state.corrected_sigma == pytest.approx(sqrt(2.0))
    assert state.skewness == pytest.approx(0.0)
    assert state.kurtosis == pytest.approx(1.0)
    assert state.excess_kurtosis == pytest.approx(-2.0)
    assert state.effective_sample_size == pytest.approx(2.0)


def test_singleton_derived_views():
    state = AdaptiveMomentsState().update(
        3.0,
        0.0,
    )

    assert state.variance == 0.0
    assert state.sigma == 0.0
    assert state.effective_sample_size == 1.0
    assert state.correction_denominator == 0.0
    assert isnan(state.corrected_variance)
    assert isnan(state.corrected_sigma)
    assert isnan(state.skewness)
    assert isnan(state.kurtosis)
    assert isnan(state.excess_kurtosis)


def test_variable_alpha_matches_explicit_normalized_weights():
    values = [
        2.0,
        5.0,
        -1.0,
        8.0,
        4.0,
        10.0,
    ]
    alphas = [
        0.0,
        0.10,
        0.25,
        0.00,
        0.70,
        0.80,
    ]
    state = AdaptiveMomentsState()

    for value, alpha in zip(values, alphas):
        state.update(value, alpha)

    assert retained_state(state) == pytest.approx(
        explicit_weighted_moments(values, alphas),
        abs=1e-12,
    )


def test_translation_invariance():
    original = AdaptiveMomentsState()
    translated = AdaptiveMomentsState()
    shift = 1_000.0

    for value, alpha in zip(
        [1.0, -4.0, 7.0, 2.0, 12.0],
        [0.2, 0.4, 0.1, 0.7, 0.3],
    ):
        original.update(value, alpha)
        translated.update(value + shift, alpha)

    assert translated.mean == pytest.approx(original.mean + shift)
    assert translated.m2 == pytest.approx(original.m2)
    assert translated.m3 == pytest.approx(original.m3)
    assert translated.m4 == pytest.approx(original.m4)
    assert translated.weight_square_sum == pytest.approx(
        original.weight_square_sum
    )


def test_reflection_symmetry():
    original = AdaptiveMomentsState()
    reflected = AdaptiveMomentsState()

    for value, alpha in zip(
        [1.0, -4.0, 7.0, 2.0, 12.0],
        [0.2, 0.4, 0.1, 0.7, 0.3],
    ):
        original.update(value, alpha)
        reflected.update(-value, alpha)

    assert reflected.mean == pytest.approx(-original.mean)
    assert reflected.m2 == pytest.approx(original.m2)
    assert reflected.m3 == pytest.approx(-original.m3)
    assert reflected.m4 == pytest.approx(original.m4)
    assert reflected.skewness == pytest.approx(-original.skewness)
    assert reflected.kurtosis == pytest.approx(original.kurtosis)


def test_scale_transformation():
    original = AdaptiveMomentsState()
    scaled = AdaptiveMomentsState()
    factor = -3.0

    for value, alpha in zip(
        [1.0, -4.0, 7.0, 2.0, 12.0],
        [0.2, 0.4, 0.1, 0.7, 0.3],
    ):
        original.update(value, alpha)
        scaled.update(factor * value, alpha)

    assert scaled.mean == pytest.approx(factor * original.mean)
    assert scaled.m2 == pytest.approx(factor**2 * original.m2)
    assert scaled.m3 == pytest.approx(factor**3 * original.m3)
    assert scaled.m4 == pytest.approx(factor**4 * original.m4)
    assert scaled.sigma == pytest.approx(abs(factor) * original.sigma)
    assert scaled.skewness == pytest.approx(-original.skewness)
    assert scaled.kurtosis == pytest.approx(original.kurtosis)


def test_constant_sequence_remains_zero_variance_population():
    state = AdaptiveMomentsState()

    for alpha in [0.1, 0.2, 0.0, 0.7, 1.0, 0.4]:
        state.update(6.0, alpha)

    assert state.mean == 6.0
    assert state.m2 == 0.0
    assert state.m3 == 0.0
    assert state.m4 == 0.0
    assert state.variance == 0.0
    assert state.sigma == 0.0
    assert isnan(state.skewness)
    assert isnan(state.kurtosis)


def test_weight_square_sum_exact_recurrence():
    state = AdaptiveMomentsState().update(
        1.0,
        0.8,
    )
    expected = 1.0

    for index, alpha in enumerate([0.1, 0.35, 0.0, 0.8], start=2):
        coefficient = max(0.0, min(1.0, alpha))
        expected = (
            (1.0 - coefficient) ** 2 * expected
            + coefficient**2
        )
        state.update(float(index), alpha)
        assert state.weight_square_sum == pytest.approx(expected)


def test_constant_alpha_effective_sample_size_limit():
    state = AdaptiveMomentsState()
    alpha = 0.1
    state.update(0.0, alpha)

    for index in range(1000):
        state.update(float(index), alpha)

    assert state.weight_square_sum == pytest.approx(
        alpha / (2.0 - alpha),
        abs=1e-12,
    )
    assert state.effective_sample_size == pytest.approx(
        (2.0 - alpha) / alpha,
        abs=1e-10,
    )
