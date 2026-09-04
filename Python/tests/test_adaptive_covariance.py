from math import isnan, sqrt

import numpy as np
import pytest

from src.adaptive_covariance import AdaptiveCovarianceState


def explicit_weighted_moments(
    x_values: list[float],
    y_values: list[float],
    alphas: list[float],
) -> dict[str, float]:

    weights = np.array(
        [
            1.0
        ],
        dtype=float,
    )

    for alpha in alphas[
        1:
    ]:

        coefficient = max(
            0.0,
            min(
                1.0,
                alpha,
            ),
        )

        weights *= (
            1.0
            - coefficient
        )

        weights = np.append(
            weights,
            coefficient,
        )

    x_array = np.asarray(
        x_values,
        dtype=float,
    )

    y_array = np.asarray(
        y_values,
        dtype=float,
    )

    mean_x = float(
        np.sum(
            weights
            * x_array
        )
    )

    mean_y = float(
        np.sum(
            weights
            * y_array
        )
    )

    centered_x = (
        x_array
        - mean_x
    )

    centered_y = (
        y_array
        - mean_y
    )

    variance_x = float(
        np.sum(
            weights
            * centered_x
            * centered_x
        )
    )

    variance_y = float(
        np.sum(
            weights
            * centered_y
            * centered_y
        )
    )

    covariance = float(
        np.sum(
            weights
            * centered_x
            * centered_y
        )
    )

    weight_square_sum = float(
        np.sum(
            weights
            * weights
        )
    )

    return {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "variance_x": variance_x,
        "variance_y": variance_y,
        "covariance": covariance,
        "weight_square_sum": (
            weight_square_sum
        ),
    }


def test_first_valid_pair_initializes_directly():

    state = AdaptiveCovarianceState()

    state.update(
        2.0,
        -3.0,
        0.20,
    )

    assert state.mean_x == pytest.approx(
        2.0
    )

    assert state.mean_y == pytest.approx(
        -3.0
    )

    assert state.variance_x == pytest.approx(
        0.0
    )

    assert state.variance_y == pytest.approx(
        0.0
    )

    assert state.covariance == pytest.approx(
        0.0
    )

    assert state.weight_square_sum == pytest.approx(
        1.0
    )


def test_alpha_zero_initializes_empty_state():

    state = AdaptiveCovarianceState()

    state.update(
        2.0,
        3.0,
        0.0,
    )

    assert state.mean_x == pytest.approx(
        2.0
    )

    assert state.mean_y == pytest.approx(
        3.0
    )

    assert state.weight_square_sum == pytest.approx(
        1.0
    )


def test_alpha_above_one_initializes_directly():

    state = AdaptiveCovarianceState()

    state.update(
        2.0,
        3.0,
        5.0,
    )

    assert state.mean_x == pytest.approx(
        2.0
    )

    assert state.mean_y == pytest.approx(
        3.0
    )


def test_alpha_zero_freezes_active_state():

    state = AdaptiveCovarianceState()

    state.update(
        1.0,
        2.0,
        0.5,
    )

    state.update(
        3.0,
        6.0,
        0.5,
    )

    previous = (
        state.mean_x,
        state.mean_y,
        state.variance_x,
        state.variance_y,
        state.covariance,
        state.weight_square_sum,
    )

    state.update(
        100.0,
        -100.0,
        0.0,
    )

    assert (
        state.mean_x,
        state.mean_y,
        state.variance_x,
        state.variance_y,
        state.covariance,
        state.weight_square_sum,
    ) == previous


def test_negative_alpha_clamps_to_zero():

    state = AdaptiveCovarianceState()

    state.update(
        1.0,
        2.0,
        0.5,
    )

    state.update(
        3.0,
        6.0,
        0.5,
    )

    previous = (
        state.covariance
    )

    state.update(
        100.0,
        -100.0,
        -1.0,
    )

    assert state.covariance == pytest.approx(
        previous
    )


def test_alpha_above_one_clamps_to_complete_replacement():

    state = AdaptiveCovarianceState()

    state.update(
        1.0,
        2.0,
        0.5,
    )

    state.update(
        3.0,
        6.0,
        0.5,
    )

    state.update(
        10.0,
        -4.0,
        2.0,
    )

    assert state.mean_x == pytest.approx(
        10.0
    )

    assert state.mean_y == pytest.approx(
        -4.0
    )

    assert state.variance_x == pytest.approx(
        0.0
    )

    assert state.variance_y == pytest.approx(
        0.0
    )

    assert state.covariance == pytest.approx(
        0.0
    )

    assert state.weight_square_sum == pytest.approx(
        1.0
    )


def test_two_point_half_alpha_hand_calculation():

    state = AdaptiveCovarianceState()

    state.update(
        0.0,
        0.0,
        0.5,
    )

    state.update(
        2.0,
        4.0,
        0.5,
    )

    assert state.mean_x == pytest.approx(
        1.0
    )

    assert state.mean_y == pytest.approx(
        2.0
    )

    assert state.variance_x == pytest.approx(
        1.0
    )

    assert state.variance_y == pytest.approx(
        4.0
    )

    assert state.covariance == pytest.approx(
        2.0
    )

    assert state.weight_square_sum == pytest.approx(
        0.5
    )


def test_exact_variable_alpha_weight_identity():

    x_values = [
        2.0,
        5.0,
        -1.0,
        8.0,
        4.0,
        10.0,
    ]

    y_values = [
        -3.0,
        7.0,
        2.0,
        -5.0,
        6.0,
        11.0,
    ]

    alphas = [
        0.0,
        0.10,
        0.25,
        0.00,
        0.70,
        0.80,
    ]

    state = AdaptiveCovarianceState()

    for (
        x,
        y,
        alpha,
    ) in zip(
        x_values,
        y_values,
        alphas,
        strict=True,
    ):

        state.update(
            x,
            y,
            alpha,
        )

    expected = explicit_weighted_moments(
        x_values,
        y_values,
        alphas,
    )

    assert state.mean_x == pytest.approx(
        expected[
            "mean_x"
        ],
        abs=1e-12,
    )

    assert state.mean_y == pytest.approx(
        expected[
            "mean_y"
        ],
        abs=1e-12,
    )

    assert state.variance_x == pytest.approx(
        expected[
            "variance_x"
        ],
        abs=1e-12,
    )

    assert state.variance_y == pytest.approx(
        expected[
            "variance_y"
        ],
        abs=1e-12,
    )

    assert state.covariance == pytest.approx(
        expected[
            "covariance"
        ],
        abs=1e-12,
    )

    assert state.weight_square_sum == pytest.approx(
        expected[
            "weight_square_sum"
        ],
        abs=1e-12,
    )


def test_two_point_corrected_covariance():

    state = AdaptiveCovarianceState()

    state.update(
        0.0,
        0.0,
        0.5,
    )

    state.update(
        2.0,
        4.0,
        0.5,
    )

    assert state.corrected_covariance == pytest.approx(
        4.0
    )


def test_two_point_corrected_variances():

    state = AdaptiveCovarianceState()

    state.update(
        0.0,
        0.0,
        0.5,
    )

    state.update(
        2.0,
        4.0,
        0.5,
    )

    assert state.corrected_variance_x == pytest.approx(
        2.0
    )

    assert state.corrected_variance_y == pytest.approx(
        8.0
    )


def test_two_point_effective_sample_size():

    state = AdaptiveCovarianceState()

    state.update(
        0.0,
        0.0,
        0.5,
    )

    state.update(
        2.0,
        4.0,
        0.5,
    )

    assert state.effective_sample_size == pytest.approx(
        2.0
    )


def test_perfect_positive_correlation():

    state = AdaptiveCovarianceState()

    for x in [
        1.0,
        2.0,
        3.0,
        5.0,
        8.0,
    ]:

        state.update(
            x,
            3.0 * x + 7.0,
            0.2,
        )

    assert state.correlation == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_perfect_negative_correlation():

    state = AdaptiveCovarianceState()

    for x in [
        1.0,
        2.0,
        3.0,
        5.0,
        8.0,
    ]:

        state.update(
            x,
            -2.0 * x + 4.0,
            0.2,
        )

    assert state.correlation == pytest.approx(
        -1.0,
        abs=1e-12,
    )


def test_correction_factor_cancels_from_correlation():

    state = AdaptiveCovarianceState()

    for (
        x,
        y,
    ) in [
        (
            1.0,
            5.0,
        ),
        (
            2.0,
            4.0,
        ),
        (
            4.0,
            8.0,
        ),
        (
            7.0,
            6.0,
        ),
    ]:

        state.update(
            x,
            y,
            0.25,
        )

    corrected_rho = (
        state.corrected_covariance
        / sqrt(
            state.corrected_variance_x
            * state.corrected_variance_y
        )
    )

    assert state.correlation == pytest.approx(
        corrected_rho,
        abs=1e-12,
    )


def test_missing_x_preserves_state():

    state = AdaptiveCovarianceState()

    state.update(
        1.0,
        2.0,
        0.2,
    )

    state.update(
        3.0,
        4.0,
        0.2,
    )

    previous = vars(
        state
    ).copy()

    state.update(
        None,
        10.0,
        0.2,
    )

    assert vars(
        state
    ) == previous


def test_missing_y_preserves_state():

    state = AdaptiveCovarianceState()

    state.update(
        1.0,
        2.0,
        0.2,
    )

    state.update(
        3.0,
        4.0,
        0.2,
    )

    previous = vars(
        state
    ).copy()

    state.update(
        10.0,
        None,
        0.2,
    )

    assert vars(
        state
    ) == previous


def test_missing_alpha_preserves_state():

    state = AdaptiveCovarianceState()

    state.update(
        1.0,
        2.0,
        0.2,
    )

    state.update(
        3.0,
        4.0,
        0.2,
    )

    previous = vars(
        state
    ).copy()

    state.update(
        10.0,
        20.0,
        None,
    )

    assert vars(
        state
    ) == previous


@pytest.mark.parametrize(
    (
        "missing_x",
        "missing_y",
        "missing_alpha",
    ),
    [
        (
            float(
                "nan"
            ),
            1.0,
            0.2,
        ),
        (
            1.0,
            float(
                "nan"
            ),
            0.2,
        ),
        (
            1.0,
            2.0,
            float(
                "nan"
            ),
        ),
    ],
)
def test_nan_inputs_preserve_empty_state(
    missing_x,
    missing_y,
    missing_alpha,
):

    state = AdaptiveCovarianceState()

    state.update(
        missing_x,
        missing_y,
        missing_alpha,
    )

    assert isnan(
        state.mean_x
    )

    assert isnan(
        state.mean_y
    )

    assert state.variance_x == pytest.approx(
        0.0
    )

    assert state.variance_y == pytest.approx(
        0.0
    )

    assert state.covariance == pytest.approx(
        0.0
    )

    assert state.weight_square_sum == pytest.approx(
        1.0
    )


def test_reset_clears_complete_state():

    state = AdaptiveCovarianceState()

    state.update(
        1.0,
        2.0,
        0.2,
    )

    state.update(
        3.0,
        4.0,
        0.2,
    )

    state.reset()

    assert isnan(
        state.mean_x
    )

    assert isnan(
        state.mean_y
    )

    assert state.variance_x == pytest.approx(
        0.0
    )

    assert state.variance_y == pytest.approx(
        0.0
    )

    assert state.covariance == pytest.approx(
        0.0
    )

    assert state.weight_square_sum == pytest.approx(
        1.0
    )


def test_reset_allows_fresh_initialization():

    state = AdaptiveCovarianceState()

    state.update(
        1.0,
        2.0,
        0.2,
    )

    state.update(
        3.0,
        4.0,
        0.2,
    )

    state.reset()

    state.update(
        100.0,
        -50.0,
        0.7,
    )

    assert state.mean_x == pytest.approx(
        100.0
    )

    assert state.mean_y == pytest.approx(
        -50.0
    )

    assert state.covariance == pytest.approx(
        0.0
    )


def test_translation_invariance_of_central_moments():

    original = AdaptiveCovarianceState()
    translated = AdaptiveCovarianceState()

    sample = [
        (
            1.0,
            5.0,
        ),
        (
            2.0,
            -4.0,
        ),
        (
            7.0,
            8.0,
        ),
        (
            -3.0,
            1.0,
        ),
    ]

    for (
        x,
        y,
    ) in sample:

        original.update(
            x,
            y,
            0.2,
        )

        translated.update(
            x + 100.0,
            y - 50.0,
            0.2,
        )

    assert translated.mean_x == pytest.approx(
        original.mean_x
        + 100.0
    )

    assert translated.mean_y == pytest.approx(
        original.mean_y
        - 50.0
    )

    assert translated.variance_x == pytest.approx(
        original.variance_x
    )

    assert translated.variance_y == pytest.approx(
        original.variance_y
    )

    assert translated.covariance == pytest.approx(
        original.covariance
    )

    assert translated.correlation == pytest.approx(
        original.correlation
    )


def test_positive_scale_equivariance():

    original = AdaptiveCovarianceState()
    scaled = AdaptiveCovarianceState()

    factor_x = 3.0
    factor_y = 5.0

    sample = [
        (
            1.0,
            5.0,
        ),
        (
            2.0,
            -4.0,
        ),
        (
            7.0,
            8.0,
        ),
        (
            -3.0,
            1.0,
        ),
    ]

    for (
        x,
        y,
    ) in sample:

        original.update(
            x,
            y,
            0.2,
        )

        scaled.update(
            factor_x * x,
            factor_y * y,
            0.2,
        )

    assert scaled.variance_x == pytest.approx(
        factor_x
        * factor_x
        * original.variance_x
    )

    assert scaled.variance_y == pytest.approx(
        factor_y
        * factor_y
        * original.variance_y
    )

    assert scaled.covariance == pytest.approx(
        factor_x
        * factor_y
        * original.covariance
    )

    assert scaled.correlation == pytest.approx(
        original.correlation
    )


def test_reflecting_x_flips_covariance_and_correlation():

    original = AdaptiveCovarianceState()
    reflected = AdaptiveCovarianceState()

    sample = [
        (
            1.0,
            5.0,
        ),
        (
            2.0,
            -4.0,
        ),
        (
            7.0,
            8.0,
        ),
        (
            -3.0,
            1.0,
        ),
    ]

    for (
        x,
        y,
    ) in sample:

        original.update(
            x,
            y,
            0.2,
        )

        reflected.update(
            -x,
            y,
            0.2,
        )

    assert reflected.variance_x == pytest.approx(
        original.variance_x
    )

    assert reflected.variance_y == pytest.approx(
        original.variance_y
    )

    assert reflected.covariance == pytest.approx(
        -original.covariance
    )

    assert reflected.correlation == pytest.approx(
        -original.correlation
    )


def test_exchange_symmetry():

    xy = AdaptiveCovarianceState()
    yx = AdaptiveCovarianceState()

    sample = [
        (
            1.0,
            5.0,
        ),
        (
            2.0,
            -4.0,
        ),
        (
            7.0,
            8.0,
        ),
        (
            -3.0,
            1.0,
        ),
    ]

    for (
        x,
        y,
    ) in sample:

        xy.update(
            x,
            y,
            0.2,
        )

        yx.update(
            y,
            x,
            0.2,
        )

    assert yx.mean_x == pytest.approx(
        xy.mean_y
    )

    assert yx.mean_y == pytest.approx(
        xy.mean_x
    )

    assert yx.variance_x == pytest.approx(
        xy.variance_y
    )

    assert yx.variance_y == pytest.approx(
        xy.variance_x
    )

    assert yx.covariance == pytest.approx(
        xy.covariance
    )

    assert yx.correlation == pytest.approx(
        xy.correlation
    )


def test_zero_variance_makes_correlation_undefined():

    state = AdaptiveCovarianceState()

    for x in [
        1.0,
        2.0,
        3.0,
        4.0,
    ]:

        state.update(
            x,
            7.0,
            0.2,
        )

    assert isnan(
        state.correlation
    )


def test_single_observation_corrected_moments_are_undefined():

    state = AdaptiveCovarianceState()

    state.update(
        1.0,
        2.0,
        0.2,
    )

    assert isnan(
        state.corrected_covariance
    )

    assert isnan(
        state.corrected_variance_x
    )

    assert isnan(
        state.corrected_variance_y
    )

    assert isnan(
        state.correlation
    )


def test_alpha_one_returns_to_single_observation_population():

    state = AdaptiveCovarianceState()

    state.update(
        1.0,
        2.0,
        0.2,
    )

    state.update(
        3.0,
        5.0,
        0.2,
    )

    state.update(
        10.0,
        20.0,
        1.0,
    )

    assert state.mean_x == pytest.approx(
        10.0
    )

    assert state.mean_y == pytest.approx(
        20.0
    )

    assert state.weight_square_sum == pytest.approx(
        1.0
    )

    assert isnan(
        state.corrected_covariance
    )

    assert isnan(
        state.correlation
    )


def test_weight_square_sum_stays_in_unit_interval():

    state = AdaptiveCovarianceState()

    sample = [
        (
            1.0,
            2.0,
            0.01,
        ),
        (
            3.0,
            4.0,
            0.20,
        ),
        (
            5.0,
            -1.0,
            0.70,
        ),
        (
            8.0,
            9.0,
            -1.0,
        ),
        (
            10.0,
            20.0,
            2.0,
        ),
    ]

    for (
        x,
        y,
        alpha,
    ) in sample:

        state.update(
            x,
            y,
            alpha,
        )

        assert (
            0.0
            <= state.weight_square_sum
            <= 1.0
        )


def test_cauchy_schwarz_bound_for_raw_moments():

    state = AdaptiveCovarianceState()

    sample = [
        (
            1.0,
            5.0,
        ),
        (
            2.0,
            -4.0,
        ),
        (
            7.0,
            8.0,
        ),
        (
            -3.0,
            1.0,
        ),
        (
            9.0,
            2.0,
        ),
    ]

    for (
        x,
        y,
    ) in sample:

        state.update(
            x,
            y,
            0.2,
        )

    bound = sqrt(
        state.variance_x
        * state.variance_y
    )

    assert abs(
        state.covariance
    ) <= (
        bound
        + 1e-12
    )


def test_constant_alpha_asymptotic_weight_square_sum():

    state = AdaptiveCovarianceState()

    alpha = 0.10

    state.update(
        0.0,
        0.0,
        alpha,
    )

    for index in range(
        1000
    ):

        state.update(
            float(
                index
            ),
            float(
                -index
            ),
            alpha,
        )

    expected = (
        alpha
        / (
            2.0
            - alpha
        )
    )

    assert state.weight_square_sum == pytest.approx(
        expected,
        abs=1e-12,
    )

    assert state.effective_sample_size == pytest.approx(
        (
            2.0
            - alpha
        )
        / alpha,
        abs=1e-10,
    )


# ============================================================================
# DERIVED SIMPLE-REGRESSION VIEWS
# ============================================================================


def test_exact_regression_line_and_correction_cancellation():

    state = AdaptiveCovarianceState()

    state.update(
        0.0,
        1.0,
        0.5,
    )

    state.update(
        2.0,
        5.0,
        0.5,
    )

    assert state.beta_y_on_x == pytest.approx(
        2.0,
        abs=1e-12,
    )

    assert state.intercept_y_on_x == pytest.approx(
        1.0,
        abs=1e-12,
    )

    assert state.r_squared == pytest.approx(
        1.0,
        abs=1e-12,
    )

    corrected_beta = (
        state.corrected_covariance
        / state.corrected_variance_x
    )

    assert state.beta_y_on_x == pytest.approx(
        corrected_beta,
        abs=1e-12,
    )


def test_regression_affine_properties():

    sample = [
        (
            1.0,
            4.0,
        ),
        (
            2.0,
            7.0,
        ),
        (
            5.0,
            13.0,
        ),
        (
            -2.0,
            -1.0,
        ),
    ]

    original = AdaptiveCovarianceState()
    translated_y = AdaptiveCovarianceState()
    translated_x = AdaptiveCovarianceState()
    scaled = AdaptiveCovarianceState()

    offset_y = 100.0
    offset_x = 50.0

    factor_x = 3.0
    factor_y = 5.0

    for (
        x,
        y,
    ) in sample:

        original.update(
            x,
            y,
            0.20,
        )

        translated_y.update(
            x,
            y + offset_y,
            0.20,
        )

        translated_x.update(
            x + offset_x,
            y,
            0.20,
        )

        scaled.update(
            factor_x * x,
            factor_y * y,
            0.20,
        )

    assert translated_y.beta_y_on_x == pytest.approx(
        original.beta_y_on_x,
        abs=1e-12,
    )

    assert translated_y.intercept_y_on_x == pytest.approx(
        original.intercept_y_on_x
        + offset_y,
        abs=1e-12,
    )

    assert translated_x.beta_y_on_x == pytest.approx(
        original.beta_y_on_x,
        abs=1e-12,
    )

    assert translated_x.intercept_y_on_x == pytest.approx(
        original.intercept_y_on_x
        - original.beta_y_on_x
        * offset_x,
        abs=1e-12,
    )

    assert scaled.beta_y_on_x == pytest.approx(
        factor_y
        / factor_x
        * original.beta_y_on_x,
        abs=1e-12,
    )

    assert scaled.intercept_y_on_x == pytest.approx(
        factor_y
        * original.intercept_y_on_x,
        abs=1e-12,
    )

    assert scaled.r_squared == pytest.approx(
        original.r_squared,
        abs=1e-12,
    )


def test_regression_undefined_when_x_variance_is_zero():

    state = AdaptiveCovarianceState()

    for y in [
        1.0,
        5.0,
        -2.0,
        8.0,
    ]:

        state.update(
            7.0,
            y,
            0.2,
        )

    assert isnan(
        state.beta_y_on_x
    )

    assert isnan(
        state.intercept_y_on_x
    )

    assert isnan(
        state.r_squared
    )


def test_r_squared_is_correlation_squared():

    state = AdaptiveCovarianceState()

    sample = [
        (
            1.0,
            5.0,
        ),
        (
            2.0,
            -4.0,
        ),
        (
            7.0,
            8.0,
        ),
        (
            -3.0,
            1.0,
        ),
        (
            9.0,
            2.0,
        ),
    ]

    for (
        x,
        y,
    ) in sample:

        state.update(
            x,
            y,
            0.2,
        )

    assert state.r_squared == pytest.approx(
        state.correlation
        * state.correlation,
        abs=1e-12,
    )
