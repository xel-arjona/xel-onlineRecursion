"""
Exponentially weighted covariance / correlation / regression reference.

For previous means mx and my and current paired observation (x, y):

    dx = x - mx
    dy = y - my

    mx += alpha * dx
    my += alpha * dy

    variance_x =
        (1-alpha) * (variance_x + alpha * dx * dx)

    variance_y =
        (1-alpha) * (variance_y + alpha * dy * dy)

    covariance =
        (1-alpha) * (covariance + alpha * dx * dy)

The state also tracks squared normalized-weight concentration:

    weight_square_sum =
        (1-alpha)^2 * previous_weight_square_sum + alpha^2

For deterministic or observation-independent recursive weights, raw weighted
central moments have expectation multiplied by:

    correction_denominator =
        1 - weight_square_sum

Finite-weight corrected covariance and variances therefore divide the raw
central moments by that denominator when it is strictly positive.

Correlation uses the raw central moments directly because their common
finite-weight attenuation factor cancels algebraically.

Simple linear regression of Y on X is also derived directly from the same
paired weighted population:

    beta_y_on_x =
        covariance / variance_x

    intercept_y_on_x =
        mean_y - beta_y_on_x * mean_x

    r_squared =
        correlation^2

The finite-weight correction factor cancels exactly from beta because
covariance and variance_x share identical weights.

Regression therefore introduces no additional recursive state.

The supplied alpha is clamped to [0, 1].

The first valid paired observation initializes both means directly and starts
with zero central moments and weight_square_sum = 1.

Missing x, y, or alpha preserves the complete retained state.
"""

from math import isnan, sqrt


def _is_missing(
    value: float | None,
) -> bool:
    """Returns True for None or NaN."""

    if value is None:
        return True

    return isnan(
        float(
            value
        )
    )


def _clamp(
    value: float,
    lower: float,
    upper: float,
) -> float:
    """Constrains value to the inclusive interval [lower, upper]."""

    return max(
        lower,
        min(
            upper,
            value,
        ),
    )


class AdaptiveCovarianceState:
    """Persistent exponentially weighted bivariate central-moment state."""

    def __init__(
        self,
    ) -> None:

        self.mean_x: float = float(
            "nan"
        )

        self.mean_y: float = float(
            "nan"
        )

        self.variance_x: float = 0.0
        self.variance_y: float = 0.0

        self.covariance: float = 0.0

        self.weight_square_sum: float = 1.0

    def reset(
        self,
    ) -> "AdaptiveCovarianceState":
        """Discards all retained bivariate moment state."""

        self.mean_x = float(
            "nan"
        )

        self.mean_y = float(
            "nan"
        )

        self.variance_x = 0.0
        self.variance_y = 0.0

        self.covariance = 0.0

        self.weight_square_sum = 1.0

        return self

    def update(
        self,
        x: float | None,
        y: float | None,
        alpha: float | None,
    ) -> "AdaptiveCovarianceState":
        """Updates the exponentially weighted paired-observation state."""

        valid_observation = (
            not _is_missing(
                x
            )
            and not _is_missing(
                y
            )
            and not _is_missing(
                alpha
            )
        )

        if not valid_observation:
            return self

        observation_x = float(
            x
        )

        observation_y = float(
            y
        )

        coefficient = _clamp(
            float(
                alpha
            ),
            0.0,
            1.0,
        )

        if (
            _is_missing(
                self.mean_x
            )
            or _is_missing(
                self.mean_y
            )
        ):

            self.mean_x = (
                observation_x
            )

            self.mean_y = (
                observation_y
            )

            self.variance_x = 0.0
            self.variance_y = 0.0

            self.covariance = 0.0

            self.weight_square_sum = 1.0

            return self

        dx = (
            observation_x
            - self.mean_x
        )

        dy = (
            observation_y
            - self.mean_y
        )

        retention = (
            1.0
            - coefficient
        )

        self.mean_x += (
            coefficient
            * dx
        )

        self.mean_y += (
            coefficient
            * dy
        )

        self.variance_x = (
            retention
            * (
                self.variance_x
                + coefficient
                * dx
                * dx
            )
        )

        self.variance_y = (
            retention
            * (
                self.variance_y
                + coefficient
                * dy
                * dy
            )
        )

        self.covariance = (
            retention
            * (
                self.covariance
                + coefficient
                * dx
                * dy
            )
        )

        self.weight_square_sum = (
            retention
            * retention
            * self.weight_square_sum
            + coefficient
            * coefficient
        )

        return self

    @property
    def correction_denominator(
        self,
    ) -> float:
        """Returns 1 minus squared normalized-weight concentration."""

        return (
            1.0
            - self.weight_square_sum
        )

    @property
    def corrected_covariance(
        self,
    ) -> float:
        """Returns finite-weight corrected covariance when defined."""

        denominator = (
            self.correction_denominator
        )

        if denominator <= 0.0:

            return float(
                "nan"
            )

        return (
            self.covariance
            / denominator
        )

    @property
    def corrected_variance_x(
        self,
    ) -> float:
        """Returns finite-weight corrected X variance when defined."""

        denominator = (
            self.correction_denominator
        )

        if denominator <= 0.0:

            return float(
                "nan"
            )

        return (
            self.variance_x
            / denominator
        )

    @property
    def corrected_variance_y(
        self,
    ) -> float:
        """Returns finite-weight corrected Y variance when defined."""

        denominator = (
            self.correction_denominator
        )

        if denominator <= 0.0:

            return float(
                "nan"
            )

        return (
            self.variance_y
            / denominator
        )

    @property
    def correlation(
        self,
    ) -> float:
        """Returns exponentially weighted Pearson correlation."""

        if (
            self.variance_x <= 0.0
            or self.variance_y <= 0.0
        ):

            return float(
                "nan"
            )

        denominator = sqrt(
            self.variance_x
            * self.variance_y
        )

        if denominator <= 0.0:

            return float(
                "nan"
            )

        return (
            self.covariance
            / denominator
        )

    @property
    def effective_sample_size(
        self,
    ) -> float:
        """Returns inverse normalized-weight concentration when defined."""

        if self.weight_square_sum <= 0.0:

            return float(
                "nan"
            )

        return (
            1.0
            / self.weight_square_sum
        )

    @property
    def beta_y_on_x(
        self,
    ) -> float:
        """Returns weighted simple-regression slope of Y on X."""

        if self.variance_x <= 0.0:

            return float(
                "nan"
            )

        return (
            self.covariance
            / self.variance_x
        )

    @property
    def intercept_y_on_x(
        self,
    ) -> float:
        """Returns weighted simple-regression intercept of Y on X."""

        beta = (
            self.beta_y_on_x
        )

        if (
            isnan(
                beta
            )
            or isnan(
                self.mean_x
            )
            or isnan(
                self.mean_y
            )
        ):

            return float(
                "nan"
            )

        return (
            self.mean_y
            - beta
            * self.mean_x
        )

    @property
    def r_squared(
        self,
    ) -> float:
        """Returns R-squared for simple weighted regression with intercept."""

        rho = (
            self.correlation
        )

        if isnan(
            rho
        ):

            return float(
                "nan"
            )

        return (
            rho
            * rho
        )
