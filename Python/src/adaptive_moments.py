"""Exponentially weighted univariate central-moment reference.

The retained state represents one normalized recursively weighted population.
The first valid observation creates a singleton population independently of
the supplied (clamped) transition coefficient.  Once active, the state uses
the accepted HeavyTail central-moment recurrence from the canonical Pine
implementation.
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


class AdaptiveMomentsState:
    """Persistent normalized exponentially weighted moment population."""

    def __init__(
        self,
    ) -> None:

        self.mean: float = float(
            "nan"
        )

        self.m2: float = 0.0
        self.m3: float = 0.0
        self.m4: float = 0.0

        self.weight_square_sum: float = 1.0

    def reset(
        self,
    ) -> "AdaptiveMomentsState":
        """Discards the retained population and restores empty sentinels."""

        self.mean = float(
            "nan"
        )

        self.m2 = 0.0
        self.m3 = 0.0
        self.m4 = 0.0

        self.weight_square_sum = 1.0

        return self

    def update(
        self,
        input: float | None,
        alpha: float | None,
    ) -> "AdaptiveMomentsState":
        """Updates the population from one observation when inputs are valid."""

        if (
            _is_missing(
                input
            )
            or _is_missing(
                alpha
            )
        ):
            return self

        observation = float(
            input
        )

        coefficient = _clamp(
            float(
                alpha
            ),
            0.0,
            1.0,
        )

        if _is_missing(
            self.mean
        ):
            self.mean = observation
            self.m2 = 0.0
            self.m3 = 0.0
            self.m4 = 0.0
            self.weight_square_sum = 1.0
            return self

        if coefficient == 0.0:
            return self

        if coefficient == 1.0:
            self.mean = observation
            self.m2 = 0.0
            self.m3 = 0.0
            self.m4 = 0.0
            self.weight_square_sum = 1.0
            return self

        old_mean = self.mean
        old_m2 = self.m2
        old_m3 = self.m3
        old_m4 = self.m4
        old_weight_square_sum = self.weight_square_sum

        retention = (
            1.0
            - coefficient
        )

        delta = (
            observation
            - old_mean
        )

        delta2 = delta * delta
        delta3 = delta2 * delta
        delta4 = delta2 * delta2

        self.mean = (
            old_mean
            + coefficient
            * delta
        )

        self.m2 = (
            retention
            * old_m2
            + coefficient
            * retention
            * delta2
        )

        self.m3 = (
            retention
            * old_m3
            - 3.0
            * coefficient
            * retention
            * delta
            * old_m2
            + coefficient
            * retention
            * (
                retention
                - coefficient
            )
            * delta3
        )

        self.m4 = (
            retention
            * old_m4
            - 4.0
            * coefficient
            * retention
            * delta
            * old_m3
            + 6.0
            * coefficient
            * coefficient
            * retention
            * delta2
            * old_m2
            + coefficient
            * retention
            * (
                1.0
                - 3.0
                * coefficient
                * retention
            )
            * delta4
        )

        self.weight_square_sum = (
            retention
            * retention
            * old_weight_square_sum
            + coefficient
            * coefficient
        )

        return self

    @property
    def variance(
        self,
    ) -> float:
        """Returns the raw weighted population variance when initialized."""

        if _is_missing(
            self.mean
        ):
            return float(
                "nan"
            )

        return self.m2

    @property
    def correction_denominator(
        self,
    ) -> float:
        """Returns one minus normalized-weight concentration."""

        if _is_missing(
            self.mean
        ):
            return float(
                "nan"
            )

        return (
            1.0
            - self.weight_square_sum
        )

    @property
    def corrected_variance(
        self,
    ) -> float:
        """Returns finite-weight corrected variance when defined."""

        denominator = self.correction_denominator

        if (
            isnan(
                denominator
            )
            or denominator <= 0.0
        ):
            return float(
                "nan"
            )

        return (
            self.m2
            / denominator
        )

    @property
    def sigma(
        self,
    ) -> float:
        """Returns the raw weighted population standard deviation."""

        variance = self.variance

        if (
            isnan(
                variance
            )
            or variance < 0.0
        ):
            return float(
                "nan"
            )

        return sqrt(
            variance
        )

    @property
    def corrected_sigma(
        self,
    ) -> float:
        """Returns finite-weight corrected standard deviation when defined."""

        variance = self.corrected_variance

        if (
            isnan(
                variance
            )
            or variance < 0.0
        ):
            return float(
                "nan"
            )

        return sqrt(
            variance
        )

    @property
    def skewness(
        self,
    ) -> float:
        """Returns raw moment skewness when population variance is positive."""

        if (
            _is_missing(
                self.mean
            )
            or self.m2 <= 0.0
        ):
            return float(
                "nan"
            )

        return (
            self.m3
            / (
                self.m2
                ** 1.5
            )
        )

    @property
    def kurtosis(
        self,
    ) -> float:
        """Returns raw Pearson moment kurtosis when variance is positive."""

        if (
            _is_missing(
                self.mean
            )
            or self.m2 <= 0.0
        ):
            return float(
                "nan"
            )

        return (
            self.m4
            / (
                self.m2
                * self.m2
            )
        )

    @property
    def excess_kurtosis(
        self,
    ) -> float:
        """Returns Pearson kurtosis minus three when kurtosis is defined."""

        kurtosis = self.kurtosis

        if isnan(
            kurtosis
        ):
            return float(
                "nan"
            )

        return (
            kurtosis
            - 3.0
        )

    @property
    def effective_sample_size(
        self,
    ) -> float:
        """Returns inverse normalized-weight concentration when defined."""

        if (
            _is_missing(
                self.mean
            )
            or self.weight_square_sum <= 0.0
        ):
            return float(
                "nan"
            )

        return (
            1.0
            / self.weight_square_sum
        )
