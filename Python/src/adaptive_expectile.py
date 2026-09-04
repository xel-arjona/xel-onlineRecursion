"""
Recursive adaptive expectile reference implementation.

This module mirrors the production Pine Script AdaptiveExpectileState
semantics used by the onlineRecursion library.

For retained expectile e and current observation x:

    innovation = x - e

    asymmetric_weight =
        level       if innovation > 0
        1 - level   if innovation < 0
        0           otherwise

    e += alpha * asymmetric_weight * innovation

The supplied alpha is a baseline recursive adaptation coefficient.
It is clamped to [0, 1].

The expectile level is fixed for the lifetime of an active state and must
satisfy:

    0 < level < 1

Unlike the adaptive quantile estimator, no auxiliary scale estimator is
required because the innovation itself supplies the dimensioned correction.

At level = 0.5 the recursion targets the ordinary squared-loss location mean,
but its effective recursive coefficient is alpha / 2.
"""

from math import isnan


def _is_missing(
    value: float | None,
) -> bool:
    """
    Returns True for None or NaN.
    """

    return (
        value is None
        or (
            isinstance(
                value,
                float,
            )
            and isnan(
                value
            )
        )
    )


def _clamp(
    value: float,
    lower: float,
    upper: float,
) -> float:
    """
    Constrains value to the inclusive interval [lower, upper].
    """

    return max(
        lower,
        min(
            upper,
            value,
        ),
    )


class AdaptiveExpectileState:
    """
    Persistent recursively adaptive expectile estimator state.

    Attributes
    ----------
    value
        Current recursively tracked expectile estimate.

    level
        Expectile level bound to the active recursive state.
    """

    def __init__(
        self,
    ) -> None:

        self.value: float = float(
            "nan"
        )

        self.level: float = float(
            "nan"
        )

    def reset(
        self,
    ) -> "AdaptiveExpectileState":
        """
        Discards the retained expectile location and level binding.
        """

        self.value = float(
            "nan"
        )

        self.level = float(
            "nan"
        )

        return self

    def update(
        self,
        input_value: float | None,
        level: float,
        alpha: float | None,
    ) -> "AdaptiveExpectileState":
        """
        Updates the recursively adaptive expectile state.

        Parameters
        ----------
        input_value
            Current observation.

        level
            Fixed expectile level satisfying 0 < level < 1.

        alpha
            Baseline recursive adaptation coefficient. Valid numeric
            observations are clamped to [0, 1].

        Returns
        -------
        AdaptiveExpectileState
            This mutated state object.

        Notes
        -----
        Configuration validation occurs before observation-validity handling,
        matching the Pine implementation.

        Missing input or alpha preserves the complete retained state.

        The first valid observation initializes the estimator directly from
        the input, regardless of the clamped alpha value. Consequently,
        alpha = 0 can initialize an empty state and then freeze it.

        The recursion is:

            innovation = input_value - value

            weight =
                level       when innovation > 0
                1 - level   when innovation < 0
                0           otherwise

            value += alpha * weight * innovation
        """

        # ------------------------------------------------------------------
        # CONFIGURATION VALIDATION
        # ------------------------------------------------------------------

        if (
            _is_missing(
                level
            )
            or level <= 0.0
            or level >= 1.0
        ):
            raise ValueError(
                "Expectile level must be strictly between zero and one."
            )

        if (
            not _is_missing(
                self.level
            )
            and level != self.level
        ):
            raise ValueError(
                "Expectile level cannot change inside an active state."
            )

        # ------------------------------------------------------------------
        # OBSERVATION VALIDITY
        # ------------------------------------------------------------------

        valid_observation = (
            not _is_missing(
                input_value
            )
            and not _is_missing(
                alpha
            )
        )

        if not valid_observation:
            return self

        # The validity check above establishes these as numeric values.
        observation = float(
            input_value
        )

        coefficient = _clamp(
            float(
                alpha
            ),
            0.0,
            1.0,
        )

        # ------------------------------------------------------------------
        # INITIALIZATION
        # ------------------------------------------------------------------

        if _is_missing(
            self.value
        ):
            self.value = observation
            self.level = level

            return self

        # ------------------------------------------------------------------
        # EXPECTILE RECURSION
        # ------------------------------------------------------------------

        innovation = (
            observation
            - self.value
        )

        if innovation > 0.0:
            asymmetric_weight = (
                level
            )

        elif innovation < 0.0:
            asymmetric_weight = (
                1.0
                - level
            )

        else:
            asymmetric_weight = (
                0.0
            )

        self.value += (
            coefficient
            * asymmetric_weight
            * innovation
        )

        return self
