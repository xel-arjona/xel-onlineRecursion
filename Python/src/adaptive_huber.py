"""
Recursive adaptive Huber-location reference implementation.

This module mirrors the intended production Pine Script AdaptiveHuberState
semantics for the onlineRecursion library.

For retained location m, current observation x, external non-negative scale s,
positive tuning constant c, and recursive coefficient alpha:

    innovation = x - m

    limit = c * s

    correction = clamp(
        innovation,
        -limit,
        +limit,
    )

    m += alpha * correction

The supplied alpha is clamped to [0, 1].

Inside the Huber clipping region:

    abs(innovation) <= c * s

the recursion is exactly the ordinary first-order IIR update:

    m += alpha * innovation

Outside the clipping region, one-observation influence is bounded:

    abs(delta m) <= alpha * c * s

Scale is deliberately external and may vary from observation to observation.

The tuning constant is bound to the active state and cannot change without
reset.

A scale of zero is valid and produces zero correction after initialization.

The first valid observation initializes location directly from the input,
regardless of alpha or scale.

Missing input, scale, or alpha preserves the complete retained state.
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


class AdaptiveHuberState:
    """
    Persistent recursively adaptive Huber-location state.

    Attributes
    ----------
    value
        Current recursively tracked robust location.

    tuning
        Positive Huber clipping constant bound to the active state.
    """

    def __init__(
        self,
    ) -> None:

        self.value: float = float(
            "nan"
        )

        self.tuning: float = float(
            "nan"
        )

    def reset(
        self,
    ) -> "AdaptiveHuberState":
        """
        Discards the retained location and tuning binding.
        """

        self.value = float(
            "nan"
        )

        self.tuning = float(
            "nan"
        )

        return self

    def update(
        self,
        input_value: float | None,
        scale: float | None,
        tuning: float,
        alpha: float | None,
    ) -> "AdaptiveHuberState":
        """
        Updates the recursively adaptive Huber location.

        Parameters
        ----------
        input_value
            Current observation.

        scale
            Current external non-negative scale in input units.

        tuning
            Fixed positive Huber clipping constant.

        alpha
            Recursive update coefficient. Numeric values are clamped to
            [0, 1].

        Returns
        -------
        AdaptiveHuberState
            This mutated state object.

        Notes
        -----
        Configuration validation occurs before observation-validity handling.

        Missing input, scale, or alpha preserves the complete retained state.

        The first valid observation initializes directly from the input,
        regardless of alpha or scale.

        A zero scale is valid and produces zero correction after
        initialization.

        Alpha = 0 initializes an empty state from the first valid observation
        and freezes an already-active state.
        """

        # ------------------------------------------------------------------
        # CONFIGURATION VALIDATION
        # ------------------------------------------------------------------

        if (
            _is_missing(
                tuning
            )
            or tuning <= 0.0
        ):
            raise ValueError(
                "Huber tuning must be strictly positive."
            )

        if (
            not _is_missing(
                self.tuning
            )
            and tuning != self.tuning
        ):
            raise ValueError(
                "Huber tuning cannot change inside an active state."
            )

        if (
            not _is_missing(
                scale
            )
            and scale < 0.0
        ):
            raise ValueError(
                "Huber scale cannot be negative."
            )

        # ------------------------------------------------------------------
        # OBSERVATION VALIDITY
        # ------------------------------------------------------------------

        valid_observation = (
            not _is_missing(
                input_value
            )
            and not _is_missing(
                scale
            )
            and not _is_missing(
                alpha
            )
        )

        if not valid_observation:
            return self

        observation = float(
            input_value
        )

        external_scale = float(
            scale
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
            self.tuning = tuning

            return self

        # ------------------------------------------------------------------
        # HUBER LOCATION RECURSION
        # ------------------------------------------------------------------

        innovation = (
            observation
            - self.value
        )

        limit = (
            tuning
            * external_scale
        )

        correction = _clamp(
            innovation,
            -limit,
            +limit,
        )

        self.value += (
            coefficient
            * correction
        )

        return self
