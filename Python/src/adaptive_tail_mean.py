"""
Exponentially weighted adaptive conditional tail mean.

This module defines the deterministic reference implementation intended to
mirror the production Pine Script semantics of AdaptiveTailMeanState.

For a supplied threshold q and observation x, define tail membership as:

    lower tail:
        I = 1 when x <= q
            0 otherwise

    upper tail:
        I = 1 when x >= q
            0 otherwise

The state recursively tracks:

    weighted_tail[t]
        = (1 - alpha) * weighted_tail[t-1]
          + alpha * I[t] * x[t]

    tail_mass[t]
        = (1 - alpha) * tail_mass[t-1]
          + alpha * I[t]

and reports:

    value[t]
        = weighted_tail[t] / tail_mass[t]

whenever tail_mass is numerically positive.

This is an observation-clock exponentially weighted conditional mean.

A non-tail observation decays numerator and denominator by the same factor,
so an already-defined tail mean remains unchanged.

A tail observation performs a normalized convex update toward that
observation.

The supplied threshold is deliberately external and may vary from observation
to observation. In particular, it may be supplied by an adaptive quantile
estimator.

When the threshold is the corresponding quantile of a continuous population,
the target equals the corresponding conditional tail expectation and therefore
has the usual Expected Shortfall / CVaR interpretation under the selected sign
convention.

For populations with probability mass exactly at the threshold, this generic
conditional-tail estimator must not automatically be identified with the
canonical quantile-integral Expected Shortfall definition.

Alpha is an observation-clock recursive coefficient and is clamped to [0, 1].

A zero alpha is a valid zero-weight observation. It does not initialize an
empty state.

Missing input, threshold, or alpha preserves the complete retained state.
"""

from math import isnan


MIN_FLOAT = 1e-16


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


class AdaptiveTailMeanState:
    """
    Persistent exponentially weighted conditional tail-mean state.

    Attributes
    ----------
    value
        Current normalized conditional tail mean, or NaN when no numerically
        positive retained tail mass exists.

    weighted_tail
        Exponentially weighted tail-value numerator.

    tail_mass
        Exponentially weighted tail-membership mass.

    side
        Tail-side configuration bound to the active state. Either "lower",
        "upper", or None before activation.
    """

    def __init__(
        self,
    ) -> None:

        self.value: float = float(
            "nan"
        )

        self.weighted_tail: float = 0.0

        self.tail_mass: float = 0.0

        self.side: str | None = None

    def reset(
        self,
    ) -> "AdaptiveTailMeanState":
        """
        Discards the complete retained tail state.
        """

        self.value = float(
            "nan"
        )

        self.weighted_tail = 0.0
        self.tail_mass = 0.0

        self.side = None

        return self

    def update(
        self,
        input_value: float | None,
        threshold: float | None,
        alpha: float | None,
        side: str,
    ) -> "AdaptiveTailMeanState":
        """
        Updates the exponentially weighted conditional tail mean.

        Parameters
        ----------
        input_value
            Current observation.

        threshold
            Current external tail threshold. The threshold may vary through
            time and is not bound to the state.

        alpha
            Observation-clock recursive coefficient. Numeric values are
            clamped to [0, 1].

        side
            Fixed tail-side configuration: "lower" or "upper".

        Returns
        -------
        AdaptiveTailMeanState
            This mutated state object.

        Notes
        -----
        Configuration validation precedes observation-validity handling.

        Once a positive-weight observation activates the state, `side` cannot
        change without reset.

        A positive-weight non-tail observation may activate the side
        configuration even though the normalized tail mean remains undefined
        until positive tail mass exists.

        Alpha = 0 performs no state mutation and does not activate an empty
        state.

        Missing input, threshold, or alpha preserves the complete state.
        """

        # ------------------------------------------------------------------
        # CONFIGURATION VALIDATION
        # ------------------------------------------------------------------

        if side not in (
            "lower",
            "upper",
        ):
            raise ValueError(
                "Tail side must be 'lower' or 'upper'."
            )

        if (
            self.side is not None
            and side != self.side
        ):
            raise ValueError(
                "Tail side cannot change inside an active state."
            )

        # ------------------------------------------------------------------
        # OBSERVATION VALIDITY
        # ------------------------------------------------------------------

        valid_observation = (
            not _is_missing(
                input_value
            )
            and not _is_missing(
                threshold
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

        tail_threshold = float(
            threshold
        )

        coefficient = _clamp(
            float(
                alpha
            ),
            0.0,
            1.0,
        )

        # A zero-weight observation performs no state mutation.
        if coefficient <= 0.0:
            return self

        # Bind tail-side configuration when a positive-weight sequence begins.
        if self.side is None:
            self.side = side

        # ------------------------------------------------------------------
        # TAIL MEMBERSHIP
        # ------------------------------------------------------------------

        if side == "lower":

            indicator = (
                1.0
                if observation <= tail_threshold
                else 0.0
            )

        else:

            indicator = (
                1.0
                if observation >= tail_threshold
                else 0.0
            )

        retention = (
            1.0
            - coefficient
        )

        # ------------------------------------------------------------------
        # EXPONENTIALLY WEIGHTED CONDITIONAL MOMENTS
        # ------------------------------------------------------------------

        self.weighted_tail = (
            retention
            * self.weighted_tail
            + coefficient
            * indicator
            * observation
        )

        self.tail_mass = (
            retention
            * self.tail_mass
            + coefficient
            * indicator
        )

        # ------------------------------------------------------------------
        # NORMALIZED TAIL MEAN
        # ------------------------------------------------------------------

        self.value = (
            self.weighted_tail
            / self.tail_mass
            if self.tail_mass > MIN_FLOAT
            else float(
                "nan"
            )
        )

        return self
