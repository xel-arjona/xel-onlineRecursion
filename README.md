# XeL OnlineRecursion

**XeL OnlineRecursion** is a platform-oriented toolkit for online / streaming
statistics, recursive estimation, and finance-oriented stateful processing.

The project is heavily inspired in architectural spirit by public online
statistics and incremental-learning frameworks such as River, while remaining
an independent implementation rather than a River port or compatibility layer.

Its primary implementation target is Pine Script, with Python serving as the
mathematical reference, research, and validation environment.

Additional platform ports may be developed independently while preserving the
same accepted statistical semantics.

## Release status

Current development release:

```text
1.0.0-rc.2 (2026-09-04)
```

Status:

```text
Pre-release / architectural hardening
```

The public API has not yet been declared stable.

Project release numbering is independent of TradingView's Pine library
publication-version numbering.

For example, a TradingView library import revision such as `/1` identifies a
published Pine library revision and should not be interpreted as equivalent to
semantic version `1.0.0`.

---

## Purpose

XeL OnlineRecursion is intended to provide a compact set of foundational
online-statistical primitives for continuous numerical and financial data
streams.

The project emphasizes:

- explicit retained statistical state
- O(1) per-observation updates where practical
- clearly defined recursive population semantics
- deterministic initialization behavior
- explicit reset semantics
- explicit missing-observation behavior
- composable statistical primitives
- minimal redundant state
- finance-oriented recursive weighting
- statistical validation before production integration

The library is intended primarily as statistical infrastructure rather than as
a collection of trading signals.

---

## Architectural philosophy

The central design principle is:

> Retained state represents a statistical population. Derived statistics and
> model interpretations should be built from that state rather than introducing
> redundant recursion.

Conceptually:

```text
stream / population mechanics
            |
            v
    statistical state
            |
      +-----+-----+
      |           |
      v           v
derived       model
statistics    interpretations
      |           |
      +-----+-----+
            |
            v
finance-oriented evidence
```

A new state object should exist only when a genuinely different retained
population or set of sufficient statistics is required.

If a statistic can be derived algebraically from an existing state, the
preferred design is to expose it as a derived view or composition.

Examples:

```text
MAD
    -> composition over adaptive quantile state

Linear regression / beta
    -> derived from covariance state

Skewness / kurtosis
    -> derived from generic moment state

Autocorrelation
    -> intended to reuse covariance machinery

Z-score
    -> intended as a composition over retained location and scale
```

---

## Statistical population semantics

XeL OnlineRecursion treats population geometry as part of the mathematical API.

Different tools may represent different population types, including:

```text
cumulative population

finite rolling population

exponentially weighted population

anchored population

conditional population

observation-clock population

event-clock population
```

These interpretations are not considered interchangeable.

Initialization, missing observations, coefficient behavior, reset timing, and
population boundaries are therefore documented and tested as statistical
semantics rather than implementation details.

---

## Statistical backbone

The project is being organized around generic retained statistical state.

### Generic univariate moments

The intended generic moment backbone is:

```text
AdaptiveMomentsState
    mean
    m2
    m3
    m4
    weightSquareSum
```

From that state, generic derived statistics may include:

```text
variance
corrected variance

sigma
corrected sigma

skewness

kurtosis
excess kurtosis

effective sample size
```

Higher-level distribution interpretations should be built above this generic
moment state.

### Generic bivariate moments

The current bivariate backbone is:

```text
AdaptiveCovarianceState
    meanX
    meanY

    varianceX
    varianceY

    covariance

    weightSquareSum
```

Derived statistics include:

```text
finite-weight corrected covariance
finite-weight corrected variances

correlation

effective sample size

regression beta
regression intercept
R²
```

Simple exponentially weighted linear regression therefore introduces no
separate regression state.

---

## Specialized statistical state

Some estimators require genuinely distinct recursive population semantics and
therefore retain their own state.

Current examples include:

```text
AdaptiveQuantileState
AdaptiveExpectileState
AdaptiveTailMeanState
AdaptiveHuberState
P2QuantileState
```

These state objects exist because their recursions cannot be represented solely
as algebraic views of ordinary central moments.

---

## Model interpretations

Higher-level statistical models should sit above generic state whenever
possible.

### Heavy-tail model

The existing heavy-tail machinery estimates recursively weighted central
moments and derives quantities such as:

```text
standard scale

Pearson kurtosis

Student-t moment-implied degrees of freedom

Student-t variance-equivalent scale

Gaussian-equivalent absolute-innovation scale
```

`AdaptiveMomentsState` owns the generic retained central-moment population.
`HeavyTailState` embeds that state and adds HeavyTail-specific model
interpretation, including absolute-innovation scale, standard scale, kurtosis,
Student-t degrees of freedom, and Student-t scale.

---

## Current statistical capabilities

XeL OnlineRecursion currently includes:

### Stream and population mechanics

- first-order IIR recursion
- recursive extrema
- sample-and-hold state
- conditional settlement state
- cumulative accumulation
- exact rolling sums
- exact sampled rolling sums
- settled population sums

### Quantile and asymmetric location statistics

- P² cumulative quantile estimation
- adaptive quantile estimation
- adaptive median estimation
- adaptive expectile estimation

### Robust statistics

- adaptive Huber location
- adaptive median absolute deviation
- Gaussian-equivalent MAD scale

### Tail statistics

- exponentially weighted conditional lower-tail means
- exponentially weighted conditional upper-tail means

### Bivariate statistics

- exponentially weighted covariance
- finite-weight corrected covariance
- marginal variances
- Pearson correlation
- effective sample size

### Regression

- exponentially weighted beta of Y on X
- regression intercept
- R²
- complete simple linear-regression interface

### Distribution modeling

- recursively weighted central moments
- standard scale
- kurtosis
- bounded Student-t moment interpretation
- Student-t variance-equivalent scale
- absolute-innovation comparison scale

### Recursive weighting

- fixed decay coefficients
- anchored normalized coefficients
- complementary alpha composition
- volume participation weighting
- time-domain participation weighting
- open-interest turnover weighting
- float-turnover weighting
- fund-turnover weighting

### Finance-oriented models

- relative returns
- relative projections
- additive moment scaling
- realized-return distributions
- recursive market dispersion

---

## Repository structure

XeL OnlineRecursion is organized as one conceptual statistical project with
independent platform implementations.

```text
xel-onlineRecursion/
|
+-- PineScript/
|   |
|   +-- onlineRecursion.pine
|
+-- Python/
|   |
|   +-- src/
|   +-- tests/
|   +-- experiments/
|   +-- results/
|
+-- Quantower/          future
|
+-- other ports         future
|
+-- README.md
+-- AGENTS.md
+-- LICENSE
+-- .gitignore
```

### `PineScript/`

Contains the canonical Pine Script v6 production implementation.

The principal library source is:

```text
PineScript/onlineRecursion.pine
```

TradingView remains the authoritative compiler and runtime environment for the
Pine implementation.

### `Python/`

Contains:

```text
src/
    transparent mathematical reference implementations

tests/
    permanent deterministic and semantic regression tests

experiments/
    statistical research, Monte Carlo studies, and validation programs

results/
    selected retained validation evidence
```

Python is the primary mathematical research environment used before statistical
changes are accepted into the Pine implementation.

### Future platform ports

Additional implementations may be introduced in independent top-level
directories, for example:

```text
Quantower/
NinjaTrader/
Rust/
```

Each port should use the natural architecture of its target environment while
preserving the project's accepted statistical semantics.

---

## Validation methodology

Statistical primitives are validated before being considered production-ready.

The preferred workflow is:

```text
1. define exact mathematical semantics

2. implement a transparent Python reference

3. prove deterministic identities and invariants

4. perform statistical / Monte Carlo validation when necessary

5. add permanent Python regression tests

6. integrate the accepted primitive into Pine Script

7. validate Pine behavior independently

8. freeze accepted estimator mathematics
```

Particular attention is given to:

- initialization
- missing observations
- reset-before-current behavior
- alpha boundaries
- alpha clamping
- dimensional consistency
- symmetry and invariance properties
- population timing
- finite-weight effects
- effective sample size
- caller/state timing when estimators are composed

An empirical improvement in one simulation is not considered sufficient
justification for an ad-hoc statistical correction.

---

## Relationship to River and other online-statistics frameworks

XeL OnlineRecursion is heavily inspired by the general architectural ideas
found in mature online-statistics and incremental-learning frameworks such as
River.

Shared conceptual themes include:

- one-observation-at-a-time processing
- persistent sufficient state
- incremental statistics
- bounded-memory computation
- composability
- online adaptation

XeL OnlineRecursion is not a River port and does not attempt API or behavioral
compatibility.

Its scope is intentionally narrower and places greater emphasis on:

- Pine Script execution constraints
- recursive population geometry
- exponentially weighted statistics
- explicit timing semantics
- robust statistics
- finance-oriented weighting
- streaming market-data applications

---

## Computational scope

Where practical, algorithms are designed for:

```text
O(1) arithmetic per observation
```

and:

```text
O(1) retained state with respect to population length
```

Some exact finite-window operations necessarily retain or address finite
population membership and are documented accordingly.

Bounded-state operation does not imply that every estimator represents the same
type of population.

---

## Licensing

XeL OnlineRecursion is intended to remain open statistical infrastructure.

The project is licensed under the:

```text
Mozilla Public License 2.0
SPDX-License-Identifier: MPL-2.0
```

MPL 2.0 preserves openness of modifications to MPL-covered source files while
allowing the library to be incorporated into larger works without imposing the
same license on unrelated source files.

Platform-specific publication and reuse rules may impose additional
requirements beyond the software license. Users distributing or publishing
derived work are responsible for complying with both the applicable license and
the rules of the target platform.

See:

```text
LICENSE
```

for the complete license terms.

---

## Current development baseline

Before the generic univariate-moment refactor, the permanent Python regression
suite is:

```text
161 passed
```

This baseline is treated as the pre-refactor semantic checkpoint.

---

## Project status

XeL OnlineRecursion is currently in pre-release architectural hardening.

Current work is focused on:

```text
generic statistical-state architecture

univariate moment-state extraction

separation of generic moments from HeavyTail interpretation

API consistency

documentation consistency

cross-platform project organization
```

The project has not yet declared a stable public API.

Breaking architectural improvements may therefore still occur before the first
stable release.
