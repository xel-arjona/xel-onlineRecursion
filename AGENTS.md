# XeL OnlineRecursion — Repository Agent Contract

This file defines the mathematical, architectural, validation, repository, and
tooling rules for work on XeL OnlineRecursion.

Read this file completely before modifying repository content.

When implementation details, older comments, or convenience patterns conflict
with an accepted contract in this file, stop and report the discrepancy before
changing semantics.

---

# Project identity

Public project name:

```text
XeL OnlineRecursion
```

Repository name:

```text
xel-onlineRecursion
```

Canonical Pine source:

```text
PineScript/onlineRecursion.pine
```

Technical Pine library identifier:

```text
onlineRecursion
```

Current project release notation:

```text
1.0.0-rc.2 (2026-09-04)
```

The project remains in pre-release architectural hardening and its public API
has not yet been declared stable.

TradingView publication revisions such as:

```text
/1
/2
/3
```

are TradingView publication revisions and are independent of project SemVer.

Do not conflate the two versioning systems.

License:

```text
Mozilla Public License 2.0
MPL-2.0
```

The repository contains the full license text in:

```text
LICENSE
```

River and similar libraries may be acknowledged as conceptual or architectural
inspiration where appropriate.

XeL OnlineRecursion is not a River port, API compatibility layer, or wrapper.

---

# Repository organization

Canonical repository:

```text
~/Dev/github/xel-onlineRecursion/
```

Primary structure:

```text
xel-onlineRecursion/
├── AGENTS.md
├── LICENSE
├── README.md
├── PineScript/
│   └── onlineRecursion.pine
└── Python/
    ├── src/
    ├── tests/
    ├── experiments/
    └── results/
```

The repository copy is canonical.

Historical or scratch Pine files outside this repository must not be treated as
the authoritative implementation.

Do not edit unrelated historical copies when repository work is requested.

---

# Architectural principle

The project is organized around retained statistical populations.

The governing rule is:

```text
Foundational state represents the population.
Derived statistics interpret that population.
Models add model-specific assumptions.
Applications decide when evidence is sufficient to act.
```

This separation is mandatory.

Startup maturity, warmup policy, trading readiness, signal gating, minimum-bar
requirements, and application-level confidence must not leak into foundational
statistical state unless mathematically required by the estimator itself.

---

# Population semantics

A foundational recursive state represents the statistical population retained
by the recursion.

Initialization therefore means:

```text
population creation
```

It is not merely the first ordinary recursive transition.

This distinction is especially important when:

```text
alpha = 0
```

A valid first observation can create a valid singleton population even though
an already-active state with alpha zero must freeze.

The population contract takes precedence over convenience behavior inherited
from older model-specific implementations.

---

# Statistical architecture

The intended generic backbone is:

```text
STREAM / POPULATION MECHANICS
    existing stream states

STATISTICAL STATE BACKBONE
    AdaptiveMomentsState
        mean
        m2
        m3
        m4
        weightSquareSum

    AdaptiveCovarianceState
        meanX
        meanY
        varianceX
        varianceY
        covariance
        weightSquareSum

    AdaptiveQuantileState
    AdaptiveExpectileState
    AdaptiveTailMeanState
    AdaptiveHuberState

DERIVED STATISTICS
    variance
    corrected variance
    sigma
    corrected sigma
    skewness
    kurtosis
    excess kurtosis
    effective sample size
    covariance
    correlation
    regression views
    MAD
    Gaussian-equivalent MAD scale

MODEL INTERPRETATIONS
    HeavyTail
        Student-t degrees of freedom
        t scale
        absolute-innovation scale
        other heavy-tail-specific interpretations
```

Generic sufficient statistics should be reused rather than duplicated inside
higher-level models.

---

# Validation discipline

Statistical components are accepted in stages.

A mathematically closed component should not be reopened without evidence of a
real discrepancy.

Valid reasons to reopen a closed component include:

```text
failed parity
failed deterministic validation
new mathematical contradiction
runtime/compiler discrepancy
documented implementation mismatch
```

Preference or stylistic cleanup alone is not sufficient.

Validation should proceed from the most independent oracle available.

Preferred hierarchy:

```text
1. explicit mathematical identities
2. deterministic direct-weight calculations
3. independent Python references
4. observation-by-observation parity against accepted implementations
5. Pine compiler/runtime validation in TradingView
```

Tests should distinguish:

```text
semantic identity
mathematical correctness
boundary behavior
runtime integration
```

Do not treat one implementation copied into another language as the only
evidence of correctness.

---

# Missing-data convention

For foundational recursive states, missing required input preserves retained
state unless a component-specific contract explicitly says otherwise.

In Python, missing values normally include:

```text
None
NaN
```

In Pine, missing values use:

```text
na
```

Missing current data may cause a functional wrapper to return `na` for that
call while the underlying state remains unchanged.

State preservation and functional-output gating are separate semantics.

---

# Alpha convention

Where the accepted contract specifies a feedback coefficient:

```text
alpha
```

clamp it to:

```text
[0, 1]
```

unless a particular estimator explicitly defines different bounds.

For normalized exponentially weighted population states:

```text
active alpha = 0
```

means exact freeze.

```text
active alpha = 1
```

means replacement by the current singleton population.

First-observation semantics are defined separately from active-state semantics.

---

# AdaptiveExpectile — closed

Accepted recurrence:

```text
innovation = input - previousValue

weight =
    level       if innovation > 0
    1-level     if innovation < 0
    0           if innovation == 0

value += alpha * weight * innovation
```

Contract:

```text
0 < level < 1
```

No scale normalization.

No factor of two.

No bandwidth correction.

No internal normalization.

`level` is validated while active.

`alpha` clamps to `[0,1]`.

Missing input preserves state.

The first valid observation initializes directly regardless of alpha.

Active alpha zero freezes.

Reset occurs before processing the current observation.

Functional calls return `na` on missing required current input while retained
state remains preserved.

At:

```text
level = 0.5
```

the estimator targets the mean with effective recursion coefficient
`alpha / 2`.

This component is mathematically closed.

---

# AdaptiveTailMean — closed

Accepted exponentially weighted conditional-tail recursion:

```text
indicator =
    lower tail: input <= threshold
    upper tail: input >= threshold

weightedTail =
    (1-alpha) * previousWeightedTail
    + alpha * indicator * input

tailMass =
    (1-alpha) * previousTailMass
    + alpha * indicator

value =
    weightedTail / tailMass
```

when tail mass is positive.

This is observation-clock recursion, not event-clock recursion.

The threshold is external.

When coupled to an adaptive quantile, caller timing uses the previous quantile:

```text
previousQuantile = quantile.value

tail.update(
    input,
    previousQuantile,
    tailAlpha,
    side
)

quantile.update(...)
```

This component is mathematically closed.

---

# AdaptiveHuber — closed

Accepted recurrence:

```text
innovation = input - previousValue

limit = tuning * scale

correction =
    clamp(
        innovation,
        -limit,
        +limit
    )

value += alpha * correction
```

Contract:

```text
tuning > 0
scale >= 0
```

Scale is external.

No internal bandwidth normalization.

No alpha-dependent tuning normalization.

Missing input preserves state.

The first valid observation initializes directly.

Active alpha zero freezes.

When coupled to an adaptive scale estimator, use caller-controlled previous
scale timing rather than silently introducing same-step feedback.

This component is mathematically closed.

---

# Adaptive MAD and robust scale — closed

There is no dedicated `AdaptiveMADState`.

MAD is composition:

```text
deviation = abs(input - externalCenter)

rawMAD =
    adaptiveMedian(
        deviation,
        alpha
    )
```

Gaussian-equivalent robust scale uses:

```text
MAD_TO_GAUSSIAN_SCALE
    ≈ 1.482602218505602
```

which is:

```text
1 / Phi^-1(0.75)
```

Caller timing uses the previous center where appropriate.

Pine interfaces include:

```text
adaptiveMAD(...)
gaussianScaleFromMAD(...)
```

This component is mathematically closed.

---

# AdaptiveCovariance — closed

Retained state:

```text
meanX
meanY
varianceX
varianceY
covariance
weightSquareSum
```

Accepted recursive update:

```text
dx = x - meanX
dy = y - meanY

meanX += alpha * dx
meanY += alpha * dy

covariance =
    (1-alpha)
    * (
        covariance
        + alpha * dx * dy
    )

varianceX =
    (1-alpha)
    * (
        varianceX
        + alpha * dx^2
    )

varianceY =
    (1-alpha)
    * (
        varianceY
        + alpha * dy^2
    )

weightSquareSum =
    (1-alpha)^2 * weightSquareSum
    + alpha^2
```

The finite-weight correction denominator is:

```text
1 - weightSquareSum
```

Corrected second moments are:

```text
correctedMoment =
    rawMoment
    / (1 - weightSquareSum)
```

when the denominator is strictly positive.

Effective sample size is:

```text
Neff =
    1 / weightSquareSum
```

when `weightSquareSum > 0`.

For constant alpha in a continuing stationary recursion:

```text
weightSquareSum
    -> alpha / (2 - alpha)
```

and therefore:

```text
Neff
    -> (2 - alpha) / alpha
```

Correlation uses the raw ratio because the common finite-weight multiplicative
correction cancels.

The correction above is a finite-weight second-moment correction under the
appropriate normalized recursive-weight assumptions.

It does not remove general nonlinear finite-sample bias from correlation or
other nonlinear transforms.

This component is mathematically closed.

---

# Recursive linear regression — closed

Regression is derived algebraically from `AdaptiveCovarianceState`.

For `Y` on `X`:

```text
beta =
    covariance / varianceX
```

when `varianceX > 0`.

Intercept:

```text
intercept =
    meanY - beta * meanX
```

Coefficient of determination:

```text
R^2 =
    correlation^2
```

No separate regression state is required.

Finite-weight correction cancels exactly from beta because covariance and
variance share the same multiplicative correction.

Pine methods include:

```text
betaYOnX()
interceptYOnX()
rSquared()
```

Functional interfaces include:

```text
adaptiveBeta()
adaptiveLinearRegression()
```

This component is mathematically closed.

---

# AdaptiveMomentsState

`AdaptiveMomentsState` is the generic normalized exponentially weighted
univariate central-moment population state.

It must remain model-agnostic.

Retained state:

```text
mean
m2
m3
m4
weightSquareSum
```

Do not add model-specific quantities such as:

```text
absolute innovation
Student-t degrees of freedom
t scale
HeavyTail-specific sigma policy
warmup readiness
signal maturity
```

to this foundational state.

---

# AdaptiveMomentsState initialization

Initialization is population creation.

The first valid observation with a valid alpha creates the normalized singleton
population regardless of alpha after clamping.

This includes:

```text
alpha = 0
```

Singleton initialization:

```text
mean = input
m2 = 0
m3 = 0
m4 = 0
weightSquareSum = 1
```

Therefore:

```text
empty state
+ valid input
+ alpha = 0
```

must create a singleton population.

This intentionally differs from historical HeavyTail startup behavior.

---

# AdaptiveMomentsState active alpha behavior

For an initialized state:

```text
alpha = 0
```

must freeze the complete state exactly.

No retained field changes.

For:

```text
alpha = 1
```

the state becomes the current singleton exactly:

```text
mean = input
m2 = 0
m3 = 0
m4 = 0
weightSquareSum = 1
```

Values below zero clamp to zero.

Values above one clamp to one.

---

# AdaptiveMomentsState missing-data behavior

Missing input or missing alpha preserves the complete retained state.

No partial update is allowed.

In Python, missing means:

```text
None
NaN
```

as supported by established repository conventions.

In Pine, missing means:

```text
na
```

---

# AdaptiveMomentsState reset

`reset()` clears the statistical population before any subsequent current
observation is processed.

Reset representation:

```text
mean = NaN / na
m2 = 0
m3 = 0
m4 = 0
weightSquareSum = 1
```

A reset followed by a valid observation creates a new singleton.

This remains true when:

```text
alpha = 0
```

Reset plus missing required current input remains empty.

---

# AdaptiveMomentsState central-moment recurrence

For an initialized state and:

```text
0 < alpha < 1
```

snapshot the complete old state before mutation.

Let:

```text
a = alpha
b = 1 - a

delta =
    input - oldMean

delta2 =
    delta * delta

delta3 =
    delta2 * delta

delta4 =
    delta2 * delta2
```

Then update:

```text
mean' =
    oldMean
    + a * delta
```

Second central moment:

```text
m2' =
    b * oldM2
    + a * b * delta2
```

Third central moment:

```text
m3' =
    b * oldM3
    - 3 * a * b * delta * oldM2
    + a * b * (b - a) * delta3
```

Fourth central moment:

```text
m4' =
    b * oldM4
    - 4 * a * b * delta * oldM3
    + 6 * a * a * b * delta2 * oldM2
    + a * b * (1 - 3 * a * b) * delta4
```

Weight concentration:

```text
weightSquareSum' =
    b * b * oldWeightSquareSum
    + a * a
```

Every right-hand-side moment must use the saved old state.

Do not use newly updated lower moments in higher-moment equations.

The accepted HeavyTail central-moment recurrence is the migration oracle for:

```text
mean
m2
m3
m4
```

once the two states have been initialized consistently.

---

# AdaptiveMomentsState weight concentration

`weightSquareSum` is:

```text
sum_i w_i^2
```

for the current normalized recursive observation weights.

Its recursion is:

```text
S2_new =
    (1-alpha)^2 * S2_old
    + alpha^2
```

For constant alpha in a continuing stationary recursion:

```text
S2
    -> alpha / (2 - alpha)
```

and therefore:

```text
Neff
    -> (2 - alpha) / alpha
```

---

# AdaptiveMomentsState empty-state interpretation

The reset / empty storage representation is an implementation sentinel, not a
statistical population.

The empty state is identified by:

```text
mean = NaN
```

or, in Pine:

```text
mean = na
```

Other retained fields contain neutral reset placeholders:

```text
m2 = 0
m3 = 0
m4 = 0
weightSquareSum = 1
```

These placeholder values must not be interpreted as statistics of an empty
population.

Therefore, while the state is empty, all derived statistical views are
undefined.

They must return:

```text
NaN
```

in Python or:

```text
na
```

in Pine.

This applies to:

```text
variance
correctionDenominator
correctedVariance
sigma
correctedSigma
skewness
kurtosis
excessKurtosis
effectiveSampleSize
```

After singleton initialization, the same stored numeric values acquire
statistical meaning.

For a valid singleton population:

```text
mean = input
m2 = 0
m3 = 0
m4 = 0
weightSquareSum = 1
```

and therefore:

```text
variance = 0
sigma = 0
effectiveSampleSize = 1
correctionDenominator = 0
```

while:

```text
correctedVariance = undefined
correctedSigma = undefined
skewness = undefined
kurtosis = undefined
excessKurtosis = undefined
```

The implementation must distinguish an empty-state placeholder from a valid
singleton population through initialization state, not merely through the
numeric values of the retained moment fields.

---

# AdaptiveMomentsState derived views

Initial generic derived statistics are:

```text
variance
correctionDenominator
correctedVariance

sigma
correctedSigma

skewness
kurtosis
excessKurtosis

effectiveSampleSize
```

No finite-sample corrections for skewness or kurtosis are part of the
foundational contract.

---

# AdaptiveMomentsState variance

Raw population variance:

```text
variance =
    m2
```

when initialized.

While empty:

```text
variance =
    undefined
```

For a singleton:

```text
variance =
    0
```

---

# AdaptiveMomentsState finite-weight correction

Correction denominator:

```text
correctionDenominator =
    1 - weightSquareSum
```

when initialized.

While empty:

```text
correctionDenominator =
    undefined
```

Corrected variance:

```text
correctedVariance =
    m2
    / (1 - weightSquareSum)
```

only when:

```text
correctionDenominator > 0
```

Otherwise it is undefined.

For a singleton:

```text
correctionDenominator = 0
correctedVariance = undefined
```

---

# AdaptiveMomentsState sigma

Raw sigma:

```text
sigma =
    sqrt(m2)
```

only when the state is initialized and:

```text
m2 >= 0
```

While empty, sigma is undefined.

If numerical state contains:

```text
m2 < 0
```

generic `AdaptiveMomentsState` must not silently clamp it.

Return undefined instead.

Do not introduce an epsilon or tolerance policy without an explicit separate
decision.

Historical HeavyTail behavior using:

```text
sqrt(max(m2, 0))
```

remains model-specific and must not automatically propagate into the generic
state.

Corrected sigma is:

```text
correctedSigma =
    sqrt(correctedVariance)
```

only when corrected variance is defined and non-negative.

---

# AdaptiveMomentsState skewness

Raw moment skewness:

```text
skewness =
    m3
    / m2^(3/2)
```

only when:

```text
m2 > 0
```

Otherwise it is undefined.

No finite-sample skewness correction is part of the foundational state.

---

# AdaptiveMomentsState kurtosis

Pearson moment kurtosis:

```text
kurtosis =
    m4
    / m2^2
```

only when:

```text
m2 > 0
```

Otherwise it is undefined.

Excess kurtosis:

```text
excessKurtosis =
    kurtosis - 3
```

only when kurtosis is defined.

No finite-sample kurtosis correction is part of the foundational state.

A zero-variance singleton therefore has undefined generic kurtosis.

Historical HeavyTail fallback behavior such as:

```text
kurtosis = 3
```

at zero variance is model-specific and must not be copied into
`AdaptiveMomentsState`.

---

# AdaptiveMomentsState effective sample size

Effective sample size:

```text
effectiveSampleSize =
    1 / weightSquareSum
```

when initialized and:

```text
weightSquareSum > 0
```

While empty it is undefined.

For a singleton:

```text
effectiveSampleSize = 1
```

---

# AdaptiveMomentsState maturity policy

The foundational state must not contain:

```text
warmup counters
minimum bars
minimum Neff
startup suppression
maturity thresholds
trading readiness
signal gating
```

A higher-level model or application may decide that a statistical estimate is
not yet mature enough to use.

That decision must not alter the mathematical population represented by the
state.

---

# AdaptiveMomentsState Python validation

The accepted Python reference is:

```text
Python/src/adaptive_moments.py
```

Permanent tests are:

```text
Python/tests/test_adaptive_moments.py
```

The Python implementation must validate:

```text
empty-state semantics
singleton initialization
empty alpha-zero initialization
active alpha-zero freeze
alpha-one replacement
alpha clamping
missing-data preservation
reset behavior
reset plus alpha-zero singleton creation

HeavyTail recurrence identity
fixed alpha
variable alpha
boundary alpha behavior

explicit normalized recursive weights
translation invariance
reflection symmetry
scale transformation
constant sequence behavior

weightSquareSum recursion
constant-alpha asymptotic Neff

variance
corrected variance
sigma
corrected sigma
skewness
kurtosis
excess kurtosis
effective sample size
```

The accepted Python baseline after introduction of `AdaptiveMomentsState` is:

```text
189 passed
```

unless later intentionally expanded.

---

# HeavyTail migration policy

Pre-migration HeavyTail behavior served as the validation oracle during
generic-moment migration.

The migration is complete and validated. `HeavyTailState` now embeds one
`AdaptiveMomentsState` and delegates generic central-moment population mechanics
to it. Historical HeavyTail behavior remains preserved except for the explicitly
accepted pre-release structural field-path change from direct `mean` / `m2` /
`m3` / `m4` fields to fields under `moments`.

Completed migration sequence:

```text
1. validate Python AdaptiveMomentsState
2. validate Pine AdaptiveMomentsState
3. prove Pine mean/m2/m3/m4 identity against HeavyTail
4. validate weightSquareSum independently
5. validate derived generic views
6. close generic moment state
7. refactor HeavyTail onto generic moments
8. prove HeavyTail outputs remain unchanged except explicitly approved startup cleanup
```

HeavyTail retains ownership of model-specific state beyond generic central
moments.

Examples include:

```text
absInnovation
sigma
tScale
absInnovationScale
df
kurtosis interpretation
```

These must not be transferred wholesale into `AdaptiveMomentsState`.

`absInnovation` is not an ordinary central moment.

It remains outside the generic moment backbone.

---

# Known HeavyTail startup difference

Historical HeavyTail behavior:

```text
empty state
+ valid input
+ alpha = 0
-> remains empty
```

Generic AdaptiveMoments behavior:

```text
empty state
+ valid input
+ alpha = 0
-> singleton population
```

This difference is intentional during parity testing.

Do not assert HeavyTail identity for this startup case.

Once both states have been initialized consistently, identity is required for:

```text
mean
m2
m3
m4
```

under equivalent valid updates.

Historical HeavyTail empty-plus-alpha-zero behavior remains preserved after the
completed migration. Harmonizing it with the generic population contract is a
separate deferred policy decision.

---

# Pine AdaptiveMoments integration policy

Pine `AdaptiveMomentsState` exposes the state object and state methods only.

Current public type:

```text
AdaptiveMomentsState
```

Current public methods:

```text
reset()
update()

variance()
correctionDenominator()
correctedVariance()

sigma()
correctedSigma()

skewness()
kurtosis()
excessKurtosis()

effectiveSampleSize()
```

A public functional remains intentionally deferred:

```text
adaptiveMoments(...)
```

It is not required for component closure and must not be added until its tuple
shape, output order, and missing-call output policy are explicitly designed and
accepted.

---

# Pine validation policy

TradingView is the Pine compiler/runtime oracle.

Do not intentionally use destructive crash paths in production validation when
the same semantics can be tested deterministically elsewhere.

Temporary positive deterministic Pine harnesses are allowed.

Temporary harnesses must:

```text
use deterministic constants
test exact intended semantics
produce explicit failure diagnostics
be removed after successful validation
```

For AdaptiveMoments, Pine validation should cover:

```text
empty state
singleton initialization
empty alpha-zero initialization
active alpha-zero freeze
alpha-one singleton replacement
missing preservation
reset-before-current behavior

HeavyTail mean/m2/m3/m4 parity

weightSquareSum independent recursion

variance
correction denominator
corrected variance
sigma
corrected sigma
skewness
kurtosis
excess kurtosis
effective sample size
```

A useful deterministic derived-view anchor is:

```text
observations:
    0
    2

alpha:
    0.5
    0.5
```

Expected initialized state after the second observation:

```text
mean = 1
m2 = 1
m3 = 0
m4 = 1
weightSquareSum = 0.5
```

Expected derived views:

```text
variance = 1
correctionDenominator = 0.5
correctedVariance = 2

sigma = 1
correctedSigma = sqrt(2)

skewness = 0
kurtosis = 1
excessKurtosis = -2

effectiveSampleSize = 2
```

After successful TradingView validation, remove the temporary harness.

Do not retain disposable validation code in the production library.

---

# Public functional interfaces

Functional wrappers and persistent state methods are distinct layers.

A functional wrapper may choose current-call output gating that differs from
the retained-state preservation semantics.

Do not infer a new wrapper API solely from the existence of a state type.

Before creating a new functional interface, explicitly decide:

```text
return shape
return ordering
raw vs corrected statistics
missing-call output behavior
reset behavior
public documentation
```

Avoid prematurely freezing unnecessary public tuple contracts.

---

# /1 Alpha / Anchor / Demo architecture

The following decisions are accepted for the final TradingView /1 Alpha /
Anchor / Demo architecture phase. They define the implementation contract;
recording them does not assert that the Demo implementation is complete.

## Anchor Model

Anchor Model is a Demo-level selector, not a new exported exchange/calendar
enum. Its options are:

```text
No Anchor
Rolling Anchor
Minute Anchor
Hour Anchor
Daily Anchor
Weekly Anchor
Monthly Anchor
```

Existing public generic boundary seams remain primary:

```text
anchoredAlpha(contribution, boundaryWhen)
iir1pole(..., resetWhen = boundaryWhen)
marketDispersion(..., anchorWhen = boundaryWhen)
```

A future external time-policy library may supply arbitrary series-bool boundary
events through these interfaces. Exchange/session-specific clocks remain
explicitly deferred.

`anchoredAlpha()` retains its existing volume-normalized semantics. Do not
redesign it.

## Anchor Span semantics

No Anchor contributes alpha zero, causes no estimator reset, and ignores Span.

Rolling Anchor uses non-overlapping blocks of N chart observations, with
boundaries at observation indices `0, N, 2N, 3N, ...`. The anchor observation
is the first observation of the new population. Span = 1 anchors every
observation. Elapsed Time Continuity has no effect on Rolling Anchor.

For Minute / Hour / Day / Week / Month Anchor, Span = N native periods.
Day, Week, and Month refer to the Daily, Weekly, and Monthly selector options.

With Elapsed Time Continuity OFF, count observed native periods. The first
encountered native period has ordinal 0. Each newly observed native period
increments the ordinal once. Missing/unobserved periods do not increment it;
repeated processing observations inside the same native period do not increment
it. Anchor when entering ordinal `N, 2N, 3N, ...`.

With Elapsed Time Continuity ON, use absolute aligned clock/calendar bucket
identities. Anchor whenever the retained bucket identity changes. If time jumps
across one or more empty buckets, perform one reset before the next available
processing observation. Do not synthesize missing observations.

Preserve the existing serial/alignment conventions:

```text
Minute: absolute minute serial
Hour:   absolute hour serial
Day:    trading-day serial
Week:   Monday-aligned week serial
Month:  year * 12 + month - 1
```

## Discrete observation limitation

onlineRecursion does not split or synthesize estimator observations when an
internal time boundary occurs inside one chart bar. A reset applies at the first
available processing observation that can begin the new population. Intrabar
estimator splitting is outside /1 scope.

## Elapsed Time Continuity

Elapsed Time Continuity is a shared temporal-geometry control for time-based
Participation models and time-based Anchor models:

```text
OFF: observed-period continuity
ON:  absolute clock/calendar continuity
```

It does not alter Recursive Decay, No Anchor, Rolling Anchor, Equal
Participation, Rolling Volume, Open-Interest Turnover, Float Turnover, Fund
Turnover, or other non-time Participation models.

Participation settlement timing and Anchor reset timing remain distinct even
when they share serial/bucket arithmetic.

## Composite Alpha

Composite becomes three-component using the existing complementary-retention
algebra. Do not introduce a new combining rule.

```text
compositeAlpha(
    compositeAlpha(recursiveAlpha, participationAlpha),
    anchorAlpha
)

Equivalent: 1 - (1-r)(1-p)(1-a)
```

Neutral component values are:

```text
Recursive Decay Model = Bypass           -> 0
Participation Model   = No Participation -> 0
Anchor Model          = No Anchor        -> 0
```

Each disabled component is an identity element. Missing-input propagation of
`compositeAlpha()` itself remains unchanged. Do not hide anchor initialization
policy inside `compositeAlpha()`.

## Canonical Demo alpha routing

The Demo must produce one canonical estimator coefficient, `selectedAlpha`:

```text
component alphas
    -> sourceAlpha
    -> anchor initialization policy
    -> selectedAlpha
```

The exact same `selectedAlpha` must be supplied to the recursive mean, supplied
to dispersion/band calculations, and available to the lower-pane diagnostic
system. No independent display-alpha recurrence is allowed. No duplicate
estimator should exist only for visualization.

## Demo anchor initialization policy

At the Demo/application layer:

```text
selectedAlpha = estimatorAnchorWhen ? 1.0 : sourceAlpha
```

`estimatorAnchorWhen` is true only when all three conditions hold:

```text
Anchor Model is enabled
AND Alpha Source is Anchored or Composite
AND the selected Anchor boundary occurs
```

This makes the new population's current valid price observation full-weight at
an active estimator reset. It resolves the Demo edge case where dispersion can
force alpha = 1 at an anchor while the outer recursive mean can receive `na`.

This is an application/Demo policy. It does not change `anchoredAlpha()`,
`compositeAlpha()`, generic missing-data behavior outside anchor events,
AdaptiveMoments, HeavyTail, covariance, quantile, expectile, tail mean, Huber,
or regression.

## Demo input organization

Target logical groups:

```text
GENERAL
    Return Model
    Span
    Elapsed Time Continuity

DISPERSION
    Dispersion Model

ALPHA
    Alpha Source
    Recursive Decay Model
    Participation Model
    Anchor Model

DIAGNOSTIC
    Statistic
```

Span is one common positive Demo horizon N; its physical/statistical units
remain model-dependent. Dispersion Model changes band/dispersion behavior.
Alpha Source determines the coefficient-generation policy.

## Lower-pane Diagnostic Statistic

The lower pane must not be reduced permanently to Selected Alpha only. It is a
selectable statistical demonstration/diagnostic. The Demo-level Statistic
selector contains exactly four primary modes:

```text
Selected Alpha
Innovation
Standardized Innovation
Developing Dispersion
```

Only one primary diagnostic series is visibly plotted at a time. Do not
simultaneously overlay quantities with incompatible units/scales. The selector
is Demo-level and does not require a new exported public enum.

## Selected Alpha diagnostic

Selected Alpha plots the exact `selectedAlpha` supplied to the upper recursive
estimators. This is a strict invariant: the lower pane must never display a
different alpha while labeling it as the active/selected coefficient.

Individual component values may remain available as secondary diagnostics,
preferably Data Window only:

```text
Recursive Alpha
Participation Alpha
Anchor Alpha
Raw Composite Alpha
```

They must not be confused visually with Selected Alpha.

## Innovation diagnostic

Innovation demonstrates the new evidence seen by the recursive location process
before the current recursive update. It compares the current location input
with the previous recursive mean, in coordinates compatible with the selected
Return Model:

```text
innovation = current evidence relative to prior retained location,
             expressed in the selected Return Model coordinates
```

Do not blindly use raw `price - mean` when the selected Return Model uses
relative/logarithmic coordinates. Use existing production return/displacement
transformation helpers wherever possible. The implementation phase must recover
and reuse the existing OR transform rather than invent a parallel formula.
Do not introduce a duplicate statistical recurrence.

At first initialization, explicit estimator anchor/reset, or whenever no valid
previous population exists, Innovation should be `na` unless an already
accepted production semantic clearly defines otherwise.

## Standardized Innovation diagnostic

The name is Standardized Innovation. Do not call it a generic "Z-Score":
Dispersion Model need not represent Gaussian sigma. This is a z-score-like
dimensionless diagnostic.

Its numerator is the Innovation defined above. Its denominator is the prior
valid dispersion from the same retained estimator population and in compatible
coordinates:

```text
standardizedInnovation = innovation / priorDispersion
```

This is defined only when Innovation is valid, priorDispersion is valid,
priorDispersion > 0, and no estimator reset starts a new population on the
current observation. Otherwise return `na`.

An anchor/reset observation has no prior scale belonging to the new population,
so Standardized Innovation is `na` on that reset observation. Do not substitute
zero, clamp the score, or add Gaussian assumptions.

## Developing Dispersion diagnostic

Developing Dispersion exposes the exact developing dispersion quantity used by
the upper band/projection construction. The diagnostic observes the production
value already used by the Demo. Do not run a second dispersion estimator for
the oscillator.

## Diagnostic architectural rule

The oscillator observes production recursion. It must not create parallel
statistical state merely to draw diagnostics. Preferred dependencies are:

```text
current evidence + prior location
    -> Innovation

component alphas -> sourceAlpha -> anchor policy -> selectedAlpha
    selectedAlpha -> recursive location
    selectedAlpha -> dispersion -> Developing Dispersion diagnostic
    selectedAlpha -> Selected Alpha diagnostic

Innovation + prior dispersion from the same retained population
    -> Standardized Innovation
```

Diagnostics are algebraic views or observations of the same retained production
state. Selecting a diagnostic does not change estimator state.

## Demo tooltips

Tooltips are required for Return Model, Span, Elapsed Time Continuity,
Dispersion Model, Alpha Source, Recursive Decay Model, Participation Model,
Anchor Model, and Diagnostic Statistic. They must describe actual semantics
rather than marketing prose.

They must explicitly communicate:

```text
Span:
    common positive horizon N; units depend on active model

Elapsed Time Continuity:
    OFF counts observed periods
    ON retains absolute clock/calendar continuity including gaps

Anchor Model:
    Rolling means non-overlapping blocks, not a sliding window

Alpha Source:
    Composite combines enabled Recursive, Participation, and Anchor legs

Anchor initialization:
    active anchor observations use alpha 1 for estimator initialization

Diagnostic Statistic:
    selects one lower-pane statistical view without changing estimator state

Standardized Innovation:
    dimensionless innovation relative to prior valid dispersion;
    not a claim of Gaussian normality
```

## Future external time-policy library

Do not create exchange/session-specific enums in onlineRecursion /1. Do not add
New York open/close policies, CME session policies, London policies, Asia
policies, holiday calendars, or exchange calendars.

A future external time-policy library can emit series-bool boundary events into
the existing generic OR boundary/reset seams listed under Anchor Model. The
built-in Minute/Hour/Day/Week/Month Demo policies are sufficient for /1.

## Closed statistical components during /1 Demo work

Do not reopen or modify any closed statistical component absent a genuine
contradiction required by this contract. In particular, do not change
AdaptiveMoments, HeavyTail, AdaptiveCovariance, AdaptiveQuantile,
AdaptiveExpectile, AdaptiveTailMean, AdaptiveHuber, or regression/beta.

---

# Numerical policy

Do not silently introduce estimator tolerances, clipping rules, epsilon
thresholds, finite-sample corrections, or defensive transformations merely
because they appear numerically convenient.

Distinguish:

```text
mathematical state semantics
model-specific defensive policy
validation tolerance
application-level gating
```

Validation tolerances may be used when comparing floating-point results.

A validation comparison tolerance must not alter estimator state or become an
undocumented estimator parameter.

---

# Python workflow

Python is the primary local reference environment for mathematical validation.

Typical validation sequence:

```text
python -m py_compile <source>
python -m py_compile <test>

python -m pytest -q <targeted-test>
python -m pytest -q
```

Do not add dependencies unless explicitly required and approved.

Prefer deterministic tests over stochastic tests when exact identities are
available.

Experiments may be used for statistical behavior that cannot be fully captured
by deterministic unit tests.

---

# Pine workflow

Pine edits occur in:

```text
PineScript/onlineRecursion.pine
```

TradingView compilation and runtime behavior are authoritative.

When porting an already accepted Python state:

```text
1. preserve semantics exactly
2. follow Pine naming conventions
3. keep implementation mechanically close to accepted recurrence
4. compile in TradingView
5. run deterministic validation
6. remove temporary harness
7. only then proceed to dependent refactors
```

Do not combine unrelated Pine migrations in one validation step.

---

# Codex workflow

Codex CLI may be used for repository inspection, Python implementation,
mechanical refactors, test execution, and Pine source editing.

For statistically sensitive work:

```text
use GPT-5.6 Sol
use default mode unless latency is the primary concern
```

The chat remains responsible for:

```text
architecture
mathematical semantics
experiment design
audit
acceptance decisions
```

Codex is especially useful for:

```text
repository inspection
precise file edits
test execution
diff review
mechanical migrations
```

For new sensitive components, prefer:

```text
inspection-only audit
-> review
-> narrowly authorized implementation
-> tests
-> diff review
-> manual Git commit
```

---

# Codex safety rules

Unless explicitly authorized otherwise, Codex must not:

```text
commit
push
pull
merge
rebase
reset
amend
delete branches
force-update refs
create tags
modify remotes
force checkout
discard changes
```

For first-pass implementation work, Git writes should remain under direct user
control.

Codex may use read-only Git commands such as:

```text
git status
git diff
git log
git branch --show-current
```

when allowed by the task.

---

# Git discipline

Keep commits narrow and semantically meaningful.

Do not mix:

```text
mathematical changes
documentation cleanup
unrelated refactors
formatting churn
release work
```

without a clear reason.

Before commit:

```text
verify intended files
run relevant tests
inspect staged diff
confirm no unrelated modifications
```

For new untracked files, remember that ordinary:

```text
git diff
```

does not show them until staged.

After staging, use:

```text
git diff --cached
git diff --cached --check
git diff --cached --stat
```

before commit.

---

# Shell environment

The user shell is Fish.

All shell instructions must be Fish-compatible.

Avoid Bash-only variable-assignment syntax.

Be careful with unmatched wildcards.

For example, this can fail in Fish when the directory is empty:

```text
rm -rf /tmp/example/*
```

Prefer:

```text
rm -rf /tmp/example
mkdir -p /tmp/example
```

when full directory recreation is appropriate.

---

# Local environment conventions

System packages should preferably be managed through Fedora RPM/DNF where
practical.

Development source trees and build projects belong under:

```text
~/Dev
```

Avoid unnecessary clutter directly under:

```text
$HOME
```

The project repository is:

```text
~/Dev/github/xel-onlineRecursion
```

---

# Component closure rule

Once a statistical component is accepted as mathematically closed, future work
should build above it rather than casually revise it.

Closed components currently include:

```text
adaptive quantile
adaptiveExpectile
adaptiveTailMean
adaptiveHuber
adaptive MAD / robust scale
adaptive covariance / correlation
recursive regression / beta
Python and Pine AdaptiveMomentsState
HeavyTail generic-moment migration
```

Python and Pine `AdaptiveMomentsState` are implemented, validated, and closed.

HeavyTail migration onto the generic moment state is complete and validated.
HeavyTail startup harmonization and a public functional `adaptiveMoments(...)`
remain separate deferred policy/API decisions.

---

# Future generic extensions

Potential future additions should reuse the generic backbone where possible.

Examples:

```text
previous-state z-score
autocorrelation via covariance machinery
CUSUM-style monitoring
other normalized recursive sufficient statistics
```

Do not introduce a new specialized state if an accepted generic sufficient
statistic already represents the required population.

---

# Final guiding rule

When uncertain about architectural ownership, ask:

```text
Is this quantity part of the statistical population,
a derived interpretation of that population,
a model-specific assumption,
or an application-level decision?
```

Then place it accordingly.

The repository should continue to obey:

```text
Foundational state represents the population.
Derived statistics interpret that population.
Models add model-specific assumptions.
Applications decide when evidence is sufficient to act.
```
