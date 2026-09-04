# XeL OnlineRecursion agent instructions

## Project identity

**XeL OnlineRecursion** is a toolkit for online / streaming statistics,
recursive estimation, and finance-oriented stateful processing.

The project is heavily inspired in architectural spirit by public
online-statistics and incremental-learning frameworks such as River, but it is
an independent implementation and does not target River API or behavioral
compatibility.

Current development release:

```text
1.0.0-rc.2 (2026-09-04)
```

License:

```text
Mozilla Public License 2.0
SPDX-License-Identifier: MPL-2.0
```

The project is currently in pre-release architectural hardening. Its public API
has not yet been declared stable.

---

## Repository organization

The repository root is platform-neutral.

Current implementations:

```text
PineScript/
    canonical Pine Script v6 production implementation

Python/
    mathematical reference implementations
    permanent semantic tests
    statistical experiments
    retained validation evidence
```

Future platform ports must remain isolated in their own top-level directories,
for example:

```text
Quantower/
NinjaTrader/
Rust/
```

Do not mix platform-specific implementation files across these directories.

The canonical Pine production source is:

```text
PineScript/onlineRecursion.pine
```

The Python reference and validation environment is rooted at:

```text
Python/
```

---

# Architectural principles

## 1. Retained state represents a statistical population

A state object should correspond to a meaningful retained population or set of
sufficient statistics.

Do not create separate retained state merely because a derived statistic has a
different name.

The preferred architecture is:

```text
observation stream
        |
        v
generic population state
        |
   +----+----+
   |         |
   v         v
derived     model
statistics interpretations
   |         |
   +----+----+
        |
        v
application / trading logic
```

The layers have different responsibilities and must not be conflated.

---

## 2. Foundational state owns mathematical population semantics

A foundational statistical state should represent its population according to
the mathematical definition of that population.

It must not contain arbitrary application-level startup rules merely to make a
derived statistic look mature sooner or later.

Examples of policies that generally do **not** belong in foundational state:

```text
minimum number of bars
minimum effective sample size
burn-in periods
warm-up suppression
confidence thresholds
signal gating
trading-session maturity rules
ignore-first-N-observations behavior
```

Those belong to derived statistics, model interpretations, or application
logic when mathematically or operationally appropriate.

---

## 3. Initialization is population creation

Initialization and recursive transition are distinct concepts.

For a normalized recursively weighted population, the first valid observation
creates the initial normalized population.

The recursive alpha coefficient controls transitions between an already
existing population and a new observation.

It does not determine whether the initial valid observation exists.

Do not mechanically apply an active-state transition equation to an empty
population unless the mathematical population definition specifically requires
that interpretation.

---

## 4. Derived statistics should reuse sufficient state

Whenever mathematically possible, derive statistics algebraically rather than
maintaining redundant recursive state.

Examples:

```text
MAD
    -> composition over adaptive quantile state

Regression
    -> derived from AdaptiveCovarianceState

Skewness / kurtosis
    -> derived from AdaptiveMomentsState

Autocorrelation
    -> reuse covariance machinery where practical

Z-score
    -> composition over retained location and scale
```

A new state object should be introduced only when a genuinely different
population or additional sufficient statistics are required.

---

## 5. Model interpretations sit above generic statistics

A specialized statistical model should not conceptually own ordinary
statistics that are useful independently of that model.

In particular, HeavyTail should ultimately become a model interpretation above
generic moment statistics rather than remain the conceptual owner of ordinary:

```text
mean
variance
skewness
kurtosis
```

Model-specific quantities may remain model-specific.

Examples include:

```text
Student-t degrees of freedom
Student-t variance-equivalent scale
absolute-innovation comparison scale
```

---

# Statistical population semantics

Population geometry is part of the mathematical API.

Distinguish explicitly among:

```text
cumulative populations
finite rolling populations
exponentially weighted populations
anchored populations
conditional populations
observation-clock recursions
event-clock recursions
```

These interpretations are not interchangeable.

Initialization, missing-observation behavior, reset timing, coefficient
semantics, and population boundaries are observable estimator behavior and must
be documented and tested.

---

# Validation discipline

Do not modify accepted estimator mathematics without an explicit mathematical
reason.

Python reference work precedes Pine production integration for new or
refactored statistical primitives.

Prefer validation in this order:

```text
1. define exact mathematical semantics

2. implement a transparent Python reference

3. establish deterministic identities

4. establish invariance / symmetry properties

5. perform Monte Carlo or distributional validation when useful

6. add permanent semantic regression tests

7. integrate the accepted primitive into Pine Script

8. perform deterministic Pine validation

9. freeze accepted estimator mathematics
```

Passing tests alone is not sufficient evidence that changed statistical
semantics are correct.

Never introduce an ad-hoc finite-sample or bias correction merely because it
improves one simulation result.

Any correction must have an explicit mathematical interpretation and its
assumptions must be stated.

Keep raw recursive population quantities conceptually distinct from corrected
estimators or model interpretations.

---

# Mathematically closed components

The following components are considered mathematically closed unless a genuine
contradiction, derivation error, or implementation defect is discovered:

```text
adaptive quantile
adaptive expectile
adaptive conditional tail mean
adaptive Huber location
adaptive MAD / Gaussian-equivalent MAD scale
adaptive covariance / correlation
adaptive beta / simple linear regression
```

Do not redesign these components opportunistically during unrelated work.

---

# Generic statistical backbone

The intended generic retained statistical backbone includes:

```text
AdaptiveMomentsState
    mean
    m2
    m3
    m4
    weightSquareSum
```

and:

```text
AdaptiveCovarianceState
    meanX
    meanY
    varianceX
    varianceY
    covariance
    weightSquareSum
```

These objects represent generic recursively weighted statistical populations.

Higher-level statistics should be algebraic views whenever possible.

---

# AdaptiveMomentsState contract

## Purpose

`AdaptiveMomentsState` represents a normalized exponentially weighted
univariate empirical population.

Its retained state is:

```text
mean
m2
m3
m4
weightSquareSum
```

where:

```text
m2
    recursively weighted second central moment

m3
    recursively weighted third central moment

m4
    recursively weighted fourth central moment

weightSquareSum
    sum of squared normalized recursive observation weights
```

The state itself must remain model-neutral.

It must not contain Student-t-specific interpretation, application warmup, or
absolute-innovation state.

---

## Foundational initialization semantics

Initialization is population creation, not an application of the recursive
alpha transition.

For the first valid observation and valid alpha, after alpha is clamped to:

```text
[0, 1]
```

the state becomes the unique normalized singleton population:

```text
mean             = input
m2               = 0
m3               = 0
m4               = 0
weightSquareSum  = 1
```

This initialization occurs regardless of the clamped alpha, including:

```text
alpha = 0
```

Therefore:

```text
empty state
+ valid input
+ alpha = 0

    -> initialize singleton population
```

Alpha controls transitions between existing populations. It is not an
observation-admission gate for the first valid observation.

---

## Active-state alpha semantics

Once the state is initialized:

```text
alpha = 0
    preserve the complete active population exactly

0 < alpha < 1
    perform the accepted adaptive central-moment recursion

alpha = 1
    replace the retained population with the current singleton observation
```

For `alpha = 1` the resulting state is therefore:

```text
mean             = current input
m2               = 0
m3               = 0
m4               = 0
weightSquareSum  = 1
```

---

## Missing observations

Missing input or missing alpha preserves the complete retained population.

No field should partially update when the current observation cannot be
processed.

In particular:

```text
missing input
    -> preserve state

missing alpha
    -> preserve state
```

---

## Reset semantics

Reset clears the retained population before processing the current observation.

Therefore:

```text
reset
+ valid current input
+ valid current alpha

    -> current observation initializes a new singleton population
```

This remains true when:

```text
alpha = 0
```

Reset is a population-boundary operation.

It must not be implemented as an application-level warmup rule.

---

## Weight concentration

`AdaptiveMomentsState` tracks recursive normalized-weight concentration through:

```text
weightSquareSum
```

For an active state:

```text
S2_new
    = (1 - alpha)^2 * S2_old
    + alpha^2
```

where:

```text
S2 = weightSquareSum
```

The singleton initialization is:

```text
S2 = 1
```

For constant alpha in a continuing stationary recursion:

```text
S2 -> alpha / (2 - alpha)
```

and therefore:

```text
Neff -> (2 - alpha) / alpha
```

---

# AdaptiveMomentsState derived views

The initial generic derived statistics should include:

```text
variance
correctedVariance

sigma
correctedSigma

skewness

kurtosis
excessKurtosis

effectiveSampleSize
```

These should be algebraic views of retained state.

Do not introduce additional retained state merely to cache these quantities
unless a demonstrated implementation constraint requires it.

---

## Raw variance

Raw recursively weighted population variance is:

```text
variance = m2
```

For a valid singleton population:

```text
variance = 0
```

This is a valid raw population quantity.

---

## Finite-weight corrected variance

Under the appropriate IID paired-observation / recursively weighted population
assumptions, with deterministic weights or weights independent of the observed
values, the accepted finite-weight correction denominator is:

```text
1 - weightSquareSum
```

Therefore:

```text
correctedVariance
    = m2 / (1 - weightSquareSum)
```

when:

```text
1 - weightSquareSum > 0
```

For a singleton:

```text
weightSquareSum = 1
```

so corrected variance is mathematically undefined.

Do not alter the raw singleton population merely to force corrected variance to
exist.

---

## Sigma

Raw sigma is:

```text
sigma = sqrt(m2)
```

when the raw variance is valid and non-negative.

For a singleton:

```text
sigma = 0
```

Corrected sigma is:

```text
correctedSigma = sqrt(correctedVariance)
```

when corrected variance is mathematically defined.

---

## Skewness

Generic raw moment skewness is:

```text
skewness
    = m3 / m2^(3/2)
```

when:

```text
m2 > 0
```

It is undefined for zero-variance populations.

Do not invent a finite-sample correction during the initial
`AdaptiveMomentsState` implementation.

---

## Kurtosis

Generic Pearson moment kurtosis is:

```text
kurtosis
    = m4 / m2^2
```

when:

```text
m2 > 0
```

Excess kurtosis is:

```text
excessKurtosis
    = kurtosis - 3
```

No finite-sample skewness or kurtosis correction is part of the initial generic
moment state.

---

## Effective sample size

Effective sample size is:

```text
effectiveSampleSize
    = 1 / weightSquareSum
```

when:

```text
weightSquareSum > 0
```

For a singleton:

```text
effectiveSampleSize = 1
```

Effective sample size is a property of recursive weight concentration. It is
not a substitute for application-level warmup policy.

---

# Separation between raw state and higher abstractions

The foundational state must remain mathematically valid even when some derived
statistics are not yet defined.

For example, immediately after singleton initialization:

```text
mean
    defined

variance
    0

sigma
    0

effectiveSampleSize
    1

correctedVariance
    undefined because 1 - weightSquareSum = 0

skewness
    undefined because m2 = 0

kurtosis
    undefined because m2 = 0
```

This is expected behavior.

Do not corrupt or delay population initialization merely because a higher-order
derived statistic is not yet defined.

A higher abstraction may independently decide that it requires, for example:

```text
Neff >= threshold
variance > threshold
minimum history
model-specific convergence
```

before exposing or acting on a result.

Such rules must not modify the generic retained population semantics.

---

# HeavyTail oracle and migration policy

## Existing HeavyTail role

The current HeavyTail implementation contains an already accepted recursive
central-moment engine.

During the initial `AdaptiveMomentsState` implementation, that existing engine
is the numerical oracle for:

```text
mean
m2
m3
m4
```

Do not refactor HeavyTail during the first AdaptiveMoments phase.

---

## Known alpha-zero boundary difference

The current HeavyTail implementation has a historical behavior in which:

```text
empty HeavyTail state
+ valid input
+ alpha = 0

    -> remains empty
```

That behavior must **not** be copied into generic `AdaptiveMomentsState`.

The generic state follows the normalized-population initialization semantics
defined above:

```text
empty AdaptiveMomentsState
+ valid input
+ alpha = 0

    -> singleton population
```

This is an intentional semantic difference.

It must be explicitly tested rather than treated as a numerical discrepancy.

---

## Identity requirement against HeavyTail

For trajectories in which both states have been initialized consistently,
`AdaptiveMomentsState` must reproduce the accepted HeavyTail central-moment
recursion observation by observation for:

```text
mean
m2
m3
m4
```

This identity must be tested for:

```text
fixed alpha
variable alpha
alpha in (0, 1)
active-state alpha = 0
active-state alpha = 1
missing observations where applicable
```

The known empty-state `alpha = 0` startup difference is excluded from direct
identity because the population semantics intentionally differ there.

---

## HeavyTail migration sequence

Do not rewrite HeavyTail simultaneously with the initial generic moment state.

Preferred sequence:

```text
1. implement AdaptiveMomentsState in Python

2. prove deterministic central-moment identity against existing HeavyTail

3. validate generic derived statistics

4. add permanent Python tests

5. port AdaptiveMomentsState to Pine

6. validate Pine moment identity

7. only then refactor HeavyTail onto the generic moment backbone

8. prove HeavyTail public outputs remain unchanged except for any explicitly
   approved pre-release semantic cleanup
```

The existing HeavyTail implementation remains the migration oracle until the
new generic moment state is independently accepted.

---

# Absolute innovation

Absolute innovation is not an ordinary central moment.

Do not add it to:

```text
AdaptiveMomentsState
```

without an explicit architectural decision.

The current HeavyTail absolute-innovation machinery should remain separate
during the first generic-moment refactor.

Possible future placement should be evaluated as either:

```text
model-specific HeavyTail state
```

or:

```text
a separate generic robust / absolute-deviation population primitive
```

Do not decide this opportunistically during the initial moments extraction.

---

# Python workflow

Run Python commands from:

```text
Python/
```

The current pre-AdaptiveMoments full-suite baseline is:

```text
161 passed
```

This is the semantic regression baseline established before the generic moments
refactor.

Any unexpected reduction must be investigated.

For changed Python source and test files, run:

```text
python -m py_compile <changed source files>
python -m py_compile <changed test files>

python -m pytest -q <targeted tests>

python -m pytest -q
```

Report exact test counts.

Prefer simple and transparent mathematical reference implementations over
clever abstraction.

Do not add external Python dependencies unless explicitly approved.

Do not modify accepted unrelated estimators while implementing
`AdaptiveMomentsState`.

---

# Pine Script workflow

The canonical Pine production source is:

```text
PineScript/onlineRecursion.pine
```

TradingView remains the authoritative Pine compiler and runtime environment.

Do not edit Pine during Python-only research tasks.

Pine changes require explicit instruction.

Do not assume that code which appears syntactically plausible compiles in Pine.

When adding Pine functionality:

```text
preserve explicit population semantics
preserve reset-before-current semantics
preserve missing-observation semantics
preserve coefficient semantics
prefer generic raw state plus derived algebraic methods
avoid redundant retained state
```

Temporary deterministic Pine validation harnesses must be removed after
successful validation.

---

# Public identity

Project / documentation name:

```text
XeL OnlineRecursion
```

Repository name:

```text
xel-onlineRecursion
```

Canonical Pine source:

```text
onlineRecursion.pine
```

Technical Pine library identifier:

```text
onlineRecursion
```

Current project release:

```text
1.0.0-rc.2 (2026-09-04)
```

TradingView publication revisions use TradingView's independent library
versioning system and must not be conflated with project semantic-version
releases.

The GitHub project release serves as the external project/version reference.

---

# Licensing

The project uses:

```text
Mozilla Public License 2.0
SPDX-License-Identifier: MPL-2.0
```

The repository-root:

```text
LICENSE
```

file is authoritative.

Do not replace or modify licensing terms unless explicitly requested.

Platform-specific publication rules may impose requirements beyond the software
license.

---

# Git and repository safety

Unless explicitly requested, an automated coding agent must not:

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
```

Do not modify files unrelated to the assigned task.

Before editing, state which files are intended to change.

Do not perform broad cleanup opportunistically during a narrowly scoped task.

If a requested change exposes a separate architectural issue, report it rather
than silently expanding scope.

---

# Codex operating policy

Normal Codex work should remain inside the repository workspace.

Do not request unrestricted filesystem access unless the assigned task
genuinely requires it.

The normal division of responsibility is:

```text
ChatGPT / human review
    architecture
    mathematical semantics
    experiment design
    acceptance criteria
    audit and final approval

Codex
    repository inspection
    Python implementation
    deterministic testing
    mechanical refactoring
    diff preparation

TradingView
    Pine compilation
    Pine runtime validation
```

Codex must not make statistical-design decisions merely because one
implementation is easier.

When mathematical behavior changes, the change must be reported explicitly.

Passing tests do not authorize a semantic change that was not requested.

---

# Task scope for the first AdaptiveMoments implementation

The first implementation task is Python-only.

It must:

```text
create a generic AdaptiveMomentsState

reuse the accepted HeavyTail central-moment recursion

implement the initialization semantics defined in this document

track weightSquareSum

provide the initial generic derived views

add permanent deterministic tests

prove identity against the existing HeavyTail moment engine

preserve all existing accepted Python behavior
```

It must **not**:

```text
modify PineScript/

refactor HeavyTail

change accepted HeavyTail public behavior

add absoluteInnovation to AdaptiveMomentsState

introduce finite-sample skewness correction

introduce finite-sample kurtosis correction

change unrelated estimators

commit or push automatically
```

---

# Task completion report

At completion of every implementation task, report:

```text
files changed

mathematical changes, if any

population-semantic changes, if any

API changes, if any

tests executed

exact targeted test count

exact full-suite test count

unresolved questions

deviations from requested scope

whether Pine files were modified
```

Do not commit or push after reporting unless explicitly instructed.

---

# Guiding rule

When implementation convenience and statistical meaning conflict, preserve the
statistical meaning.

Foundational state represents the population.

Derived statistics interpret that population.

Models add model-specific assumptions.

Applications decide when evidence is sufficient to act.
