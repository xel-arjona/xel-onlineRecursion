# XeL OnlineRecursion agent instructions

## Project purpose

XeL OnlineRecursion is a toolkit for online / streaming statistics, recursive
estimation, and finance-oriented stateful processing.

It is heavily inspired in architectural spirit by public online-statistics and
incremental-learning frameworks such as River, but it is not a River port and
does not target River API compatibility.

Current development release:

```text
1.0.0-rc.2 (2026-09-04)
```

License:

```text
MPL-2.0
```

The project is not yet published as a stable public API.

---

## Repository organization

The repository root is platform-neutral.

Current implementations:

```text
PineScript/
    canonical Pine Script v6 production implementation

Python/
    mathematical reference implementations
    permanent tests
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

---

## Core architectural principles

### 1. Retained state represents a statistical population

State objects should correspond to meaningful retained populations or
sufficient statistics.

Do not create separate state merely because a derived statistic has a distinct
name.

### 2. Generic statistical state belongs in foundational state objects

Generic raw statistical quantities should be represented independently of
higher-level model interpretations.

The intended generic statistical backbone includes:

```text
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
```

### 3. Derived statistics should reuse sufficient state

Whenever mathematically possible, derive statistics algebraically rather than
maintaining redundant recursion.

Examples:

```text
MAD
    -> composition over adaptive quantile state

Regression
    -> derived from AdaptiveCovarianceState

Skewness / kurtosis
    -> derived from AdaptiveMomentsState

Autocorrelation
    -> reuse covariance state where practical

Z-score
    -> composition over retained location and scale
```

### 4. Model interpretations sit above generic statistics

A specialized model should not conceptually own ordinary statistics that are
useful outside that model.

In particular, HeavyTail should ultimately be treated as a higher-level model
interpretation above generic moment statistics rather than as the conceptual
owner of ordinary mean / variance / skewness / kurtosis state.

---

## Statistical semantics

Population semantics are part of the mathematical API.

Distinguish explicitly among:

- cumulative populations
- rolling finite populations
- exponentially weighted populations
- anchored populations
- conditional populations
- observation-clock recursions
- event-clock recursions

Do not silently substitute one population interpretation for another.

Initialization, missing-data behavior, reset timing, coefficient semantics, and
population boundaries must be treated as observable estimator behavior and
tested accordingly.

---

## Validation discipline

Do not modify accepted estimator mathematics without an explicit mathematical
reason.

Python reference work precedes Pine production integration for new or
refactored statistical primitives.

Prefer validation in this order:

```text
1. exact derivation
2. deterministic identity tests
3. invariance / symmetry tests
4. Monte Carlo or distributional validation when useful
5. permanent regression tests
6. Pine integration
7. deterministic Pine validation
```

Never introduce an ad-hoc finite-sample or bias correction merely because it
improves one simulation result.

State the assumptions under which any correction is valid.

Keep raw recursive population moments conceptually distinct from finite-weight
corrected estimators.

---

## Current accepted statistical decisions

The following components are considered mathematically closed unless a genuine
contradiction or defect is discovered:

- adaptive quantile
- adaptive expectile
- adaptive conditional tail mean
- adaptive Huber location
- adaptive MAD / Gaussian-equivalent MAD scale
- adaptive covariance / correlation
- adaptive beta / simple linear regression

Do not redesign these components opportunistically during unrelated work.

---

## AdaptiveMoments refactor policy

The next architectural objective is to introduce a generic:

```text
AdaptiveMomentsState
```

The intended retained state is:

```text
mean
m2
m3
m4
weightSquareSum
```

### Initial Python phase

During the initial AdaptiveMoments Python phase:

- modify only files under `Python/`
- do not modify `PineScript/`
- do not refactor HeavyTail yet
- use the currently accepted HeavyTail central-moment recursion as the numerical
  oracle
- preserve current HeavyTail behavior
- prove observation-by-observation equivalence for:
  - mean
  - m2
  - m3
  - m4

The initial generic derived views should include:

- raw variance
- finite-weight corrected variance
- raw sigma
- finite-weight corrected sigma
- skewness
- kurtosis
- excess kurtosis
- effective sample size

Do not invent finite-sample corrections for skewness or kurtosis during this
phase.

### Absolute innovation

Absolute innovation is not an ordinary central moment.

Do not add it to `AdaptiveMomentsState` without an explicit architectural
decision.

The current HeavyTail absolute-innovation machinery should remain separate
during the first generic-moment refactor.

### Finite-weight concentration

`AdaptiveMomentsState` should track:

```text
weightSquareSum
```

using the same normalized-weight concentration semantics already accepted for
`AdaptiveCovarianceState`.

For deterministic or observation-independent recursive weights under the
appropriate IID population assumptions:

```text
correction denominator
    = 1 - weightSquareSum
```

Raw variance and finite-weight corrected variance must remain distinct.

---

## HeavyTail migration policy

Do not rewrite HeavyTail simultaneously with the initial generic-moment state.

Migration should happen incrementally.

Preferred sequence:

```text
1. build AdaptiveMomentsState in Python

2. prove exact central-moment identity against existing HeavyTail

3. validate derived generic statistics

4. add permanent Python tests

5. port AdaptiveMomentsState to Pine

6. validate Pine moment identity

7. only then refactor HeavyTail onto the generic moment backbone

8. prove HeavyTail public outputs remain unchanged
```

The existing HeavyTail implementation is the migration oracle until the new
generic moment state is independently accepted.

---

## Python workflow

Run Python commands from:

```text
Python/
```

For changed Python source or test files, always run:

```text
python -m py_compile <changed source files>
python -m py_compile <changed test files>
python -m pytest -q <targeted tests>
python -m pytest -q
```

Current pre-AdaptiveMoments full-suite baseline:

```text
161 passed
```

Any unexpected reduction in this count must be investigated.

Prefer simple and transparent mathematical reference implementations over
clever abstraction.

Do not add external Python dependencies unless explicitly approved.

Existing dependencies may be used where they are already part of the project.

---

## Pine Script workflow

The canonical Pine production source is:

```text
PineScript/onlineRecursion.pine
```

Do not edit it during Python-only research tasks.

Pine Script changes require explicit instruction.

TradingView remains the authoritative Pine compiler/runtime environment.

Do not assume that syntactically plausible Pine code compiles.

When adding Pine functionality:

- preserve established naming conventions
- preserve reset-before-current semantics
- preserve missing-observation semantics
- preserve alpha semantics
- use explicit state methods for raw statistical state
- use algebraic methods/functions for derived statistics where possible

Temporary deterministic Pine validation harnesses must be removed after
successful validation.

---

## Public identity

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

TradingView publication revisions use TradingView's own independent versioning
system and must not be conflated with project semantic-version releases.

---

## Licensing

The project uses:

```text
Mozilla Public License 2.0
SPDX-License-Identifier: MPL-2.0
```

Do not replace or modify licensing terms unless explicitly requested.

The repository-root `LICENSE` file is authoritative for project licensing.

Platform-specific publication rules may impose additional requirements beyond
the software license.

---

## Git and repository safety

Unless explicitly requested, do not:

- commit
- push
- pull
- merge
- rebase
- reset
- amend
- delete branches
- force-update refs
- create tags
- modify remotes

Do not modify files unrelated to the assigned task.

Before editing, state which files you intend to modify.

Do not perform broad cleanup opportunistically during a narrowly scoped task.

If a requested change exposes a separate architectural issue, report it rather
than silently expanding scope.

---

## Codex operating policy

Normal Codex work should remain inside the repository workspace.

Do not request unrestricted filesystem access unless the assigned task
genuinely requires it.

During statistical development:

```text
ChatGPT / human review
    architecture
    statistical semantics
    acceptance criteria

Codex
    repository inspection
    implementation
    deterministic testing
    mechanical refactoring
    diff preparation

TradingView
    Pine compilation
    Pine runtime validation
```

Codex must not treat passing tests alone as evidence that altered statistical
semantics are correct.

When mathematics changes, explain the mathematical change explicitly.

---

## Task completion report

At completion of every implementation task, report:

- files changed
- mathematical changes, if any
- API changes, if any
- tests executed
- exact targeted test count
- exact full-suite test count
- unresolved questions
- deviations from requested scope
- whether Pine files were modified

Do not commit or push after reporting unless explicitly instructed.
