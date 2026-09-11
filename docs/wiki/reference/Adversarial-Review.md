# Adversarial Review

`ccflow.ai` is a worked example that uses agents as graph nodes: one writes a poem, two argue
opposing positions about it, and a fourth decides which argued better. It is a complete, runnable
illustration of typed hand-offs, interface-typed upstreams, config groups, and fan-in through the
cache evaluator.

It needs the optional extra:

```bash
pip install ccflow[ai]
```

## Running it

Everything is steered from the command line, one config group at a time:

```bash
python -m ccflow.ai checks=off              # offline, no credentials
python -m ccflow.ai model=openai            # against a real model
python -m ccflow.ai task=counterbalanced    # judge both presentation orders
python -m ccflow.ai spec=haiku rubrics=strict profile=steelman
```

Or load it into the registry, as the other bundled examples do:

```python
from ccflow import ModelRegistry
from ccflow.ai import load_config

load_config(overrides=["checks=off"])
result = ModelRegistry.root()["/task"]()
print(result.verdict.winner, result.verdict.decisive)
```

`checks=off` is needed with the default `model=test` because pydantic-ai's `TestModel` emits
placeholder strings that cannot satisfy a real-quotation check. Enforcement is on by default so
pointing at a real model cannot silently lose it.

## Config groups

| Group       | Options                                                                                                     | Supplies                                            |
| ----------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `model`     | `test`, `openai`, `anthropic`                                                                               | which language model every session uses             |
| `profile`   | `default`, `steelman`                                                                                       | which persona file plays each role                  |
| `agents`    | `poem`, `poem_concise`                                                                                      | the agent session definitions                       |
| `rubrics`   | `poem`, `strict`                                                                                            | critique and arbitration criteria                   |
| `lessons`   | `none`, `house_style`, `producer_only`                                                                      | standing corrections, scoped by role                |
| `spec`      | `sea`, `haiku`                                                                                              | what the poem must be                               |
| `checks`    | `enforced`, `off`                                                                                           | whether mechanical constraints are enforced in code |
| `publisher` | `none`, `json`, `print`                                                                                     | where results go                                    |
| `task`      | `produce`, `advocate`, `arbitrate`, `end_to_end`, `counterbalanced`, `rebuttal`, `rebuttal_counterbalanced` | which nodes run                                     |

Group options are discovered from the filesystem and parametrized in tests, so a new variant is
covered the moment it is added.

## Tasks

| Task                               | Model calls | What it adds                                           |
| ---------------------------------- | ----------- | ------------------------------------------------------ |
| `end_to_end`                       | 4           | the whole graph, writing only the verdict              |
| `counterbalanced`                  | 5           | arbitrates in both orders; a winner only if both agree |
| `rebuttal`                         | 6           | each advocate answers the opposing case first          |
| `rebuttal_counterbalanced`         | 7           | both defences                                          |
| `produce`, `advocate`, `arbitrate` | 1 each      | one node, reading its inputs from published files      |

The single-node tasks chain through the publisher:

```bash
python -m ccflow.ai task=produce   publisher=json
python -m ccflow.ai task=advocate  publisher=json position=pro
python -m ccflow.ai task=advocate  publisher=json position=con
python -m ccflow.ai task=arbitrate publisher=json model=anthropic
```

This is not only convenience. Model output is not reproducible, so re-running the producer hands the
advocates a different poem. Publishing once and reading the file *pins* the artifact, which is what
makes it meaningful to re-judge the same two cases under a different model or rubric.

## Models

| Model                                  | Role                                                                                                            |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `ProduceModel`                         | Writes the poem from a `PoemSpec`.                                                                              |
| `AdvocateModel`                        | Argues one assigned position, grounding each claim in a quotation.                                              |
| `RebuttalModel`                        | Revises a case after reading the opposing one. Returns a `Critique`, so it drops in wherever an advocate would. |
| `ArbitrateModel`                       | Scores the cases and picks a winner.                                                                            |
| `CounterbalancedArbitrateModel`        | Holds two arbiters over the same cases in opposite order and reconciles them.                                   |
| `DebateModel`                          | The task handoff; returns the verdict.                                                                          |
| `AgentSession`                         | The single place that calls a model. Every node composes one.                                                   |
| `PoemFileSource`, `CritiqueFileSource` | Stand in for a live node by reading its published output.                                                       |

Upstreams are typed as interfaces — `PoemProvider`, `CritiqueProvider`, `VerdictProvider` — so a
live node and a file reader are interchangeable. That is what yields the single-node tasks from one
graph definition, with no second copy to drift.

## Contracts

Every edge is a validated pydantic model; no node hands free text to the next.

| Edge               | Contract                 |
| ------------------ | ------------------------ |
| into `produce`     | `PoemSpec`               |
| out of `produce`   | `Poem`                   |
| into an advocate   | `CritiqueBrief[Poem]`    |
| into a rebuttal    | `RebuttalBrief[Poem]`    |
| out of either      | `Critique`               |
| into arbitration   | `ArbitrationBrief[Poem]` |
| out of arbitration | `Verdict`                |

Field descriptions are part of the JSON schema the model is constrained by, so they are written as
instructions rather than as notes to the reader. On a schema violation the node is re-prompted with
the validation error attached — repair, not a blind retry.

## Two kinds of constraint

Mechanical constraints are decided in code, in `ccflow.ai.checks`, never by asking a model. An
arbiter asked whether a poem respected a line budget has to *count*, and a model that miscounts
produces a confident verdict resting on a false premise.

| Check                       | Enforces                                                     |
| --------------------------- | ------------------------------------------------------------ |
| `LinesWithinLimit`          | the poem respects `PoemSpec.max_lines`                       |
| `ClaimsWithinLimit`         | an advocate respects `CritiqueRubric.max_claims`             |
| `EvidenceAppearsInArtifact` | every quoted string occurs in what the advocate was shown    |
| `PositionAsAssigned`        | an advocate reports the side it was assigned                 |
| `VerdictWellFormed`         | scores cover the argued positions; the winner is one of them |

A violation raises `ModelRetry`, so it consumes the same repair budget as a schema failure rather
than adding a second failure path, and all violations are reported at once so one re-prompt can fix
everything.

Judgement criteria — imagery, originality, whether a case is well argued — stay in the rubric, which
is a config group rather than prose buried in a prompt.

## Presentation neutrality

An arbiter can be swayed by *how* a case is presented rather than by what it argues. Replaying a
fixed set of cases with only the presentation varied moved the outcome: most decisions followed
whichever case was shown first, and naming the positions shifted them again.

Three consequences are built in:

- `Critique.confidence` is withheld from the arbitration and rebuttal briefs. A self-rating is a
  claim about a case, not evidence for it.
- Positions are replaced with neutral slot labels while judging and restored afterwards
  (`ArbitrateModel.blind_positions`, on by default).
- The two advocate personas are symmetric. Giving one side a rule the other lacks is a thumb on the
  scale that is very hard to see in an output.

Order is not addressed by any of those, which is what `task=counterbalanced` is for. It costs one
extra call rather than a second graph, because both arbiters reference the same advocates.

## Fan-in is a correctness property

`produce` is referenced by both advocates and by the arbiter, and runs once because `base.yaml`
configures a graph evaluator and a memory cache evaluator. Were it to run three times, the advocates
would argue about *different* poems while the arbiter judged a third — and because model output is
not reproducible, nothing would look wrong. There is a test asserting the shared reference rather
than trusting it.

## Personas and lessons

Personas are markdown files in `ccflow/ai/personas`, selected by the `profile` group. Set
`CCFLOW_AI_AGENTS_PATH` to prepend your own directories without forking the package.

`Lessons` carries standing corrections between runs, scoped by role and injected into a session's
instructions:

```yaml
lessons:
  general: [Plain words beat ornamental ones.]
  by_role:
    producer: ["Do not use these words: seam, stitch, weave, ..."]
    arbiter: [Fluency is not argument.]
```

A session's `role` is deliberately separate from its `profile_name`, so swapping personas
(`profile=steelman` renames them to `steelman` and `skeptic`) cannot silently drop role-scoped
lessons.

Lessons are guidance, not enforcement. Anything decidable mechanically belongs in `checks`, where it
is measured rather than asked for.

## What is not enforced

- Semantic quality is judgement, by design. Nothing decides whether the imagery is good.
- A quotation can be real but irrelevant. `EvidenceAppearsInArtifact` proves a string occurs, not
  that it supports the claim.
- A verdict is one draw. Counterbalancing removes presentation order as a confound, not sampling
  noise.
