# plan2event-holodeck

**Live:** https://plan2event-holodeck.cycls.ai

Attach a venue DXF and describe an event. You get the same DXF back with the
event laid out in it — stages, tents, toilets, food, generators, fences — as
real CAD blocks at real metric coordinates, on their own `EVENT-*` layers so
the client's drawing is never touched.

This repository is **one agent**: the **Holodeck** placement engine,
**v1** variant.

## How it places things

A constraint program, solved. After allenai/Holodeck (CVPR 2024): the model writes relations — `main-stage | central`, `grandstand-1 | near, main-stage | face to, main-stage` — never coordinates. A depth-first solver scores every proved-legal candidate against those relations and backtracks when a branch dead-ends. The only engine with a `constrain` tool.

Method after allenai/Holodeck (Apache-2.0). Measured on a real 7.5 MB venue masterplan with a
25-item programme: **25/25 placed, 0 violations, 4.7 s**.

## v1

One builder pass. The model reads the brief, finds the site, declares the programme and how its pieces relate, and the engine places it.

The v2 of this engine is [plan2event-holodeck-v2](https://github.com/Eli-AMAR/plan2event-holodeck-v2).

## The idea every engine shares

**The model never chooses coordinates.** It decides what the event needs and
how the pieces relate; the engine decides where, choosing only from positions
already proved to fit against the real free-space polygon of the drawing.
Holodeck, LayoutVLM, DirectLayout and I-Design all split the work this way. The
agent it replaces, [agent-2d-to-2d-not](https://github.com/Eli-AMAR/agent-2d-to-2d-not),
did the opposite: it asked an image model to imagine the finished site and read
positions back off the picture.

## Layout

```
agent.py         the agent — deploys exactly this one
core/            DXF reading, free space, the catalogue, placement, the tool
                 surface, the prompt, the model driver
engines/holodeck.py   the placement method
reference/       two files the v2 prompt tells the model to read
assets/          the catalogue: 3 535 usable blocks, normalised to metres
tools/           harness.py, and the scripts that built the catalogue
docs/            transcript.md — the full conversation that built this
CLAUDE.md        the handover: decisions, and the bugs each one cost
```

## Run it

Python 3.12.

```bash
pip install ezdxf "shapely>=2.0" numpy scipy matplotlib 
python tools/harness.py path/to/venue.dxf
```

The harness places a fixed programme with this engine, validates it and writes
`harness_holodeck.png` next to the plan. No model, no deployment, no keys.

## Deploy it

```bash
pip install cycls==0.0.2.140
cp .env.example .env        # fill in the three keys
python agent.py deploy
```

`.env` needs `CYCLS_API_KEY`, `FAL_KEY` (image hosting for the renders shown in
the chat) and `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`).
**Leave `ANTHROPIC_API_KEY` empty.** The model runs through the Claude Code CLI
on a subscription; if the API key is present the CLI silently prefers it and
bills per token.

## All twelve agents

| Agent | Repository | Live |
|---|---|---|
| CP-SAT v1 | [plan2event-cpsat](https://github.com/Eli-AMAR/plan2event-cpsat) | https://plan2event-cpsat.cycls.ai |
| CP-SAT v2 | [plan2event-cpsat-v2](https://github.com/Eli-AMAR/plan2event-cpsat-v2) | https://plan2event-cpsat-v2.cycls.ai |
| Holodeck v1 ← this one | [plan2event-holodeck](https://github.com/Eli-AMAR/plan2event-holodeck) | https://plan2event-holodeck.cycls.ai |
| Holodeck v2 | [plan2event-holodeck-v2](https://github.com/Eli-AMAR/plan2event-holodeck-v2) | https://plan2event-holodeck-v2.cycls.ai |
| Anneal v1 | [plan2event-anneal](https://github.com/Eli-AMAR/plan2event-anneal) | https://plan2event-anneal.cycls.ai |
| Anneal v2 | [plan2event-anneal-v2](https://github.com/Eli-AMAR/plan2event-anneal-v2) | https://plan2event-anneal-v2.cycls.ai |
| Stripe v1 | [plan2event-stripe](https://github.com/Eli-AMAR/plan2event-stripe) | https://plan2event-stripe.cycls.ai |
| Stripe v2 | [plan2event-stripe-v2](https://github.com/Eli-AMAR/plan2event-stripe-v2) | https://plan2event-stripe-v2.cycls.ai |
| Cover v1 | [plan2event-cover](https://github.com/Eli-AMAR/plan2event-cover) | https://plan2event-cover.cycls.ai |
| Cover v2 | [plan2event-cover-v2](https://github.com/Eli-AMAR/plan2event-cover-v2) | https://plan2event-cover-v2.cycls.ai |
| Image round-trip, subscription | [agent-2d-to-2d-not](https://github.com/Eli-AMAR/agent-2d-to-2d-not) | https://agent-2d-to-2d-not.cycls.ai |
| Image round-trip, API | [agent-2d-to-2d](https://github.com/Eli-AMAR/agent-2d-to-2d) | https://agent-2d-to-2d.cycls.ai |

## Why it is built this way

Read `CLAUDE.md` for the state of the work and the traps already paid for —
negative site coordinates that made a solver silently infeasible, a unit
header nobody was reading, an inches-to-feet conversion factor. Read
`docs/transcript.md` for the conversation in which each of those was found.
