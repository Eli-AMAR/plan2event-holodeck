"""Engine: Holodeck.

After allenai/Holodeck (Apache-2.0, CVPR 2024) — the DSL in
`ai2holodeck/generation/prompts.py` and the solver in
`ai2holodeck/generation/floor_objects.py`, class `DFS_Solver_Floor`.

Holodeck's central claim, and the whole reason it is worth copying here: the
language model must not emit coordinates. It emits a constraint program — one
pipe-delimited line per object, naming a global position and a list of
relations to objects already introduced — and a solver turns that into metres.
A model asked for a number invents a plausible one; a model asked for a
relation states an intention, and an intention can be checked, scored and
argued with. `CONSTRAINT_HELP` below is the vocabulary as the model sees it and
`parse` is the gate; `solve` is the solver.

The solver is Holodeck's, step for step:

    grid the ground into candidate placements
    -> drop the ones that collide with what is already standing
    -> apply the global constraint as a hard filter
    -> let every remaining relation add a weighted bonus to each survivor
    -> sort by that score and descend it depth-first, backtracking on a dead end
    -> keep the best layout the descents reached, not the first

Largest object first, and a line may only refer to objects introduced above it,
so the frame every relation is measured in is already fixed when it is read.

Two things about that last step changed for a site 360 m long, and both were
changes the site forced rather than improvements looked for:

  * `DFS_Solver_Floor.dfs` collects every path it reaches and picks the best
    with `get_max_solution`. Exhausting the branch tree costs nothing for eight
    objects in a room; for twenty-five on a masterplan it is 3^25, and
    depth-first the whole budget goes on permuting the last few bins while the
    stage never moves. So the restart is over the choice that decides the
    plan — where the anchor stands — and each of a handful of well-separated
    anchor positions gets its own descent.

  * Holodeck ranks those layouts by the summed placement weight. That does not
    survive the change of scale either: twenty-five smooth scores mean a metre
    gained on six shade sails outvotes three grandstands losing sight of the
    stage, and the layout it preferred was the one a planner would throw out.
    A relation is not a smooth quantity — the brief said two things stand a
    certain way to each other and they either do or they do not — so descents
    are ranked on statements kept, with the summed weight breaking ties. See
    `_kept`, which is also what lets the engine NAME the relations it dropped.

What changed for an event site, because an indoor room's vocabulary does not
survive the move outdoors:

    edge        the site perimeter, not a wall
    north/south/east/west   added: a brief says "the water is north" far more
                often than it says "against the wall"
    central     added, for the anchor object — a main stage belongs in the
                middle of its audience, which is a different instruction from
                Holodeck's "middle", meaning merely "not against a wall"
    near / far  40 m, not 1.5 m. See NEAR_M.
    around      dropped. Chairs round a table has no event analogue.

Two constraint systems meet in this engine and they are deliberately not the
same strength. A relation in the DSL is the brief's *preference* and scores;
`Item.far` with `Item.min_far` is the brief's *rule*, is what `validate`
measures, and is a hard filter here. Holodeck scores everything because a
living room has no regulations. An event has a fire officer.
"""

from __future__ import annotations

import math
import random
import time

from core.placement import Ground, Placement, anchors

NAME = "holodeck"
DOC = ("Constraint-program layout: the model writes relations (near, far, in "
       "front of, edge, north, face to ...) instead of coordinates, and a "
       "depth-first solver with backtracking turns them into metres. Best "
       "when the brief is spatial prose rather than a packing problem.")

# Event scale, not furniture scale. Holodeck calls 0.5-1.5 m "near" because it
# is arranging a living room; on a 360 m site the same word means "the same
# part of the site", and a toilet 1.5 m from a food truck is a health notice
# rather than an adjacency. These two constants are the only numbers that
# change what the published vocabulary MEANS, so they are named and stated in
# CONSTRAINT_HELP verbatim.
NEAR_M = 40.0            # "near, X": within this many metres, edge to edge
FAR_M = 40.0             # "far, X": at least this many metres, edge to edge

EDGE_BAND = 1.5          # "edge": within this multiple of the object's short side
ALIGN_M = 10.0           # "center aligned" is fully satisfied inside this offset
LATERAL_M = 15.0         # frontal band half-width added to the target's own size

# `Item.near` is the programme's own adjacency wish rather than a line of the
# DSL, and `core.placement.score` reads it as met at 60 m of centre distance.
# That is the project's definition of the wish, so it is the one used here.
WISH_M = 60.0

STEP_DIV = 200           # anchor grid pitch = site span / this
CAND_CAP = 700           # candidate placements kept per item after thinning
BRANCH = 3               # candidates tried per item before the search retreats
NODES = 40000            # ceiling on search nodes per descent, whatever the clock says
ROOTS = 6                # anchor positions tried, each its own descent
ROOT_DIV = 10            # two roots closer than span/this describe the same plan

# Relation weights. Ordering matters more than the absolute values: a hard rule
# is a filter, so everything here is a preference competing with other
# preferences, and the numbers say which preference wins a tie.
W = {
    "near": 10.0, "far": 8.0,
    "in front of": 8.0, "side of": 6.0, "left of": 6.0, "right of": 6.0,
    "center aligned": 7.0, "face to": 6.0,
    "middle": 4.0, "central": 10.0, "compass": 5.0, "edge": 5.0,
}
W_KIND = 5.0             # cohesion: food trucks form a village
W_ITEM_NEAR = 9.0        # Item.near, the programme's own adjacency wish
W_ANCHOR = 3.0           # fallback pull toward the anchor object
W_PACK = 1.5             # mild pull toward the centre of what is already placed

GLOBALS = ("edge", "middle", "central", "north", "south", "east", "west")
COMPASS = ("north", "south", "east", "west")
RELATIONS = ("near", "far", "in front of", "side of", "left of", "right of",
             "center aligned", "face to")

CONSTRAINT_HELP = f"""\
Do not give coordinates. Write one line per object:

    object | global constraint | constraint 1 | constraint 2 | ...

The object is the key of an item in the programme. The global constraint is
optional and there is at most one; the rest are relations, each written as
"verb, object".

GLOBAL — where on the site the object sits
    edge        against the site perimeter (hard: it will be placed there or
                the line is reported as unsatisfiable)
    middle      away from the perimeter, anywhere inside
    central     at the centre of the site — for the one object the layout is
                built around, usually the main stage
    north | south | east | west
                in that half of the site

RELATION — where the object sits relative to another object
    near, X             within {NEAR_M:g} m of X, edge to edge
    far, X              at least {FAR_M:g} m from X, edge to edge
    in front of, X      in the ground X faces
    side of, X          beside X rather than in front of or behind it
    left of, X          on X's left, as X faces
    right of, X         on X's right, as X faces
    center aligned, X   on X's centre line
    face to, X          turn this object to look at X

An object may only be named by a line BELOW its own — a relation is measured
in the target's frame, so the target has to be placed first. Order the lines
the way you would build the site: the anchor first, then what hangs off it.

    main-stage    | central
    grandstand-1  | middle | in front of, main-stage | center aligned, main-stage
    grandstand-2  | middle | near, main-stage | left of, main-stage | face to, main-stage
    food-1        | edge | far, main-stage
    wc-f-1        | edge | near, food-1

"face to" is a relation like every other one and obeys the same rule: it reads
the target's position, so the target must already stand. An anchor object has
nothing above it to face, and it does not need one — an object with no "face
to" looks inward, across the site.
"""


# ── the constraint program ───────────────────────────────────────────────

def parse(text, items_by_key):
    """Read a constraint program. -> ({key: [constraint]}, [error]).

    Holodeck's own parser is lenient — it drops what it cannot read and lays
    the scene out anyway. That is the wrong trade here. A dropped constraint
    does not read to the user as a bad line; it reads as a solver that ignored
    the brief, and the brief is the part they wrote themselves. So an unknown
    verb, an unknown object or a forward reference is an error that names the
    line, the valid constraints on that line are still kept, and the caller is
    expected to show the errors rather than swallow them.
    """
    kinds = {}
    for k, it in items_by_key.items():
        kinds.setdefault(it.kind, []).append(k)

    # An item the brief has already pinned is geometry before the program
    # starts running, so it may be referred to from the very first line.
    seen_keys = {k for k, it in items_by_key.items() if it.fixed}
    seen_kinds = {items_by_key[k].kind for k in seen_keys}

    out, errors = {}, []
    for n, raw in enumerate(str(text or "").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        key = parts[0]
        if key not in items_by_key:
            errors.append(f"line {n}: {key!r} is not in the programme — {line!r}")
            continue

        for field in parts[1:]:
            if not field:
                continue
            head, comma, target = field.partition(",")
            head = " ".join(head.lower().split())
            target = target.strip()

            if not comma:
                if head in GLOBALS:
                    out.setdefault(key, []).append(
                        {"op": head, "target": None, "line": n})
                elif head in RELATIONS:
                    errors.append(f"line {n}: {head!r} needs an object, as "
                                  f"\"{head}, <object>\" — {line!r}")
                else:
                    errors.append(f"line {n}: {field!r} is not a constraint — "
                                  f"{line!r}")
                continue

            if head not in RELATIONS:
                if head in GLOBALS:
                    errors.append(f"line {n}: {head!r} is a global constraint "
                                  f"and takes no object — {line!r}")
                else:
                    errors.append(f"line {n}: {head!r} is not a constraint — "
                                  f"{line!r}")
                continue
            if target == key:
                errors.append(f"line {n}: {key!r} cannot be {head} itself — {line!r}")
                continue
            if target in seen_keys or target in seen_kinds:
                out.setdefault(key, []).append(
                    {"op": head, "target": target, "line": n})
            elif target in items_by_key or target in kinds:
                errors.append(f"line {n}: {head}, {target} refers forward — "
                              f"{target} has to be placed on an earlier line")
            else:
                errors.append(f"line {n}: {target!r} is not in the programme — "
                              f"{line!r}")

        seen_keys.add(key)
        seen_kinds.add(items_by_key[key].kind)
    return out, errors


# ── geometry the search runs on ──────────────────────────────────────────

class _Put:
    """A placed object as the filters want to see it: an axis-aligned extent,
    the rules it carries, and the direction it looks in.

    The extent is the bounding box of the rotated footprint, so a candidate
    turned to something other than a right angle is over-, never under-,
    estimated — a conservative collision test can only refuse a legal layout,
    while an optimistic one ships an illegal one."""
    __slots__ = ("key", "kind", "x", "y", "rot", "w", "h", "item", "facing")

    def __init__(self, item, x, y, rot, facing=0.0):
        self.key, self.kind, self.item = item.key, item.kind, item
        self.x, self.y, self.rot = x, y, rot
        self.w, self.h = item.size(rot)
        self.facing = facing


def _gap(a, ax, ay, aw, ah, b):
    """Edge-to-edge distance between two axis-aligned extents; 0 if they touch.

    Centre distance is not edge distance, and every rule in this project is
    written edge to edge — separating a 12 m toilet and a 7 m truck by 20 m of
    centre distance leaves 10 m of ground between them and a violation in the
    report."""
    gx = abs(ax - b.x) - (aw + b.w) / 2
    gy = abs(ay - b.y) - (ah + b.h) / 2
    if gx <= 0.0 and gy <= 0.0:
        return 0.0
    if gx > 0.0 and gy > 0.0:
        return math.hypot(gx, gy)
    return gx if gx > gy else gy


def _separation(item, put):
    """The metres `validate` will insist on between these two.

    Read in both directions. A rule is a property of the pair, so the toilet's
    "20 m from food" binds the food truck too, whichever of them the search
    happens to place first — reading it one way round is how an engine reports
    success on a layout that fails."""
    d = 0.0
    if item.min_far > 0 and (put.key in item.far or put.kind in item.far):
        d = item.min_far
    other = put.item
    if other.min_far > 0 and (item.key in other.far or item.kind in other.far):
        d = max(d, other.min_far)
    return d


def _needs(item, state):
    """What every standing object demands of `item`: [(object, metres, why)].

    Neither the distance nor the reason depends on where `item` ends up, and
    `options` measures up to seven hundred candidate positions against the
    same unchanged `state`. Reading the rules once per item instead of once
    per candidate is the difference between 2.7 million rule lookups on the
    heavy programme and fifty-seven. Objects that demand nothing are dropped
    here rather than skipped inside the loop that matters."""
    out = []
    for p in state:
        need = 0.0
        # An overhead object — a shade sail, a canopy — exists in order to span
        # what is under it, so a sail and the food truck beneath it do not
        # collide. Two sails do: they are at the same height, and `overhead`
        # means "may span the ground", not "is not an object". `validate` is
        # laxer — it exempts a pair as soon as either one is overhead — so the
        # first draft of this line was legal and put all six sails of the
        # programme on the same square of ground.
        if item.overhead == p.item.overhead:
            need = max(item.clearance, p.item.clearance)
        sep = _separation(item, p)
        why = "clearance"
        if sep > need:
            need, why = sep, "separation"
        if need > 0.0:
            out.append((p, need, why))
    return out


def _blocked(x, y, w, h, needs):
    """None if this placement is legal against everything already standing,
    otherwise the reason, for the notes."""
    for p, need, why in needs:
        if _gap(None, x, y, w, h, p) < need:
            return why
    return None


def _band(v, span):
    """1 at zero, 0 at `span`, linear between — the shape every soft term here
    uses, so that weights compare directly against one another."""
    return max(0.0, 1.0 - v / span) if span > 0 else 0.0


# ── what the finished layout keeps ───────────────────────────────────────

def _keeps(put, op, name, others):
    """Is this one relation kept? None when there is nothing to measure it
    against, so an unmeasurable line is neither credited nor blamed."""
    tg = [p for p in others if p.key == name or p.kind == name]
    if not tg:
        return None
    near = min(tg, key=lambda p: _gap(None, put.x, put.y, put.w, put.h, p))
    d = _gap(None, put.x, put.y, put.w, put.h, near)
    f, lat, _ = _frame(near, put.x, put.y)
    across = max(near.w, near.h) / 2 + LATERAL_M
    if op == "near":
        return d <= NEAR_M
    if op == "far":
        return d >= FAR_M
    if op == "in front of":
        return f > 0 and abs(lat) <= across
    if op == "side of":
        return abs(lat) > abs(f)
    if op == "left of":
        return lat > 0
    if op == "right of":
        return lat < 0
    if op == "center aligned":
        return abs(lat) <= ALIGN_M
    if op == "face to":
        # `_facing` already turned the object toward the target; what is left
        # to check is that its broad side is the one presented, within 45°.
        bearing = math.atan2(near.y - put.y, near.x - put.x)
        front = math.radians(put.rot) + (0.0 if put.item.w >= put.item.h
                                         else math.pi / 2)
        return abs(math.cos(bearing - front - math.pi / 2)) >= 0.707
    return None


def _kept(state, cons):
    """How much of the brief the finished layout actually keeps.

    Holodeck ranks its candidate layouts by the sum of the placement weights
    along the path (`get_max_solution`), and that ranking does not survive the
    change of scale. In a room it compares eight objects; here it compares
    twenty-five, and a metre gained on six shade sails outvotes three
    grandstands losing sight of the stage — which is exactly what it did, and
    the layout it preferred was the one a planner would throw out.

    A relation is not a smooth quantity. The brief either said two things
    stand in a certain way to each other and they do, or they do not. So
    descents are ranked on how many statements they keep, and the summed
    weight — still Holodeck's — breaks the ties, of which there are many.

    Global constraints are not counted: `edge` and the compass are hard
    filters already, and any relaxation of one is reported separately, while
    `middle` and `central` are preferences the layout is free to trade away
    for a relation it can keep. Returns (kept, stated, [what was not kept])."""
    kept = stated = 0
    lost = []
    for p in state:
        others = [q for q in state if q.key != p.key]
        for c in cons.get(p.key, ()):
            if c["target"] is None:
                continue
            r = _keeps(p, c["op"], c["target"], others)
            if r is None:
                continue
            stated += 1
            if r:
                kept += 1
            else:
                lost.append(f"{p.key}: \"{c['op']}, {c['target']}\" on line "
                            f"{c['line']} is not kept — no ground on the site "
                            "both fits it and satisfies it")
        for want in p.item.near:
            tg = [q for q in others if q.key == want or q.kind == want]
            if not tg:
                continue
            stated += 1
            d = min(math.hypot(p.x - q.x, p.y - q.y) for q in tg)
            if d <= WISH_M:
                kept += 1
            else:
                lost.append(f"{p.key}: asked to be near {want}, and the "
                            f"nearest one is {d:.0f} m away")
    return kept, stated, lost


# ── the engine ───────────────────────────────────────────────────────────

def solve(items, free, region, fixed=(), items_by_key=None, seconds=20.0,
          seed=0, **_):
    t0 = time.time()
    rng = random.Random(seed)
    notes = []

    index = dict(items_by_key or {})
    index.update({it.key: it for it in items})
    x0, y0, x1, y1 = region
    span = max(x1 - x0, y1 - y0)
    centre = free.centroid if not free.is_empty else None
    cx, cy = (centre.x, centre.y) if centre else ((x0 + x1) / 2, (y0 + y1) / 2)

    cons = {}
    for it in items:
        got = getattr(it, "constraints", None)
        if got:
            cons[it.key] = list(got)

    # The ground the search always starts from: placements handed in by the
    # caller, plus anything the brief pinned. Every descent is rewound to it.
    base_state = _reset_state(fixed, items, index, cx, cy)
    pinned, todo = [], []
    for it in items:
        if it.fixed:
            fx, fy, rot = (list(it.fixed) + [0.0])[:3]
            pinned.append(Placement(it.key, it.block, float(fx), float(fy),
                                    float(rot), label=it.label))
        else:
            todo.append(it)
    if not todo:
        return pinned, notes + [
            f"every one of the {len(pinned)} items was pinned by the brief"
            if pinned else "nothing to place"]

    order = _order(todo, cons, index, {p.key for p in base_state}, notes)
    anchor_key = order[0].key
    if cons:
        notes.append(f"constraint program over {len(cons)} of {len(items)} items, "
                     f"{sum(len(v) for v in cons.values())} constraints")

    step = max(1.0, round(span / STEP_DIV, 1))
    # One raster of the ground, shared by every `anchors` call below. Omitting
    # it is what `anchors` documents as the expensive path, and it rebuilds the
    # same summed-area table once per distinct footprint: on the heavy
    # programme's eleven footprints that was 2.4 s of the budget spent
    # rasterising an unchanging polygon eleven times, before the search had
    # taken a single decision. It is also most of the reason a starved clock
    # used to degrade so unevenly.
    ground = Ground(free, step)
    cache = {}

    def candidates(it):
        # Items of one kind share a block and therefore a footprint; six food
        # trucks are six searches of the same grid unless this is cached.
        k = (round(it.w, 2), round(it.h, 2), tuple(it.rotations),
             round(it.clearance, 2))
        if k not in cache:
            # No early `limit`: `anchors` stops the moment it reaches one, and
            # it walks the grid in x order, so a limit hit confines every large
            # item to the western end of the site. Enumerate, then thin evenly.
            got = anchors(free, it.w, it.h, step=step, rotations=it.rotations,
                          margin=it.clearance / 2, limit=10 ** 6,
                          ground=ground)
            if len(got) > CAND_CAP:
                got = got[::len(got) // CAND_CAP + 1]
            cache[k] = got
        return cache[k]

    ctx = {"region": region, "span": span, "cx": cx, "cy": cy, "rng": rng,
           "anchor": anchor_key, "relaxed": set()}

    state, chosen, weight = [], [], [0.0]
    nodes, spent = [0], [0]
    deadline = [t0 + seconds]
    reasons = {}

    def options(it):
        raw = candidates(it)
        if not raw:
            reasons[it.key] = (f"{it.key}: nothing on this ground takes "
                               f"{it.w:g}x{it.h:g} m with {it.clearance:g} m clear")
            return []
        needs = _needs(it, state)
        rows = []
        for (x, y, rot) in raw:
            w, h = it.size(rot)
            if _blocked(x, y, w, h, needs) is None:
                rows.append((x, y, rot, w, h))
        if not rows:
            why = {}
            for (x, y, rot) in raw:
                w, h = it.size(rot)
                b = _blocked(x, y, w, h, needs)
                why[b] = why.get(b, 0) + 1
            reasons[it.key] = (
                f"{it.key}: {len(raw)} positions fit the ground, all of them "
                f"blocked (" + ", ".join(f"{v} on {k}" for k, v in why.items()) + ")")
            return []
        rows = _globals_filter(it, rows, cons.get(it.key, ()), ctx, reasons)
        if not rows:
            return []
        cs = cons.get(it.key, ())
        look = _lookups(it, cs, state, ctx)
        scored = [(_score(it, r, cs, look, ctx), r) for r in rows]
        # Ties are common — a dozen candidates in one empty corner score
        # identically — and always breaking them the same way stacks every item
        # of a kind onto the same grid line. `seed` therefore does something.
        rng.shuffle(scored)
        scored.sort(key=lambda t: -t[0])
        return scored

    # Two records, both ranked (how many placed, then how well). `best` is only
    # written when a descent actually reached the end of the programme;
    # `deepest` is written at every node, and is what gets returned when the
    # clock beats the search.
    best = {"n": -1, "kept": -1, "w": 0.0, "out": [], "stated": 0, "lost": []}
    deepest = {"n": -1, "w": 0.0, "out": []}

    def keep(store):
        if (len(chosen), weight[0]) > (store["n"], store["w"]):
            store.update(n=len(chosen), w=weight[0], out=list(chosen))

    def keep_complete():
        # `state` at this moment is the finished layout, facings and all.
        k, stated, lost = _kept(state, cons)
        if (len(chosen), k, weight[0]) > (best["n"], best["kept"], best["w"]):
            best.update(n=len(chosen), kept=k, w=weight[0], out=list(chosen),
                        stated=stated, lost=lost)

    def descend(i, allow_skip):
        if nodes[0] > NODES or time.time() > deadline[0]:
            return False
        nodes[0] += 1
        spent[0] += 1
        keep(deepest)
        if i >= len(order):
            return True
        it = order[i]
        for sc, (x, y, rot, w, h) in _apart(options(it), BRANCH,
                                            min(it.w, it.h)):
            face = _facing(it, x, y, rot, cons.get(it.key, ()), state, ctx)
            state.append(_Put(it, x, y, rot, face))
            chosen.append(Placement(it.key, it.block, x, y, rot, label=it.label))
            weight[0] += sc
            if descend(i + 1, allow_skip):
                return True
            weight[0] -= sc
            state.pop()
            chosen.pop()
        if allow_skip:
            # Holodeck's DFS retreats on a dead end and fails the scene if the
            # retreat runs out. An event programme is routinely wider than the
            # ground it is handed, so once backtracking has been tried the
            # engine goes on without the item and says which one, in the notes.
            # Silently dropping it is the failure mode this branch exists to
            # make impossible.
            return descend(i + 1, True)
        return False

    # The anchor's candidate positions, which are also the roots the descents
    # restart from — see the module docstring for why the restart is over this
    # choice and no other.
    #
    # Nothing is standing yet, so this is the emptiest the ground will ever be:
    # an item with no candidate now has none at all. Drop it, say so, and order
    # again without it — because an item that cannot stand is not a legitimate
    # target for anything else's wish either. The "impossible" case is exactly
    # this: a 300 x 200 m block of kind `stage` satisfies three grandstands'
    # wish to be near a stage, orders itself first, and never appears, leaving
    # them arranged around nothing.
    dead = []
    while True:
        state[:] = list(base_state)
        chosen[:] = list(pinned)
        roots = _apart(options(order[0]), ROOTS, max(20.0, span / ROOT_DIV))
        if roots or len(order) == 1:
            break
        dead.append(order[0].key)
        order = _order([it for it in todo if it.key not in dead], cons, index,
                       {p.key for p in base_state}, [])
        anchor_key = order[0].key
        ctx["anchor"] = anchor_key
    first = order[0]
    if not cons:
        # Said here rather than above, because the anchor is only settled once
        # the ground has had its say about whether it can hold the first item.
        notes.append(f"no constraint program: {anchor_key} taken as the anchor, "
                     "placed centrally, and the rest scored by proximity to it")

    # The clock, spent in two halves. The first pass may not drop an item, so
    # on a programme wider than the ground it is guaranteed to fail — and if it
    # is allowed to spend the whole budget failing, the pass that would have
    # returned a plan never runs. Within a pass the remaining time is split
    # evenly over the roots still to try, and no descent may outlive `seconds`:
    # the budget is soft, but a 1 s floor on a 0.2 s budget is not softness.
    tried = 0
    for allow_skip in (False, True):
        stop = t0 + seconds * (0.9 if allow_skip else 0.45)
        pool = roots or [None]
        for n, root in enumerate(pool):
            if time.time() >= stop:
                break
            if root is None and not allow_skip:
                continue        # no anchor and no licence to drop it: hopeless
            nodes[0] = 0
            weight[0] = 0.0
            state[:] = list(base_state)
            chosen[:] = list(pinned)
            deadline[0] = min(stop, time.time()
                              + (stop - time.time()) / (len(pool) - n))
            if root is not None:
                sc, (x, y, rot, w, h) = root
                face = _facing(first, x, y, rot, cons.get(first.key, ()),
                               state, ctx)
                state.append(_Put(first, x, y, rot, face))
                chosen.append(Placement(first.key, first.block, x, y, rot,
                                        label=first.label))
                weight[0] = sc
            tried += 1
            if descend(1, allow_skip):
                keep_complete()
        if best["n"] >= 0:
            break

    out = best["out"] if best["n"] >= 0 else deepest["out"]
    if best["n"] < 0:
        notes.append("no descent reached the end of the programme in the time "
                     "given; returning the deepest layout proved legal")
    else:
        notes.append(f"{tried} anchor position(s) tried; kept the layout that "
                     f"holds {best['kept']} of {best['stated']} stated relations")
        for msg in best["lost"][:8]:
            notes.append(msg)
        if len(best["lost"]) > 8:
            notes.append(f"...and {len(best['lost']) - 8} more relations not kept")
    # Over `todo`, not over `order` — an item dropped for being unplaceable is
    # no longer in the order, and it is the one the caller most needs named.
    have = {p.key for p in out}
    for it in todo:
        if it.key not in have:
            notes.append(reasons.get(
                it.key, f"{it.key}: no position left once the rest of the "
                        "programme was standing"))
    for msg in sorted(ctx["relaxed"]):
        notes.append(msg)
    notes.append(f"{len(out)} of {len(items)} placed, {spent[0]} search nodes, "
                 f"{time.time() - t0:.1f}s")
    return out, notes


def _apart(scored, want, spread):
    """The best `want` candidates that describe DIFFERENT plans.

    The top six candidates by score are six neighbouring cells of the same
    grid: exploring each of them explores the same layout six times. Non-
    maximum suppression instead — the best candidate, then the best that
    stands at least `spread` metres from every one already taken.

    This was written for the anchor's roots and used only there, and the
    branch set at every other level had the fault it was written to cure. The
    three positions `descend` tried for the second grandstand were (382.6,
    -1064.3), (382.6, -1062.5) and (382.6, -1066.1): three cells of one grid
    row, 1.8 m apart, occupying the same slot on the same band of ground and
    ruling out the same alternatives for everything below. The search had a
    nominal branching factor of three and an effective one of one. A root is
    just the branch set at depth zero, so both call sites take the same
    suppression, and below the root `spread` is the object's own short side —
    two placements that overlap are one plan, however the score ranks them."""
    out = []
    for row in scored:
        x, y = row[1][0], row[1][1]
        if all(math.hypot(x - q[1][0], y - q[1][1]) >= spread for q in out):
            out.append(row)
            if len(out) >= want:
                break
    return out


def _reset_state(fixed, items, index, cx, cy):
    """The state as it was before the search: externally fixed placements and
    anything the brief pinned. Every descent is rewound to this."""
    out = []
    for p in fixed:
        it = index.get(p.key)
        if it is not None:
            out.append(_Put(it, p.x, p.y, p.rot,
                            _snap(it, p.rot, math.atan2(cy - p.y, cx - p.x))))
    for it in items:
        if it.fixed:
            fx, fy, rot = (list(it.fixed) + [0.0])[:3]
            fx, fy, rot = float(fx), float(fy), float(rot)
            out.append(_Put(it, fx, fy, rot,
                            _snap(it, rot, math.atan2(cy - fy, cx - fx))))
    return out


def _order(todo, cons, index, already, notes):
    """Largest first, but never before something it refers to.

    Two rules meet here. Holodeck places the biggest object first, because a
    grandstand has few legal positions and a bin has thousands, and the bin
    must not take the grandstand's ground. And a relation is measured in its
    target's frame, so the target has to be standing. Where they disagree the
    reference wins: this is a greedy topological sort keyed on area.

    `Item.near` is a reference too, and treating it as one is what makes the
    fallback layout an event rather than a packing. Three 30 x 10 m grandstands
    each ask to be near the 20 x 12 m stage; on area alone all three go down
    first, they fill the only band of ground deep enough for a 12 m stage, and
    the stage is then placed 126 m from its own audience — legal, zero
    violations, and nonsense. So a wish to be near something orders the wisher
    after it.

    Those wishes are ordered SOFTLY. A DSL relation cannot be read before its
    target is standing, so a cycle there is a real defect worth a note; two
    items merely wishing to be near each other is an ordinary brief, and it
    must not print a warning. Hence two dependency sets, relaxed in turn.

    A dependency on a KIND is satisfied by any one member of it, which is what
    `parse` promises when it lets a line say "near, food" — that some food is
    already standing, not all of it. Reading it as every member turns
    "food-5 | near, food" and "food-6 | near, food" into a cycle between two
    lines that are both perfectly ordered."""
    members = {}
    for k, it in index.items():
        members.setdefault(it.kind, set()).add(k)

    def groups(key, targets):
        out = []
        for t in targets:
            if t is None:
                continue
            if t in index and t != key:
                out.append({t})
            elif g := (members.get(t, set()) - {key}):
                out.append(g)
        return out

    hard, soft = {}, {}
    for it in todo:
        hard[it.key] = groups(it.key, [c["target"] for c in cons.get(it.key, ())])
        soft[it.key] = groups(it.key, list(it.near))

    def ready_under(need, done):
        return all(g & done for g in need)

    order, rest, done, warned = [], list(todo), set(already), False
    while rest:
        ready = [it for it in rest if ready_under(hard[it.key], done)
                 and ready_under(soft[it.key], done)]
        if not ready:
            ready = [it for it in rest if ready_under(hard[it.key], done)]
        if not ready:
            if not warned:
                notes.append("constraints reference each other in a cycle "
                             f"({', '.join(sorted(i.key for i in rest))}); "
                             "ordered by size instead")
                warned = True
            ready = rest
        ready.sort(key=lambda i: (-(i.w * i.h), i.key))
        pick = ready[0]
        order.append(pick)
        rest.remove(pick)
        done.add(pick.key)
    return order


def _targets(name, state):
    return [p for p in state if p.key == name or p.kind == name]


def _frame(put, x, y):
    """(forward, lateral, distance) of a point in a placed object's own frame.

    Left of, right of, in front of and center aligned are all read from the
    target's point of view in Holodeck, and they stay that way here: "left of
    the stage" is the stage's left, which is the only reading that does not
    silently depend on which way the drawing happens to be printed. The
    absolute reading a brief usually wants is served by the compass globals."""
    dx, dy = x - put.x, y - put.y
    c, s = math.cos(put.facing), math.sin(put.facing)
    return dx * c + dy * s, -dx * s + dy * c, math.hypot(dx, dy)


def _globals_filter(it, rows, constraints, ctx, reasons):
    """`edge` and the compass are hard, exactly as Holodeck's `edge` is hard.

    `edge` is measured against the boundary of the REGION, not of the free
    polygon. The free polygon on a real masterplan is dozens of disjoint parts
    riddled with obstacle outlines, so nearly every point in it is within a
    metre or two of some boundary and the filter would pass everything —
    "against the perimeter" has to mean the site's perimeter to mean anything.

    A hard filter that empties the candidate list is relaxed rather than
    obeyed, and the relaxation is reported: a stage in the wrong half of the
    site is a plan to argue with, no stage is not."""
    x0, y0, x1, y1 = ctx["region"]
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    for c in constraints:
        op = c["op"]
        if op == "edge":
            lim = EDGE_BAND * min(it.w, it.h)
            keep = [r for r in rows
                    if min(r[0] - x0, x1 - r[0], r[1] - y0, y1 - r[1]) <= lim]
        elif op == "north":
            keep = [r for r in rows if r[1] >= my]
        elif op == "south":
            keep = [r for r in rows if r[1] <= my]
        elif op == "east":
            keep = [r for r in rows if r[0] >= mx]
        elif op == "west":
            keep = [r for r in rows if r[0] <= mx]
        else:
            continue
        if keep:
            rows = keep
        else:
            ctx["relaxed"].add(
                f"{it.key}: no free ground in the '{op}' of the site once the "
                "rest was standing; the constraint was relaxed to a preference")
    return rows


def _snap(it, rot, bearing):
    """A bearing, turned onto the axis the object is actually drawn on.

    An object presents its broad side to whatever it faces — a stage turns its
    width to the crowd — so the direction it looks in is perpendicular to its
    long axis, and its long axis is decided by `rot`, which the packing has
    already chosen. `_keeps` and `_score` read `face to` exactly that way.
    `_facing` did not: it returned the raw bearing to the target or to the
    middle of the ground, so a 30 x 10 m grandstand lying east-west was
    recorded as looking north-east if that was where the site centroid lay.
    Half the vocabulary was then being measured in a frame the drawing does
    not have — `face to` off the rotation, `in front of` off a free bearing —
    and the two halves disagreed about which way one object was turned.

    Keeping the SENSE of the bearing and taking the AXIS from the rotation
    settles it: of the two directions the drawn object could look along, the
    one the bearing points to within a right angle."""
    front = math.radians(rot) + (0.0 if it.w >= it.h else math.pi / 2) \
        + math.pi / 2
    return front if math.cos(bearing - front) >= 0 else front + math.pi


def _facing(it, x, y, rot, constraints, state, ctx):
    """Which way the object looks — the frame every later relation is read in.

    `face to, X` sets it. Otherwise an object looks inward, at the middle of
    the ground: a stage on the northern edge faces its audience to the south,
    which is what a reader expects from "in front of the stage" without having
    been told anything about the stage's rotation. Either way the answer is
    snapped to the object's own drawn axis — see `_snap`."""
    for c in constraints:
        if c["op"] == "face to":
            tg = _targets(c["target"], state)
            if tg:
                t = min(tg, key=lambda p: (p.x - x) ** 2 + (p.y - y) ** 2)
                return _snap(it, rot, math.atan2(t.y - y, t.x - x))
    return _snap(it, rot, math.atan2(ctx["cy"] - y, ctx["cx"] - x))


def _lookups(it, constraints, state, ctx):
    """Everything in `_score` that depends on the standing layout but not on
    the candidate position, resolved once per item instead of once per
    candidate. `options` scores up to seven hundred candidates against one
    unchanged `state`, and re-deriving the same relation targets, the same
    same-kind list and the same packing centroid seven hundred times over was
    a fifth of the search budget on the heavy programme."""
    look = {
        "targets": [None if c["target"] is None else _targets(c["target"], state)
                    for c in constraints],
        "wish": [tg for tg in (_targets(want, state) for want in it.near) if tg],
        "admirers": [p for p in state
                     if it.key in p.item.near or it.kind in p.item.near],
        "same": [p for p in state if p.kind == it.kind],
        "anchor": None,
        "pack": None,
    }
    if not look["same"] and ctx["anchor"] and not constraints:
        a = [p for p in state if p.key == ctx["anchor"]]
        look["anchor"] = a[0] if a else None
    if state:
        look["pack"] = (sum(p.x for p in state) / len(state),
                        sum(p.y for p in state) / len(state))
    return look


def _score(it, row, constraints, look, ctx):
    """A candidate's weight. Holodeck's rule: relations never remove a
    candidate, they only make it more attractive than its neighbours, so a
    programme whose relations cannot all be met still produces a layout that
    meets as many as the ground allows."""
    x, y, rot, w, h = row
    x0, y0, x1, y1 = ctx["region"]
    span = ctx["span"]
    s = 0.0

    perim = min(x - x0, x1 - x, y - y0, y1 - y)
    half = min(x1 - x0, y1 - y0) / 2

    for c, tg in zip(constraints, look["targets"]):
        op, name = c["op"], c["target"]
        if op == "edge":
            # Scored as well as filtered, because `_globals_filter` relaxes a
            # hard global that empties the candidate list and tells the reader
            # it has been "relaxed to a preference". For the compass that was
            # true — they are scored below. For `edge` it was not: nothing here
            # read it, so a relaxed `edge` silently became no constraint at all
            # and the note misdescribed what the engine had done.
            s += W["edge"] * _band(perim, EDGE_BAND * min(it.w, it.h))
            continue
        if op == "middle":
            s += W["middle"] * min(1.0, perim / max(half, 1e-6))
            continue
        if op == "central":
            s += W["central"] * _band(math.hypot(x - ctx["cx"], y - ctx["cy"]),
                                      span / 2)
            continue
        if op in COMPASS:
            reach = {"north": y - y0, "south": y1 - y,
                     "east": x - x0, "west": x1 - x}[op]
            s += W["compass"] * min(1.0, reach / max(y1 - y0 if op in ("north", "south")
                                                     else x1 - x0, 1e-6))
            continue

        if not tg:
            continue
        near = min(tg, key=lambda p: _gap(it, x, y, w, h, p))
        d = _gap(it, x, y, w, h, near)
        f, lat, ctr = _frame(near, x, y)
        across = max(near.w, near.h) / 2 + LATERAL_M

        if op == "near":
            s += W["near"] * _band(d, NEAR_M)
        elif op == "far":
            s += W["far"] * min(1.0, d / FAR_M)
        elif op == "in front of":
            if f > 0:
                s += W["in front of"] * _band(abs(lat), across) * _band(d, NEAR_M)
        elif op == "side of":
            if abs(lat) > abs(f):
                s += W["side of"] * _band(d, NEAR_M)
        elif op == "left of":
            if lat > 0:
                s += W["left of"] * _band(d, NEAR_M)
        elif op == "right of":
            if lat < 0:
                s += W["right of"] * _band(d, NEAR_M)
        elif op == "center aligned":
            s += W["center aligned"] * _band(abs(lat), ALIGN_M)
        elif op == "face to":
            # The rotation, not the position: an object faces something with
            # its broad side — a stage turns its width to the crowd — so score
            # the rotation that puts the short axis along the bearing.
            bearing = math.atan2(near.y - y, near.x - x)
            front = math.radians(rot) + (0.0 if it.w >= it.h else math.pi / 2)
            s += W["face to"] * abs(math.cos(bearing - front - math.pi / 2))

    # The programme's own adjacency wish, which `score` measures at 60 m of
    # centre distance. It is not in the DSL and it still has to be honoured.
    for tg in look["wish"]:
        d = min(math.hypot(x - p.x, y - p.y) for p in tg)
        s += W_ITEM_NEAR * _band(d, 60.0)

    # And read in the other direction, exactly as `_separation` reads `far`.
    # A wish is a fact about the pair, and largest-first routinely places the
    # thing that was wished for AFTER the things that wished for it: three
    # grandstands each 300 m² ask to be near a 240 m² stage, so all three are
    # standing before the stage chooses. Scoring only the forward direction
    # left the stage free to land 200 m from its own audience. Summed rather
    # than taken at the nearest, because a stage that satisfies three
    # grandstands should beat one that satisfies the closest and abandons two.
    for p in look["admirers"]:
        s += W_ITEM_NEAR * _band(math.hypot(x - p.x, y - p.y), 60.0)

    # Cohesion by kind. Not Holodeck — an addition, and the one that decides
    # whether this engine passes: six food trucks scattered over 360 m leave
    # nowhere at all for a toilet that has to stand 20 m clear of every one of
    # them, whereas a food village leaves the other 340 m.
    if look["same"]:
        s += W_KIND * _band(min(_gap(it, x, y, w, h, p) for p in look["same"]),
                            NEAR_M)
    elif look["anchor"] is not None:
        s += W_ANCHOR * _band(math.hypot(x - look["anchor"].x,
                                         y - look["anchor"].y), span)

    if look["pack"] is not None:
        gx, gy = look["pack"]
        s += W_PACK * _band(math.hypot(x - gx, y - gy), span)
    else:
        # Nothing standing yet: this is the anchor, and an anchor with no
        # instruction belongs in the middle of the ground it anchors.
        s += W["central"] * _band(math.hypot(x - ctx["cx"], y - ctx["cy"]),
                                  span / 2)
    return s
