"""plan2event-holodeck — a constraint program, solved.

The engine is engines/holodeck.py, after allenai/Holodeck (CVPR 2024). Its
finding is the one this whole rebuild rests on: a language model should not
emit coordinates. It emits relations — this near that, this at the edge, this
facing that — and a DFS solver scores every proved-legal candidate against
them and backtracks when a branch dies.

Run:    python agent.py
Deploy: python agent.py deploy
"""

import asyncio
import pathlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cycls

from core import wiring

# A Windows host pickles WindowsPath, which a Linux container cannot rebuild.
pathlib.WindowsPath.__reduce__ = lambda self: (pathlib.PurePosixPath, (self.as_posix(),))

IMAGE = wiring.image()
# This repository is the v1 variant. v1 and v2 of an engine are the same code;
# v2 additionally reads reference/, carries the composition rules, and is
# reviewed and revised twice after it stops.
V2 = False
NAME = "plan2event-holodeck" + ("-v2" if V2 else "")


@cycls.agent(name=NAME, image=IMAGE, web=wiring.web("Event plan — constraint program"),
             memory="4Gi", volumes=wiring.volumes(NAME))
async def plan2event_holodeck(context):
    from core import prompt, sdk, tools
    from engines import holodeck

    ws = Path(context.workspace.root)
    ws.mkdir(parents=True, exist_ok=True)
    tools.intake(context, ws)

    if not (ws / tools.PLAN).exists():
        yield ("Attach the venue plan as a **DXF** and describe the event — "
               "what it is, who comes, how many, and anything the site or the "
               "country requires.")
        return

    session = tools.Session(ws, holodeck.solve)
    specs = tools.schemas(holodeck.NAME, holodeck.DOC,
                          getattr(holodeck, "CONSTRAINT_HELP", ""))

    async for event in sdk.drive(
            context, ws, tools.make_handlers(session), specs,
            prompt.build(holodeck.NAME, holodeck.DOC,
                         takes_constraints=hasattr(holodeck, "parse"),
                         rounds=2 if V2 else 0, knowledge=V2),
            session=session, rounds=2 if V2 else 0):
        yield event

    if (ws / tools.PLAN).exists():
        await asyncio.to_thread(shutil.copy, ws / tools.PLAN, ws / tools.OUT)
        yield (f"\n\nSaved as **{tools.OUT}** — download it from the Files "
               f"panel. Everything added is on `EVENT-*` layers; deleting them "
               f"returns the drawing you sent.")


wiring.finalise(plan2event_holodeck)

if __name__ == "__main__":
    plan2event_holodeck.deploy() if "deploy" in sys.argv else plan2event_holodeck.local()
