"""applecider-agents triage --model flare IDS... [--llm] [--post] [--horizon 100]
   applecider-agents eval --model flare --list ood|test|my_list.csv [--n 20] [--llm]
   applecider-agents mcp [--transport stdio|streamable-http] [--port 8765] [--models flare]
   applecider-agents nightly --model flare [--since-days 1] [--cap 200]"""
import argparse, json, datetime
from .core import config as C


def main(argv=None):
    ap = argparse.ArgumentParser(prog="applecider-agents", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("triage"); p.add_argument("--model", default="flare"); p.add_argument("ids", nargs="+"); p.add_argument("--llm", action="store_true"); p.add_argument("--post", action="store_true"); p.add_argument("--horizon", type=float, default=100.0)
    e = sub.add_parser("eval"); e.add_argument("--model", default="flare"); e.add_argument("--list", required=True, help="ood | test | path to a CSV with obj_id[,truth]"); e.add_argument("--n", type=int); e.add_argument("--llm", action="store_true")
    m = sub.add_parser("mcp"); m.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http", "sse"]); m.add_argument("--port", type=int, default=8765); m.add_argument("--host", default="127.0.0.1"); m.add_argument("--models", default="flare")
    n = sub.add_parser("nightly"); n.add_argument("--model", default="flare"); n.add_argument("--since-days", type=float, default=1.0); n.add_argument("--cap", type=int, default=C.DAILY_CAP); n.add_argument("--llm", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "triage":
        from .core.loop import triage
        from .core.skyportal_client import publish
        for oid in a.ids:
            rep = triage(a.model, oid, horizon_days=a.horizon, mock=not a.llm)
            print(json.dumps(rep.model_dump(), indent=1)); print("\nCOMMENT:", rep.comment())
            if a.post:
                print("PUBLISH:", publish(rep))
    elif a.cmd == "eval":
        from .core.eval import run
        run(a.model, a.list, a.n, mock=not a.llm)
    elif a.cmd == "mcp":
        from .core.mcp import main as serve
        serve(a.transport, a.port, a.host, a.models.split(","))
    elif a.cmd == "nightly":
        from .core.loop import triage
        from .core.skyportal_client import SkyPortal, publish
        sp = SkyPortal(); since = (datetime.datetime.utcnow() - datetime.timedelta(days=a.since_days)).strftime("%Y-%m-%dT%H:%M:%S")
        srcs = sp.recent_sources(since, C.SKYPORTAL_GROUP_IDS)[: a.cap]
        print(f"{len(srcs)} sources saved since {since} (cap {a.cap}), mode {C.MODE}")
        for s in srcs:
            rep = triage(a.model, s["id"], mock=not a.llm); print(rep.comment()); print(" ->", publish(rep))


if __name__ == "__main__":
    main()
