"""applecider-agents triage --model flare IDS... [--llm] [--post] [--horizon 100]
   applecider-agents collaborate --model flare IDS... [--llm] [--detailed] [--post]
   applecider-agents eval --model flare --list ood|test|my_list.csv [--n 20] [--llm]
   applecider-agents mcp [--transport stdio|streamable-http] [--port 8765] [--models flare]
   applecider-agents nightly --model flare [--since-days 1] [--cap 200]"""
import argparse, json, datetime
from .core import config as C


def main(argv=None):
    ap = argparse.ArgumentParser(prog="applecider-agents", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("triage"); p.add_argument("--model", default="flare"); p.add_argument("ids", nargs="+"); p.add_argument("--llm", action="store_true"); p.add_argument("--post", action="store_true"); p.add_argument("--horizon", type=float, default=100.0)
    c = sub.add_parser("collaborate", help="Multi-agent collaborative investigation & debate"); c.add_argument("--model", default="flare"); c.add_argument("ids", nargs="+"); c.add_argument("--llm", action="store_true"); c.add_argument("--detailed", action="store_true"); c.add_argument("--post", action="store_true"); c.add_argument("--horizon", type=float, default=100.0)
    e = sub.add_parser("eval"); e.add_argument("--model", default="flare"); e.add_argument("--list", required=True, help="ood | test | path to a CSV with obj_id[,truth]"); e.add_argument("--n", type=int); e.add_argument("--llm", action="store_true")
    m = sub.add_parser("mcp"); m.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http", "sse"]); m.add_argument("--port", type=int, default=8765); m.add_argument("--host", default="127.0.0.1"); m.add_argument("--models", default="flare")
    n = sub.add_parser("nightly"); n.add_argument("--model", default="flare"); n.add_argument("--since-days", type=float, default=1.0); n.add_argument("--cap", type=int, default=C.DAILY_CAP); n.add_argument("--llm", action="store_true")
    
    # boom-connector subcommands
    b = sub.add_parser("boom-connector", help="BOOM broker connector: scan, triage, and rank discoveries")
    bsub = b.add_subparsers(dest="boom_cmd", required=True)
    bt = bsub.add_parser("triage", help="Triage broker alerts with multi-agent debate")
    bt.add_argument("ids", nargs="+")
    bt.add_argument("--model", default="flare")
    bt.add_argument("--llm", action="store_true")
    bt.add_argument("--detailed", action="store_true")
    bt.add_argument("--render-card", action="store_true")
    bt.add_argument("--fixture", help="Path to local broker fixture JSON for offline testing")
    
    bsc = bsub.add_parser("scan", help="Scan and rank broker candidate stream")
    bsc.add_argument("--limit", type=int, default=20)
    bsc.add_argument("--min-det", type=int, default=2)
    bsc.add_argument("--max-mag", type=float, default=20.0)
    bsc.add_argument("--model", default="flare")
    bsc.add_argument("--llm", action="store_true")
    bsc.add_argument("--fixture", help="Path to local broker fixture JSON for offline testing")
    sky = sub.add_parser("skyportal", help="Read SkyPortal data and jointly analyze photometry and spectra")
    sky_sub = sky.add_subparsers(dest="sky_cmd", required=True)
    sky_triage = sky_sub.add_parser("triage")
    sky_triage.add_argument("ids", nargs="+")
    sky_triage.add_argument("--detailed", action="store_true")
    sky_triage.add_argument("--post", action="store_true", help="Publish via FLARE_MODE gate (default dry-run)")
    sky_scan = sky_sub.add_parser("scan")
    sky_scan.add_argument("--since-days", type=float, default=1.0)
    sky_scan.add_argument("--limit", type=int, default=20)
    for parser in (sky_triage, sky_scan):
        parser.add_argument("--model", choices=["flare", "astranet", "joint"], default="joint")
        parser.add_argument("--horizon", type=float, default=100.0)
        parser.add_argument("--llm", action="store_true", help="Add an advisory LLM review to measured evidence")
    a = ap.parse_args(argv)
    if a.cmd == "triage":
        from .core.loop import triage
        from .core.skyportal_client import publish
        for oid in a.ids:
            rep = triage(a.model, oid, horizon_days=a.horizon, mock=not a.llm)
            print(json.dumps(rep.model_dump(), indent=1)); print("\nCOMMENT:", rep.comment())
            if a.post:
                print("PUBLISH:", publish(rep))
    elif a.cmd == "collaborate":
        from .collaboration import collaborate
        from .core.skyportal_client import publish
        for oid in a.ids:
            rep = collaborate(oid, model=a.model, horizon_days=a.horizon, use_llm=a.llm)
            if a.detailed:
                print(rep.detailed_summary())
            else:
                print(json.dumps(rep.model_dump(), indent=1))
                print("\nCOMMENT:", rep.comment())
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
    elif a.cmd == "skyportal":
        from .core.skyportal_client import SkyPortal, publish
        from .skyportal_connector import investigate, scan
        if a.horizon <= 0:
            ap.error("--horizon must be positive")
        client = SkyPortal()
        if a.sky_cmd == "triage":
            for oid in a.ids:
                report = investigate(oid, client=client, model=a.model, horizon_days=a.horizon, use_llm=a.llm)
                print(report.detailed_summary() if a.detailed else report.model_dump_json(indent=2))
                if a.post:
                    print("PUBLISH:", publish(report, client=client))
        else:
            if a.since_days <= 0 or a.limit <= 0:
                ap.error("--since-days and --limit must be positive")
            since = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=a.since_days)).isoformat()
            result = scan(since, client=client, group_ids=C.SKYPORTAL_GROUP_IDS, limit=a.limit,
                          model=a.model, horizon_days=a.horizon, use_llm=a.llm)
            print(json.dumps(result, indent=2))
    elif a.cmd == "boom-connector":
        from .boom_connector import BoomConnectorClient, adapt_boom_object, BoomScanner, BoomPublisher
        from .collaboration import collaborate
        client = BoomConnectorClient(fixture_file=a.fixture)
        if a.boom_cmd == "triage":
            publisher = BoomPublisher(client=client)
            for oid in a.ids:
                adapt_boom_object(oid, client=client, seed_core=True)
                rep = collaborate(oid, model=a.model, use_llm=a.llm)
                if a.detailed:
                    print(rep.detailed_summary())
                else:
                    print(json.dumps(rep.model_dump(), indent=1))
                    print("\nCOMMENT:", rep.comment())
                if a.render_card:
                    print("\n--- BOOM UI CARD ---\n", publisher.render_boom_tile_card(rep))
        elif a.boom_cmd == "scan":
            scanner = BoomScanner(client=client, model=a.model)
            board = scanner.scan(limit=a.limit, min_detections=a.min_det, max_mag=a.max_mag, use_llm=a.llm)
            print(board.summary_table())


if __name__ == "__main__":
    main()
