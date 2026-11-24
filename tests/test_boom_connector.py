"""Tests for the BOOM alert broker connector module."""
from pathlib import Path
from applecider_agents.boom_connector import (
    BoomConnectorClient,
    adapt_boom_object,
    BoomScanner,
    BoomPublisher,
    DiscoveryLeaderboard
)
from applecider_agents.collaboration import collaborate

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "boom_demo.json"


def test_boom_connector_client_with_fixture():
    client = BoomConnectorClient(fixture_file=FIXTURE_PATH)
    assert client.is_configured

    # Test candidate query
    candidates = client.query_recent_candidates(min_detections=1, limit=5)
    assert len(candidates) >= 1
    assert "ZTF25abhoijc" in candidates

    # Test photometry rows extraction
    rows = client.get_photometry_rows("ZTF25abhoijc")
    assert len(rows) > 0
    # verify format (mjd, fid, mag, magerr)
    assert len(rows[0]) == 4
    assert rows[0][1] in (1, 2, 3)

    # Test pre-staged cross-matches extraction
    ctx = client.get_crossmatches("ZTF25abhoijc")
    assert "host_sep" in ctx
    assert "z_phot" in ctx


def test_adapt_boom_object_and_seed():
    client = BoomConnectorClient(fixture_file=FIXTURE_PATH)
    ev, ctx = adapt_boom_object("ZTF25abhoijc", client=client, seed_core=True)

    assert ev is not None
    assert len(ev) > 0
    assert ctx["_source"] == "boom_connector"
    assert "z_phot" in ctx


def test_boom_scanner_leaderboard():
    client = BoomConnectorClient(fixture_file=FIXTURE_PATH)
    scanner = BoomScanner(client=client, model="flare")

    leaderboard = scanner.scan(object_ids=["ZTF25abhoijc", "ZTF19acbzgog"], horizon_days=100.0)
    assert isinstance(leaderboard, DiscoveryLeaderboard)
    assert leaderboard.total_scanned == 2
    assert len(leaderboard.candidates) == 2

    # Check ranking order (highest discovery potential first)
    assert leaderboard.candidates[0].discovery_potential >= leaderboard.candidates[1].discovery_potential

    # Check summary table rendering
    table = leaderboard.summary_table()
    assert "BOOM CONNECTOR: DISCOVERY LEADERBOARD" in table
    assert "ZTF25abhoijc" in table


def test_boom_publisher_annotation_and_ui_card():
    client = BoomConnectorClient(fixture_file=FIXTURE_PATH)
    adapt_boom_object("ZTF25abhoijc", client=client, seed_core=True)
    report = collaborate("ZTF25abhoijc", model="flare", horizon_days=100.0)

    publisher = BoomPublisher(client=client)
    annot = publisher.format_annotation(report)
    assert annot["origin"] == "applecider-agents-connector"
    assert annot["obj_id"] == "ZTF25abhoijc"
    assert "verdict" in annot
    assert len(annot["tags"]) > 0

    card_html = publisher.render_boom_tile_card(report)
    assert "[BOOM CONNECTOR]" in card_html
    assert "DISCOVERY" in card_html
    assert "oklch" in card_html


if __name__ == "__main__":
    test_boom_connector_client_with_fixture()
    test_adapt_boom_object_and_seed()
    test_boom_scanner_leaderboard()
    test_boom_publisher_annotation_and_ui_card()
    print("All boom_connector tests passed successfully!")
