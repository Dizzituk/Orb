# FILE: tests/test_sentinel_rules.py
# Purpose: Units for app/sentinel/rules.py (pure rule logic) + collector snapshot
#          diffing — fixture connection dicts, no live agent, no DB.
# Called-by: pytest
# Depends-on: app.sentinel.rules, app.sentinel.collector
# Last-renovated: 2026-06-12
from __future__ import annotations

from app.sentinel import collector, rules


def _pair_ctx(**overrides):
    ctx = {
        "process_name": "newtool.exe",
        "exe_path": r"C:\Program Files\NewTool\newtool.exe",
        "raddr_ip": "93.184.216.34",
        "raddr_port": 443,
        "rdns": "example.com",
        "country": "US",
        "is_public": True,
        "process_known": False,
        "known_ports": set(),
        "known_countries": set(),
        "trusted": False,
    }
    ctx.update(overrides)
    return ctx


def _keys(findings):
    return {f["rule_key"] for f in findings}


# ── R1: new process making external connections ─────────────────────────────

def test_new_process_external_is_medium():
    findings = rules.evaluate_new_pair(_pair_ctx())
    new_proc = [f for f in findings if f["rule_key"] == "new_process"]
    assert len(new_proc) == 1
    assert new_proc[0]["severity"] == rules.MEDIUM


def test_new_process_from_temp_is_severe():
    ctx = _pair_ctx(exe_path=r"C:\Users\dizzi\AppData\Local\Temp\evil.exe")
    new_proc = [f for f in rules.evaluate_new_pair(ctx) if f["rule_key"] == "new_process"]
    assert new_proc[0]["severity"] == rules.SEVERE


def test_known_process_does_not_fire_new_process():
    ctx = _pair_ctx(process_known=True, known_ports={443}, known_countries={"US"})
    assert "new_process" not in _keys(rules.evaluate_new_pair(ctx))


def test_trusted_pair_fires_nothing():
    assert rules.evaluate_new_pair(_pair_ctx(trusted=True)) == []


def test_private_remote_fires_nothing_public():
    ctx = _pair_ctx(is_public=False, rdns="", country="")
    assert _keys(rules.evaluate_new_pair(ctx)) == set()


# ── R2: raw IP with no reverse DNS ───────────────────────────────────────────

def test_raw_ip_common_port_is_low():
    ctx = _pair_ctx(rdns="", raddr_port=443, process_known=True,
                    known_ports={443}, known_countries={"US"})
    raw = [f for f in rules.evaluate_new_pair(ctx) if f["rule_key"] == "raw_ip"]
    assert len(raw) == 1
    assert raw[0]["severity"] == rules.LOW


def test_raw_ip_odd_port_is_medium():
    ctx = _pair_ctx(rdns="", raddr_port=4444, process_known=True,
                    known_ports={4444}, known_countries={"US"})
    raw = [f for f in rules.evaluate_new_pair(ctx) if f["rule_key"] == "raw_ip"]
    assert raw[0]["severity"] == rules.MEDIUM


# ── R3: unusual remote port for a known process ──────────────────────────────

def test_unusual_port_low():
    ctx = _pair_ctx(process_known=True, known_ports={443, 80},
                    known_countries={"US"}, raddr_port=9999)
    unusual = [f for f in rules.evaluate_new_pair(ctx) if f["rule_key"] == "unusual_port"]
    assert len(unusual) == 1
    assert unusual[0]["severity"] == rules.LOW


def test_common_port_never_unusual():
    ctx = _pair_ctx(process_known=True, known_ports={8443}, known_countries={"US"},
                    raddr_port=443)
    assert "unusual_port" not in _keys(rules.evaluate_new_pair(ctx))


# ── R6: known process, new country ───────────────────────────────────────────

def test_new_country_medium():
    ctx = _pair_ctx(process_known=True, known_ports={443},
                    known_countries={"US", "GB"}, country="RU")
    new_country = [f for f in rules.evaluate_new_pair(ctx) if f["rule_key"] == "new_country"]
    assert len(new_country) == 1
    assert new_country[0]["severity"] == rules.MEDIUM


def test_no_country_history_no_finding():
    ctx = _pair_ctx(process_known=True, known_ports={443},
                    known_countries=set(), country="RU")
    assert "new_country" not in _keys(rules.evaluate_new_pair(ctx))


# ── R5: new listening port ───────────────────────────────────────────────────

def test_new_listen_medium():
    findings = rules.evaluate_listen({
        "process_name": "srv.exe", "exe_path": r"C:\Program Files\Srv\srv.exe",
        "laddr_port": 9000, "listen_known": False, "trusted": False,
    })
    assert findings[0]["rule_key"] == "new_listen"
    assert findings[0]["severity"] == rules.MEDIUM


def test_new_listen_from_temp_is_severe():
    findings = rules.evaluate_listen({
        "process_name": "x.exe", "exe_path": r"C:\Users\dizzi\Downloads\x.exe",
        "laddr_port": 9000, "listen_known": False, "trusted": False,
    })
    assert findings[0]["severity"] == rules.SEVERE


def test_low_port_listen_is_severe():
    findings = rules.evaluate_listen({
        "process_name": "srv.exe", "exe_path": r"C:\Program Files\Srv\srv.exe",
        "laddr_port": 81, "listen_known": False, "trusted": False,
    })
    assert findings[0]["severity"] == rules.SEVERE


def test_known_listen_silent():
    assert rules.evaluate_listen({
        "process_name": "srv.exe", "exe_path": "", "laddr_port": 9000,
        "listen_known": True, "trusted": False,
    }) == []


# ── R4: connection-count spike ───────────────────────────────────────────────

def test_spike_fires_above_threshold():
    finding = rules.evaluate_count_spike(200, history_avg=40.0, history_days=5)
    assert finding is not None
    assert finding["severity"] == rules.MEDIUM
    assert finding["remote"] == "machine-wide"


def test_spike_quiet_below_threshold():
    assert rules.evaluate_count_spike(50, history_avg=40.0, history_days=5) is None


def test_spike_needs_history():
    assert rules.evaluate_count_spike(500, history_avg=40.0, history_days=1) is None
    assert rules.evaluate_count_spike(500, history_avg=None, history_days=0) is None


# ── Collector snapshot diffing (fixture snapshots, in-memory only) ───────────

def _conn(pname="app.exe", raddr="93.184.216.34", rport=443, status="ESTABLISHED",
          lport=51000, **extra):
    base = {
        "pid": 1234, "process_name": pname, "exe_path": rf"C:\Apps\{pname}",
        "laddr_ip": "192.168.1.10", "laddr_port": lport,
        "raddr_ip": raddr, "raddr_port": rport, "status": status, "proto": "tcp",
    }
    base.update(extra)
    return base


def _reset_collector_state():
    collector._active_pairs.clear()
    collector._active_listens.clear()


def test_diff_detects_new_pair_once():
    _reset_collector_state()
    snap = [_conn()]
    new_pairs, new_listens, ext = collector._diff_snapshot(snap)
    assert len(new_pairs) == 1 and not new_listens and ext == 1
    # same snapshot again — nothing new
    new_pairs, new_listens, ext = collector._diff_snapshot(snap)
    assert new_pairs == [] and ext == 1


def test_diff_detects_new_listener():
    _reset_collector_state()
    snap = [_conn(status="LISTEN", raddr="", rport=0, lport=8088)]
    new_pairs, new_listens, ext = collector._diff_snapshot(snap)
    assert new_listens and new_listens[0]["laddr_port"] == 8088
    assert new_pairs == [] and ext == 0


def test_diff_skips_loopback():
    _reset_collector_state()
    snap = [_conn(raddr="127.0.0.1")]
    new_pairs, _, ext = collector._diff_snapshot(snap)
    assert new_pairs == [] and ext == 0


def test_diff_private_remote_tracked_not_counted_external():
    _reset_collector_state()
    snap = [_conn(raddr="192.168.1.50", rport=445)]
    new_pairs, _, ext = collector._diff_snapshot(snap)
    assert len(new_pairs) == 1 and ext == 0


# ── First-seed cycle: boot #1 defines normal, fires nothing ──────────────────

def test_first_seed_cycle_fires_no_rules(monkeypatch, tmp_path):
    import app.db as app_db
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.db import Base
    from app.sentinel import alerts as alerts_mod
    from app.sentinel.models import (
        SentinelAlert, SentinelBaseline, SentinelConnection, SentinelState,
    )

    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine, tables=[
        SentinelBaseline.__table__, SentinelConnection.__table__,
        SentinelAlert.__table__, SentinelState.__table__,
    ])
    test_session = sessionmaker(bind=engine)
    monkeypatch.setattr(app_db, "SessionLocal", test_session)
    monkeypatch.setattr(alerts_mod, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(alerts_mod, "_ACTIVE_FILE", tmp_path / "alerts_active.json")

    # NB: must be genuinely public IPs — Python 3.13 ipaddress marks the RFC
    # documentation ranges (203.0.113.x etc.) as private, which rules ignore.
    nasty_pair = _conn(pname="evil.exe", raddr="45.83.220.7", rport=4444,
                       exe_path=r"C:\Users\dizzi\AppData\Local\Temp\evil.exe")
    nasty_listen = _conn(pname="evil.exe", status="LISTEN", raddr="", rport=0, lport=81,
                         exe_path=r"C:\Users\dizzi\AppData\Local\Temp\evil.exe")
    enrich_map = {"45.83.220.7": {"rdns": "", "country_code": "RU", "country": "Russia"}}

    # Cycle 1 — first ever: would be severe twice over, but seeding stays silent.
    out = collector._process_db([nasty_pair], [nasty_listen], 5, enrich_map)
    assert out == []
    db = test_session()
    try:
        assert db.query(SentinelAlert).count() == 0
        assert db.query(SentinelBaseline).count() >= 2  # pair + listener seeded
        assert db.query(SentinelConnection).count() == 2  # events still recorded
    finally:
        db.close()

    # Cycle 2 — a different new temp-exe process: severe rules fire now (learn
    # mode passes severe through; medium/low land suppressed).
    nasty2 = _conn(pname="evil2.exe", raddr="45.83.220.99", rport=4445,
                   exe_path=r"C:\Users\dizzi\AppData\Local\Temp\evil2.exe")
    enrich_map2 = {"45.83.220.99": {"rdns": "", "country_code": "RU", "country": "Russia"}}
    out2 = collector._process_db([nasty2], [], 5, enrich_map2)
    severities = {f["severity"] for f, _ in out2}
    assert rules.SEVERE in severities  # new_process from temp → severe, to triage
    db = test_session()
    try:
        suppressed = db.query(SentinelAlert).filter(SentinelAlert.suppressed.is_(True)).count()
        assert suppressed >= 1  # raw_ip medium recorded quietly during learn mode
    finally:
        db.close()
