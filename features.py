"""
Extract per-kill aim features from CS2 demos for all players.

Usage (run from the folder containing your .dem files and a prodemos/ folder):
    python features.py              # own = *.dem here, pro = prodemos/ (recursive)
    python features.py --windows    # same, and also save the raw per-kill traces to
                                    # windows.pkl (needed by synthetic_eval.py)
    python features.py 128          # legacy CS:GO option: treat pro demos as 128 tick.
                                    # CS2 servers are all 64 tick, so normally leave it out.

Output: kill_features.csv, one row per kill, with features describing the
attacker's aim in the 1 second before the FIRST shot of the burst that killed
the victim. Steam IDs are hashed with a private salt (see anon()).

PRIVACY: hashing hides Steam IDs, but demo name + tick still identifies who got a
kill to anyone who has the same demo (all HLTV demos are public). Do not publish
per-kill rows from kill_features.csv, anomalies.csv or windows.pkl for pro demos;
publish aggregates.

Angle conventions (Source / CS2, checked against real demo data): yaw is in
[-180, 180] with yaw 0 facing +X and yaw 90 facing +Y; pitch is in [-89, 89] and is
POSITIVE when looking DOWN.
"""
import os
import sys
import pickle
import hashlib
import secrets
from pathlib import Path

import numpy as np
import pandas as pd
from demoparser2 import DemoParser

TICKRATE = 64
WINDOW = 64          # ticks before the kill to analyse (1 second)
EYE_HEIGHT = 64.0    # approx eye height above origin when standing (crouching is ~46, not modelled)
BURST_GAP = 24       # max ticks between shots to count as one burst / spray
SETTLE_UNITS = 20.0  # on-target threshold in game units at the victim (about a torso half-width)
SETTLE_DEG = 3.0     # "on target" threshold in degrees

# When this is a list, process_demo() appends one raw window per kept kill to it.
# main() turns it on with --windows; synthetic_eval.py reads the result from windows.pkl.
TRACE_SINK = None

# Weapon names in weapon_fire that are not gunshots. They must not start or extend a burst.
NON_GUN_FIRE = ("knife", "bayonet", "grenade", "flashbang", "molotov", "decoy",
                "c4", "taser", "healthshot")

_SALT = None


def _salt() -> str:
    """Private salt for hashing Steam IDs. With a public salt anyone can hash a known
    player's Steam ID and find them in the CSV, so the salt is kept in a local file
    (.hash_salt, created on first use) or the CS_HASH_SALT environment variable.
    Never publish it."""
    global _SALT
    if _SALT is None:
        _SALT = os.environ.get("CS_HASH_SALT")
        if not _SALT:
            p = Path(__file__).with_name(".hash_salt")
            if p.exists():
                _SALT = p.read_text().strip()
            else:
                _SALT = secrets.token_hex(16)
                p.write_text(_SALT)
                print(f"Created a private hashing salt in {p.name}. Keep it out of anything you share.")
    return _SALT


def anon(steamid) -> str:
    return hashlib.sha256((_salt() + str(steamid)).encode()).hexdigest()[:10]


def wrap(angle):
    """Wrap angle difference to [-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


def build_player_tracks(ticks: pd.DataFrame) -> dict:
    tracks = {}
    for sid, g in ticks.groupby("steamid"):
        g = g.sort_values("tick")
        tracks[sid] = {
            "tick": g["tick"].to_numpy(),
            "x": g["X"].to_numpy(dtype=float),
            "y": g["Y"].to_numpy(dtype=float),
            "z": g["Z"].to_numpy(dtype=float),
            "pitch": g["pitch"].to_numpy(dtype=float),
            "yaw": g["yaw"].to_numpy(dtype=float),
            # Metadata only (not detector features): ping matters because aim error is
            # measured against SERVER positions, which a high-ping player never saw.
            "ping": g["ping"].to_numpy(dtype=float) if "ping" in g else None,
            "duck": g["duck_amount"].to_numpy(dtype=float) if "duck_amount" in g else None,
        }
    return tracks


def slice_window(tr, start, end):
    """Indices of a player's rows with start <= tick <= end."""
    lo = np.searchsorted(tr["tick"], start, side="left")
    hi = np.searchsorted(tr["tick"], end, side="right")
    return lo, hi


def value_at(tr, key, tick):
    """Value of tr[key] at the last recorded tick <= tick (NaN if unavailable)."""
    arr = tr.get(key)
    if arr is None:
        return np.nan
    i = np.searchsorted(tr["tick"], tick, side="right") - 1
    return float(arr[i]) if i >= 0 else np.nan


def ideal_angles(ax, ay, az, vx, vy, vz, eye=EYE_HEIGHT, aim=None):
    """View angles that point from the attacker's eye at the aim point on the victim.

    Positions are player ORIGINS (feet), exactly as stored in the demo. The aim point is
    `aim` units above the victim's origin (default 0.8 * EYE_HEIGHT, the upper body).
    Returns (yaw_deg, pitch_deg, 3-D distance eye -> aim point).
    """
    if aim is None:
        aim = EYE_HEIGHT * 0.8
    dx, dy = vx - ax, vy - ay
    dz = (vz + aim) - (az + eye)
    tgt_yaw = np.degrees(np.arctan2(dy, dx))
    tgt_pitch = -np.degrees(np.arctan2(dz, np.hypot(dx, dy)))  # down = +
    return tgt_yaw, tgt_pitch, np.sqrt(dx**2 + dy**2 + dz**2)


def window_features(t, yaw, pitch, ax, ay, az, vx, vy, vz):
    """Aim features from ONE aligned window: every array is on the same ticks `t`, and the
    last element is the anchor (first shot). Positions are origins (feet).

    This is the single code path for both real kills and simulated (edited) traces, so
    the two can only differ because the view angles differ.
    """
    # Angular speed (deg/s) between consecutive ticks
    dyaw = wrap(np.diff(yaw))
    dpitch = np.diff(pitch)
    dt = np.diff(t).astype(float)
    ok = dt > 0
    if ok.sum() < 8:
        return None
    speed = np.hypot(dyaw, dpitch)[ok] / dt[ok] * TICKRATE
    accel = np.diff(speed) * TICKRATE

    # Aim error: angle between attacker's view and direction to victim
    tgt_yaw, tgt_pitch, dist3 = ideal_angles(ax, ay, az, vx, vy, vz)
    err = np.hypot(wrap(tgt_yaw - yaw), tgt_pitch - pitch)

    # Ticks (before the shot) since aim error last dropped below threshold
    # and stayed there = how long the player was "settled".
    above = np.where(err > SETTLE_DEG)[0]
    settled_ticks = (len(err) - 1 - above[-1]) if len(above) else len(err)

    # Range-independent version: miss distance in game units at the victim
    err_units = dist3 * np.tan(np.radians(np.minimum(err, 89.0)))
    above_u = np.where(err_units > SETTLE_UNITS)[0]
    settled_units = (len(err_units) - 1 - above_u[-1]) if len(above_u) else len(err_units)

    # Victim movement over the window (stationary / AFK targets are less informative)
    vdt = np.maximum(np.diff(t).astype(float), 1.0)
    vspeed = np.hypot(np.diff(vx), np.diff(vy)) / vdt * TICKRATE
    victim_speed_ups = float(vspeed.mean()) if len(vspeed) else np.nan

    dist = float(np.hypot(vx[-1] - ax[-1], vy[-1] - ay[-1]))
    return {
        "peak_speed_dps": float(speed.max()),
        "mean_speed_dps": float(speed.mean()),
        "speed_at_kill_dps": float(speed[-3:].mean()),
        "peak_accel": float(np.abs(accel).max()) if len(accel) else np.nan,
        "err_at_kill_deg": float(err[-1]),
        "err_mean_last10_deg": float(err[-10:].mean()),
        "err_min_deg": float(err.min()),
        "settled_ticks": int(settled_ticks),
        "err_units_at_kill": float(err_units[-1]),
        "err_units_mean_last10": float(err_units[-10:].mean()),
        "settled_ticks_units": int(settled_units),
        "distance_units": dist,
        "victim_speed_ups": victim_speed_ups,
    }


def kill_features(att, vic, kill_tick, with_window=False):
    """Features for the window ending at kill_tick (in practice the anchor = first shot).
    Attacker and victim are aligned on their common ticks before anything is computed."""
    lo, hi = slice_window(att, kill_tick - WINDOW, kill_tick)
    if hi - lo < WINDOW // 2:
        return None
    vlo, vhi = slice_window(vic, kill_tick - WINDOW, kill_tick)
    if vhi - vlo < WINDOW // 2:
        return None
    common, ai, vi = np.intersect1d(att["tick"][lo:hi], vic["tick"][vlo:vhi], return_indices=True)
    if len(common) < 8:
        return None
    win = {
        "t": common,
        "yaw": att["yaw"][lo:hi][ai], "pitch": att["pitch"][lo:hi][ai],
        "ax": att["x"][lo:hi][ai], "ay": att["y"][lo:hi][ai], "az": att["z"][lo:hi][ai],
        "vx": vic["x"][vlo:vhi][vi], "vy": vic["y"][vlo:vhi][vi], "vz": vic["z"][vlo:vhi][vi],
    }
    feats = window_features(**win)
    if feats is None:
        return None

    # Longer-term victim movement (5 s): ~0 means AFK / frozen / planting / defusing
    vlo5, vhi5 = slice_window(vic, kill_tick - 5 * TICKRATE, kill_tick)
    if vhi5 > vlo5:
        px, py = vic["x"][vlo5:vhi5], vic["y"][vlo5:vhi5]
        feats["victim_moved_5s"] = float(np.hypot(px - px[-1], py - py[-1]).max())
    else:
        feats["victim_moved_5s"] = np.nan
    return (feats, win) if with_window else feats


def process_demo(path: Path, source: str = "own") -> list:
    parser = DemoParser(str(path))
    header = parser.parse_header()
    map_name = header.get("map_name", "unknown")

    # Sanity check for tickrate: median ticks per round.
    # Compare against your own 64-tick matches. About double means 128 tick.
    try:
        rounds = parser.parse_event("round_end")
        tpr = int(np.median(np.diff(rounds["tick"].to_numpy())))
        print(f"    median ticks/round: {tpr}")
    except Exception:
        pass

    ticks = parser.parse_ticks(["X", "Y", "Z", "pitch", "yaw", "ping", "duck_amount"])
    kills = parser.parse_event("player_death")
    if kills is None or len(kills) == 0:
        return []

    tracks = build_player_tracks(ticks)

    # Shots and damage events, used to find where each engagement really began
    shots_by_player = {}
    try:
        shots = parser.parse_event("weapon_fire").dropna(subset=["user_steamid"])
        if "weapon" in shots.columns:
            # Grenade throws and knife swings also fire weapon_fire; they are not gunshots.
            w = shots["weapon"].astype(str).str.lower()
            shots = shots[~w.str.contains("|".join(NON_GUN_FIRE))]
        for sid, g in shots.groupby(shots["user_steamid"].astype("int64")):
            shots_by_player[int(sid)] = np.sort(g["tick"].to_numpy())
    except Exception as e:
        print(f"    (no weapon_fire data: {e})")
    hurts_by_attacker = {}   # attacker id -> (hit ticks, victim ids), sorted by tick
    try:
        hurts = parser.parse_event("player_hurt").dropna(
            subset=["attacker_steamid", "user_steamid"])
        hurts = hurts.assign(
            attacker_steamid=hurts["attacker_steamid"].astype("int64"),
            user_steamid=hurts["user_steamid"].astype("int64"),
        )
        hurts = hurts[hurts["attacker_steamid"] != hurts["user_steamid"]].sort_values("tick")
        for a_id, g in hurts.groupby("attacker_steamid"):
            hurts_by_attacker[int(a_id)] = (g["tick"].to_numpy(), g["user_steamid"].to_numpy())
    except Exception as e:
        print(f"    (no player_hurt data: {e})")

    rows = []
    for k in kills.itertuples(index=False):
        a, v = k.attacker_steamid, k.user_steamid
        if a is None or v is None or pd.isna(a) or pd.isna(v) or a == v:
            continue
        a, v = int(a), int(v)
        if a == 0 or v == 0:          # bots / unknown players share id 0
            continue
        if a not in tracks or v not in tracks:
            continue
        kill_tick = int(k.tick)

        # Anchor on the FIRST shot of the burst that killed the victim, because
        # by the last bullet of a spray the crosshair has drifted with recoil.
        anchor = kill_tick
        burst_shots, burst_hits, first_shot_hit, has_shot_data = 0, 0, False, False
        sh = shots_by_player.get(a)
        hb = hurts_by_attacker.get(a)
        if sh is not None:
            lookback = kill_tick - 3 * TICKRATE
            ref = kill_tick           # the burst must end at (or just after) the last hit on the victim
            ht_v = None
            if hb is not None:
                hticks, hvics = hb
                ht_v = hticks[(hvics == v) & (hticks <= kill_tick) & (hticks >= lookback)]
                ref = int(ht_v.max()) + 2 if len(ht_v) else None   # no recorded hit on victim -> no burst
            burst = None
            if ref is not None:
                recent = sh[(sh <= ref) & (sh >= lookback)]
                if len(recent):
                    i = len(recent) - 1
                    while i > 0 and recent[i] - recent[i - 1] <= BURST_GAP:
                        i -= 1
                    burst = recent[i:]
                    if hb is not None:
                        # If the attacker damaged someone ELSE mid-burst, the spray at
                        # this victim started after that hit.
                        other = hticks[(hvics != v) & (hticks >= burst[0]) & (hticks < kill_tick)]
                        if len(other):
                            later = burst[burst > other.max()]
                            burst = later if len(later) else burst[-1:]
                        # A real burst at this victim must contain a hit on them
                        if not (ht_v >= burst[0]).any():
                            burst = None
            if burst is not None and len(burst):
                anchor = int(burst[0])
                burst_shots = len(burst)
                has_shot_data = True
                if ht_v is not None:
                    burst_hits = int(((ht_v >= anchor) & (ht_v <= kill_tick)).sum())
                    first_shot_hit = bool(((ht_v >= anchor) & (ht_v <= anchor + 2)).any())

        res = kill_features(tracks[a], tracks[v], anchor, with_window=True)
        if res is None:
            continue
        feats, win = res
        feats.update({
            "anchor_tick": anchor,
            "kill_tick_to_first_shot": kill_tick - anchor,
            "burst_shots": burst_shots,
            "burst_hits": burst_hits,
            "first_shot_hit": first_shot_hit,
            "has_shot_data": has_shot_data,
        })
        feats.update({
            "demo": path.stem,
            "source": source,
            "map": map_name,
            "tick": int(k.tick),
            "player": anon(a),
            "weapon": getattr(k, "weapon", None),
            "headshot": bool(getattr(k, "headshot", False)),
            "penetrated": bool(getattr(k, "penetrated", False)),
            "thrusmoke": bool(getattr(k, "thrusmoke", False)),
            "noscope": bool(getattr(k, "noscope", False)),
            "attackerblind": bool(getattr(k, "attackerblind", False)),
            "attackerinair": bool(getattr(k, "attackerinair", False)),
            # Context columns, NOT used by the detector
            "attacker_ping_ms": value_at(tracks[a], "ping", anchor),
            "attacker_duck": value_at(tracks[a], "duck", anchor),
            "victim_duck": value_at(tracks[v], "duck", anchor),
        })
        rows.append(feats)
        if TRACE_SINK is not None:
            TRACE_SINK.append({**win, "meta": dict(feats)})
    return rows


def main():
    """
    python features.py                 # own demos (.dem files in this folder)
                                       # + pro demos (prodemos/, searched recursively)
    python features.py --windows       # also save raw per-kill windows to windows.pkl
    python features.py 128             # legacy: treat the pro demos as 128 tick
    Output: kill_features.csv with a 'source' column ('own' or 'pro').
    """
    global TICKRATE, WINDOW, TRACE_SINK
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    save_windows = "--windows" in sys.argv[1:]
    own_dir = Path(".")
    pro_dir = Path("prodemos")
    own_tickrate = 64
    pro_tickrate = int(args[0]) if args else 64
    out = "kill_features.csv"
    if save_windows:
        TRACE_SINK = []

    groups = [
        ("own", sorted(own_dir.glob("*.dem")), own_tickrate),          # top level only
        ("pro", sorted(pro_dir.rglob("*.dem")) if pro_dir.exists() else [], pro_tickrate),
    ]
    if not pro_dir.exists():
        print("WARNING: 'prodemos' folder not found next to this script")

    all_rows = []
    for source, demos, tickrate in groups:
        TICKRATE = tickrate
        WINDOW = tickrate  # keep the window at 1 second
        print(f"\n[{source}] {len(demos)} demos, tickrate={tickrate}")
        for d in demos:
            try:
                rows = process_demo(d, source)
                print(f"  {d.name}: {len(rows)} kills")
                all_rows.extend(rows)
            except Exception as e:
                print(f"  {d.name}: FAILED ({e})")

    df = pd.DataFrame(all_rows)
    try:
        df.to_csv(out, index=False)
    except PermissionError:
        # File is probably open in Excel or locked by OneDrive. Don't lose the run.
        import time
        out = f"kill_features_{int(time.time())}.csv"
        df.to_csv(out, index=False)
        print(f"kill_features.csv was locked (open in Excel?), saved to {out} instead")
    print(f"\nWrote {len(df)} rows to {out}")
    if save_windows:
        with open("windows.pkl", "wb") as fh:
            pickle.dump(TRACE_SINK, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Wrote {len(TRACE_SINK)} raw windows to windows.pkl (local use only, do not share)")
    if len(df):
        print(df.groupby("source").size().to_string())
        print()
        print(
            df.groupby("source")[
                ["peak_speed_dps", "err_at_kill_deg", "settled_ticks", "distance_units"]
            ].median().round(2)
        )


if __name__ == "__main__":
    main()
