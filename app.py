import os
import requests
import pandas as pd
import streamlit as st

# --- Streamlit config MUST be first ---
st.set_page_config(page_title="NFL Viz", layout="wide")

# --- Defaults from environment (overridable in sidebar) ---
ENV_DEFAULT_API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8051")
ENV_DEFAULT_API_TOKEN = os.getenv("API_TOKEN", "")

# --- Persist settings in session ---
if "API_BASE" not in st.session_state:
    st.session_state["API_BASE"] = ENV_DEFAULT_API_BASE
if "API_TOKEN" not in st.session_state:
    st.session_state["API_TOKEN"] = ENV_DEFAULT_API_TOKEN
if "teams" not in st.session_state:
    st.session_state["teams"] = []

# --- Helpers ---
def _headers_api_key(token: str):
    # Backend expects x-api-key (case-insensitive)
    return {"x-api-key": token} if token else {}

def api_get(path: str, params=None, *, base=None, token=None, timeout=20):
    """GET helper that always sends x-api-key from sidebar/session (or overrides)."""
    base = (base or st.session_state["API_BASE"]).rstrip("/")
    token = token if token is not None else st.session_state["API_TOKEN"]
    url = f"{base}{path}"
    try:
        r = requests.get(url, headers=_headers_api_key(token), params=(params or {}), timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError:
        st.error(f"HTTP {r.status_code} on {url}\nResponse: {r.text[:800]}")
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"API unreachable: {e}\nTried: {url}")
        st.stop()

# ---- UI ----
st.title("NFL Viz Dashboard")

# --- Sidebar overrides ---
with st.sidebar:
    st.title("Settings")
    st.session_state["API_BASE"] = st.text_input("API Base", st.session_state["API_BASE"])
    st.session_state["API_TOKEN"] = st.text_input("API Token", st.session_state["API_TOKEN"], type="password")
    if not st.session_state["API_TOKEN"]:
        st.caption("⚠️ API_TOKEN empty — secured endpoints will return 401.")

        # ---- Player search helper ----
        if st.button("Find Player"):
            q = st.text_input("Search player name or id", value="", key="find_player_query")
            if q.strip():
                matches = api_get("/players", params={"q": q.strip(), "limit": 10})
                if not matches:
                    st.warning("No players found.")
                else:
                    def label_player(m: dict) -> str:
                        # Robust fallbacks for name / team / position
                        name = (
                            m.get("player_display_name")
                            or m.get("full_name")
                            or m.get("display_name")
                            or m.get("gsis_name")
                            or m.get("name")
                            or "Unknown"
                        )
                        team = (
                            m.get("recent_team")
                            or m.get("team")
                            or m.get("current_team")
                            or "--"
                        )
                        pos = m.get("position") or m.get("pos") or "--"
                        pid = m.get("player_id", "?")
                        return f"{name} · {team} · {pos} · {pid}"

                    options = [label_player(m) for m in matches]

                    if len(matches) == 1:
                        st.success(f"Found: {options[0]}")
                        st.session_state["selected_player_id"] = matches[0].get("player_id")
                    else:
                        choice = st.selectbox("Select player", options, key="find_player_choice")
                        if choice:
                            idx = options.index(choice)
                            st.session_state["selected_player_id"] = matches[idx].get("player_id")

# Fetch Teams
if st.button("Fetch Teams"):
    with st.spinner("Fetching teams..."):
        teams = api_get("/teams")
        st.session_state["teams"] = teams
        st.success(f"Fetched {len(teams)} teams!")
        st.json(teams)

teams = st.session_state.get("teams", [])

# Tabs
tab_leader, tab_player, tab_fantasy = st.tabs(["🏆 Leaderboard", "📈 Game Log", "🎯 Fantasy"])

# ------- helper: normalize various API response shapes into list-of-rows -------
def normalize_rows(payload):
    """
    Accepts:
      - list[dict]                        -> returns as-is
      - {"data": [...]} / {"results": [...]} / {"rows": [...]} / {"items": [...]} / {"records": [...]}
      - {"colA":[...], "colB":[...]}     -> converts parallel arrays to list of dicts
      - {"detail": "..."}                -> raises a readable error
    Returns: list[dict]
    """
    if isinstance(payload, list):
        if payload and not isinstance(payload[0], dict):
            return []
        return payload

    if isinstance(payload, dict):
        if "detail" in payload and isinstance(payload["detail"], str):
            raise RuntimeError(f"API error: {payload['detail']}")
        for key in ("results", "data", "rows", "items", "records"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        if payload and all(isinstance(v, list) for v in payload.values()):
            lengths = {len(v) for v in payload.values()}
            if len(lengths) == 1:
                n = lengths.pop()
                keys = list(payload.keys())
                return [{k: payload[k][i] for k in keys} for i in range(n)]
    return []

# -------------------------  Leaderboard tab  -------------------------
with tab_leader:
    st.header("Leaderboard")
    colA, colB, colC, colD, colE = st.columns(5)
    season_from = colA.number_input("Season From", value=2025, step=1, key="leader_season_from")
    season_to   = colB.number_input("Season To",   value=2025, step=1, key="leader_season_to")
    week_from   = colC.number_input("Week From",   value=1,    step=1, min_value=1, key="leader_week_from")
    week_to     = colD.number_input("Week To",     value=4,    step=1, min_value=1, max_value=22, key="leader_week_to")
    position    = colE.selectbox("Position", options=["", "QB","RB","WR","TE"], index=0, key="leader_pos")

    team_names = [""] + [t.get("team") for t in teams] if teams else [""]
    team = st.selectbox("Team (optional)", options=team_names, index=0, key="leader_team")

    if st.button("Fetch Leaderboard"):
        with st.spinner("Fetching leaderboard..."):
            params = {
                "season_from": season_from, "season_to": season_to,
                "week_from": week_from, "week_to": week_to,
                "position": position or None, "team": team or None,
            }
            params = {k: v for k, v in params.items() if v not in ("", None)}

            st.caption(
                f"→ GET {st.session_state['API_BASE']}/leaderboard {params} "
                f"| x-api-key set: {'yes' if st.session_state['API_TOKEN'] else 'no'}"
            )

            payload = api_get("/leaderboard", params=params)

            with st.expander("Raw API result", expanded=False):
                st.write(type(payload))
                st.json(payload)

            rows = normalize_rows(payload)

            if not rows:
                st.warning("No rows parsed from API response. If the raw JSON above looks correct, tell me its exact shape so I can add that case.")
            else:
                df = pd.DataFrame(rows)
                st.success(f"Loaded {len(df)} rows; columns: {list(df.columns)}")
                st.dataframe(df, use_container_width=True)

# -------------------------
# Game Log tab (ready-to-paste)
# -------------------------
with tab_player:
    st.header("Player Game Log")

    # One unified search bar (name or id) + optional picker
    colL, colR = st.columns([3,1])
    user_query = colL.text_input("Player (name or id)", value=st.session_state.get("find_player_query", ""), key="find_player_query")
    search_clicked = colR.button("Search", key="player_search_btn")

    # Helper to format options cleanly
    def _label_from_row(m: dict) -> str:
        pid  = m.get("player_id") or "?"
        name = (m.get("player_display_name") or m.get("full_name")
                or m.get("display_name") or m.get("gsis_name") or m.get("name") or "Unknown")
        team = m.get("recent_team") or m.get("team") or m.get("current_team") or "--"
        pos  = m.get("position") or m.get("pos") or "--"
        return f"{name} · {team} · {pos} · {pid}"

    # If user hit Search, query /players once and persist results
    if search_clicked and user_query.strip():
        try:
            st.session_state["player_search_results"] = api_get("/players", params={"q": user_query.strip(), "limit": 25}) or []
        except Exception:
            st.session_state["player_search_results"] = []

        # If exactly one match, auto-select it
        if len(st.session_state["player_search_results"]) == 1:
            only = st.session_state["player_search_results"][0]
            st.session_state["selected_player_id"] = only.get("player_id")

    results = st.session_state.get("player_search_results", [])

    # If we have results, show a sticky picker; otherwise allow direct-ID usage
    if results:
        labels = [_label_from_row(m) for m in results]
        # remember last choice
        default_idx = st.session_state.get("player_choice_idx", 0)
        idx = st.selectbox(
            "Select player",
            options=range(len(results)),
            format_func=lambda i: labels[i],
            index=min(default_idx, max(0, len(results)-1)),
            key="player_choice_idx",
        )
        st.session_state["selected_player_id"] = results[idx].get("player_id")
        st.caption(f"Selected: {labels[idx]}")
    else:
        # If user typed a full-looking id (e.g., 00-0031381), accept it directly
        if user_query and "-" in user_query and len(user_query) >= 6:
            st.session_state["selected_player_id"] = user_query.strip()

    # ---- Window & controls
    col1, col2, col3, col4 = st.columns(4)
    g_season_from = col1.number_input("Season From", value=2025, step=1, key="game_season_from")
    g_season_to   = col2.number_input("Season To",   value=2025, step=1, key="game_season_to")
    g_week_from   = col3.number_input("Week From",   value=1,    step=1, min_value=1, key="game_week_from")
    g_week_to     = col4.number_input("Week To",     value=17,   step=1, min_value=1, max_value=22, key="game_week_to")
    glimit = st.slider("Max rows", 10, 500, 200, step=10, key="game_limit")


    # --- Main action button ---
    if st.button("Fetch Game Log", key="game_go"):
        # Priority: chosen via picker → text box → error
        pid = st.session_state.get("selected_player_id") or _resolve_player_id(q)
        if not pid:
            st.warning("Enter a player name/id or use 'Find Player' to select one.")
        else:
            params = {
                "season_from": g_season_from, "season_to": g_season_to,
                "week_from": g_week_from, "week_to": g_week_to,
                "limit": glimit, "offset": 0
            }
            rows = api_get(f"/game-log/enriched/{pid}", params=params)
            df = pd.DataFrame(rows)

            if df.empty:
                st.warning("No rows for this window.")
            else:
                # Sort + clean
                if {"season","week"}.issubset(df.columns):
                    df = df.sort_values(["season","week"])

                # Title
                title = df.get("player_display_name")
                st.subheader(title.iloc[0] if isinstance(title, pd.Series) and not title.empty else "Game Log")

                # Coerce numerics (safe if missing)
                for col in [
                    "season","week","targets","carries","air_yards",
                    "receptions","receiving_yards","rushing_yards",
                    "passing_yards","passing_tds","interceptions",
                    "receiving_tds","rushing_tds","fumbles_lost",
                    "two_point_conversions","snap_pct","long_rec","long_rush"
                ]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                # Preferred column order: puts longest plays next to their families
                preferred = [
                    # Context
                    "season",                 # (int; no commas)
                    "week",
                    "player_display_name",
                    "position",
                    "recent_team",
                    "opponent_team",

                    # Receiving
                    "targets",
                    "receptions",
                    "receiving_yards",
                    "receiving_tds",
                    "air_yards",
                    "long_rec",

                    # Rushing
                    "carries",
                    "rushing_yards",
                    "rushing_tds",
                    "long_rush",

                    # Passing
                    "passing_yards",
                    "passing_tds",
                    "interceptions",

                    # Misc
                    "fumbles_lost",
                    "two_point_conversions",
                    "player_id",
                ] 

                # Ensure numeric season/week (so they render as plain integers)
                for _c in ("season", "week"):
                    if _c in df.columns:
                        df[_c] = pd.to_numeric(df[_c], errors="coerce").astype("Int64")

                # Build final order: requested columns that exist, then any extras at the end
                ordered_cols = [c for c in preferred if c in df.columns]
                ordered_cols += [c for c in df.columns if c not in ordered_cols]

                # Snap% hint if missing
                if "snap_pct" not in df.columns or df["snap_pct"].isna().all():
                    st.caption("Snap% not available yet for this range (data source has not published it).")

                st.dataframe(df[ordered_cols], use_container_width=True)

# -------------------------
# Fantasy tab
# -------------------------
with tab_fantasy:
    st.header("Fantasy Insights")

    # Controls
    colA, colB, colC = st.columns(3)
    f_season = colA.number_input("Season", value=2025, step=1, key="fantasy_season")
    f_week   = colB.number_input("Week", value=1, step=1, min_value=1, max_value=22, key="fantasy_week")

    # Team dropdown from cached teams (optional)
    team_names = [""] + [t.get("team") for t in teams] if teams else [""]
    f_team = colC.selectbox("Team (optional)", options=team_names, index=0, key="fantasy_team")

    f_player_q = st.text_input("Player (name or id fragment, optional)", value="", key="fantasy_player_q")
    flimit = st.slider("Max rows", 10, 500, 200, step=10, key="fantasy_limit")

    if st.button("Fetch Fantasy", key="fantasy_fetch"):
        params = {
            "season": f_season,
            "week": f_week,
            "team": f_team or None,
            "q": (f_player_q.strip() or None),
            "limit": flimit,
            "offset": 0,
        }
        params = {k: v for k, v in params.items() if v not in ("", None)}

        # Show the exact call we’ll make
        st.caption(
            f"→ GET {st.session_state['API_BASE']}/fantasy-insights {params} "
            f"| x-api-key set: {'yes' if st.session_state['API_TOKEN'] else 'no'}"
        )

        # Call primary endpoint; if 404 fallback to /fantasy
        try:
            payload = api_get("/fantasy-insights", params=params)
        except Exception as e:
            # If backend returns a 404 shaped error, try alternate route name
            try:
                payload = api_get("/fantasy", params=params)
                st.info("Fell back to /fantasy endpoint.")
            except Exception:
                raise  # preserve original streamlit error flow

        # Normalize to list-of-rows
        rows = normalize_rows(payload)
        if not rows:
            st.warning("No rows parsed from API response. If the raw JSON above looks correct, tell me its shape so I can add a case.")
        else:
            df = pd.DataFrame(rows)

            # Try to surface a unified fantasy_points column if the table uses different names
            fp_candidates = ["fantasy_points_ppr", "fantasy_points", "ppr_points", "points"]
            if "fantasy_points" not in df.columns:
                for c in fp_candidates:
                    if c in df.columns:
                        df["fantasy_points"] = pd.to_numeric(df[c], errors="coerce")
                        break

            # Coerce common numeric fields (won’t error if missing)
            for col in [
                "season","week",
                "fantasy_points",
                "targets","receptions","receiving_yards","receiving_tds",
                "carries","rushing_yards","rushing_tds",
                "passing_yards","passing_tds","interceptions",
                "fumbles_lost","two_point_conversions",
            ]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Preferred order – we’ll include what exists and then append any extras
            preferred = [
                "season","week",
                "player_display_name","position","recent_team","opponent_team",
                "fantasy_points",
                "targets","receptions","receiving_yards","receiving_tds","air_yards",
                "carries","rushing_yards","rushing_tds",
                "passing_yards","passing_tds","interceptions",
                "fumbles_lost","two_point_conversions",
                "player_id",
            ]
            ordered_cols = [c for c in preferred if c in df.columns]
            ordered_cols += [c for c in df.columns if c not in ordered_cols]

            st.success(f"Loaded {len(df)} rows; columns: {list(df.columns)}")
            with st.expander("Raw API result", expanded=False):
                st.write(type(payload))
                st.json(payload)

            st.dataframe(df[ordered_cols], use_container_width=True)