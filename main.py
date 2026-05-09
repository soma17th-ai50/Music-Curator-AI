"""
Streamlit chat UI for the Music Recommendation Agent.

Run:  streamlit run main.py
"""
import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Music Curator AI",
    page_icon="🎵",
    layout="wide",
)

# ── Session state init ─────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role": str, "content": str, "recs": list}
if "graph" not in st.session_state:
    st.session_state.graph = None
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []  # LangChain messages for the graph
if "profile" not in st.session_state:
    st.session_state.profile = {}
if "last_recs" not in st.session_state:
    st.session_state.last_recs = []


@st.cache_resource
def load_graph():
    from src.agent.graph import get_graph
    return get_graph()


@st.cache_resource
def load_dataset():
    from src.music.dataset import MusicDataset
    return MusicDataset()


def load_profile():
    from src.memory.profile import UserProfileManager
    mgr = UserProfileManager()
    return mgr.get_profile(st.session_state.session_id)


# ── Helpers ────────────────────────────────────────────────────────────────
def _empty_state() -> dict:
    return {
        "user_input":          "",
        "session_id":          st.session_state.session_id,
        "messages":            st.session_state.agent_messages,
        "intent":              None,
        "mood":                None,
        "activity":            None,
        "context":             None,
        "feedback_type":       None,
        "liked_artists_input": None,
        "user_profile":        None,
        "candidates":          None,
        "recommendations":     None,
        "button_event":        None,
        "target_track_id":     None,
        "target_track_name":   None,
        "target_track_artist": None,
        "response":            None,
    }


def run_agent(user_input: str) -> dict:
    graph = load_graph()
    state = _empty_state()
    state["user_input"] = user_input
    result = graph.invoke(state)
    st.session_state.agent_messages = (
        st.session_state.agent_messages + result.get("messages", [])
    )[-20:]
    return result


def run_agent_button(track: dict, feedback_type: str) -> dict:
    """👍/👎 버튼 클릭 → 그래프를 feedback intent로 invoke."""
    graph = load_graph()
    state = _empty_state()
    state.update({
        "user_input":          f"[BUTTON_{feedback_type.upper()}] {track.get('track_name')} — {track.get('artist_name')}",
        "intent":              "feedback",
        "feedback_type":       feedback_type,
        "button_event":        True,
        "target_track_id":     track.get("track_id"),
        "target_track_name":   track.get("track_name"),
        "target_track_artist": track.get("artist_name"),
    })
    result = graph.invoke(state)
    st.session_state.agent_messages = (
        st.session_state.agent_messages + result.get("messages", [])
    )[-20:]
    return result


def _spotify_url(track_id: str) -> str:
    return f"https://open.spotify.com/track/{track_id}"


def render_recommendation_cards(recs: list, key_prefix: str = "live"):
    if not recs:
        return

    cols = st.columns(min(len(recs), 3))
    for i, track in enumerate(recs[:6]):
        col = cols[i % 3]
        track_id = track.get("track_id", "")
        artist_name = track.get("artist_name", "")
        # Spotify 링크: 클릭 시 부모 윈도우 URL에 ?clicked_track=... 를 심고
        # popstate를 트리거 → 다음 rerun 때 main()이 query_params에서 읽어 기록.
        if track_id:
            onclick = (
                "const u = new URL(window.parent.location.href);"
                f"u.searchParams.set('clicked_track', '{track_id}');"
                "window.parent.history.replaceState({}, '', u);"
                "window.parent.dispatchEvent(new Event('popstate'));"
            )
            spotify_btn = (
                f'<a href="{_spotify_url(track_id)}" target="_blank" '
                f'onclick="{onclick}" style="'
                'display:inline-block; margin-top:10px; padding:5px 12px;'
                'background:#1DB954; color:#fff; border-radius:20px;'
                'font-size:0.78em; font-weight:bold; text-decoration:none;">▶ Spotify에서 듣기</a>'
            )
        else:
            spotify_btn = ""
        with col:
            st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    color: #e0e0e0;
">
    <div style="font-size:1.05em; font-weight:bold; color:#e94560; margin-bottom:4px;">
        🎵 {track.get('track_name','?')}
    </div>
    <div style="font-size:0.9em; color:#a0a0b0; margin-bottom:8px;">
        🎤 {artist_name or '?'}
    </div>
    <div style="font-size:0.78em; color:#a0c4d8; margin-bottom:8px; line-height:1.4;">
        {track.get('reason', '')}
    </div>
    <div style="font-size:0.75em; color:#606070;">
        💃 {track.get('danceability', 0):.0%} &nbsp;
        ⚡ {track.get('energy', 0):.0%} &nbsp;
        😊 {track.get('valence', 0):.0%}
    </div>
    {spotify_btn}
</div>
""", unsafe_allow_html=True)

            if track_id:
                like_col, dislike_col = st.columns(2)
                if like_col.button("👍", key=f"like_{key_prefix}_{track_id}_{i}", use_container_width=True):
                    st.session_state.pending_button = (track, "like")
                    st.rerun()
                if dislike_col.button("👎", key=f"dislike_{key_prefix}_{track_id}_{i}", use_container_width=True):
                    st.session_state.pending_button = (track, "dislike")
                    st.rerun()


def render_profile_sidebar(profile: dict):
    from src.memory.profile import UserProfileManager

    st.sidebar.header("👤 내 취향 프로필")

    # ── 등록된 아티스트 목록 (삭제 버튼 포함) ─────────────────────────────
    st.sidebar.markdown("**좋아하는 아티스트**")
    liked = profile.get("liked_artists", [])
    if liked:
        for artist in liked:
            a_col, d_col = st.sidebar.columns([4, 1])
            a_col.markdown(f"• {artist}")
            if d_col.button("✕", key=f"del_{artist}"):
                mgr = UserProfileManager()
                p = mgr.get_profile(st.session_state.session_id)
                p["liked_artists"] = [a for a in p["liked_artists"] if a != artist]
                mgr.save_profile(st.session_state.session_id, p)
                st.session_state.profile = mgr.get_profile(st.session_state.session_id)
                st.rerun()
    else:
        st.sidebar.caption("아직 등록된 아티스트가 없습니다.")

    st.sidebar.divider()

    liked_tracks = profile.get("liked_tracks", [])
    st.sidebar.markdown(f"**👍 좋아요한 곡:** {len(liked_tracks)}개")
    st.sidebar.markdown(f"**👎 별로인 곡:** {len(profile.get('disliked_tracks', []))}개")

    st.sidebar.divider()

    # ── 대화 초기화 ────────────────────────────────────────────────────────
    if st.sidebar.button("🗨️ 대화 초기화", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.agent_messages = []
        st.rerun()

    # ── 프로필 초기화 ──────────────────────────────────────────────────────
    if st.sidebar.button("🗑️ 프로필 초기화", use_container_width=True):
        mgr = UserProfileManager()
        mgr.save_profile(st.session_state.session_id, mgr._empty_profile())
        st.session_state.profile = {}
        st.rerun()


# ── Main UI ────────────────────────────────────────────────────────────────
def main():
    # Pre-load dataset (cached)
    try:
        load_dataset()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # Spotify 링크 클릭 시그널 (이전 인터랙션에서 onclick으로 심긴 query param 회수)
    clicked = st.query_params.get("clicked_track")
    if clicked:
        from src.memory.profile import UserProfileManager
        UserProfileManager().add_clicked_track(st.session_state.session_id, clicked)
        st.query_params.clear()

    # Profile sidebar
    st.session_state.profile = load_profile()
    render_profile_sidebar(st.session_state.profile)

    # Sidebar — session info
    st.sidebar.divider()
    st.sidebar.caption(f"세션 ID: `{st.session_state.session_id[:8]}…`")

    # ── Header ──────────────────────────────────────────────────────────────
    st.title("🎵 Music Curator AI")
    st.caption("상황과 기분에 맞는 음악을 추천받고, 왜 그 곡인지 설명도 들어보세요.")

    # ── Cold-start 온보딩 (빈 프로필 + 첫 진입) ─────────────────────────────
    needs_onboarding = (
        not st.session_state.chat_history
        and not st.session_state.profile.get("liked_artists")
        and not st.session_state.get("onboarding_skipped")
    )

    if needs_onboarding:
        with st.container(border=True):
            st.markdown("### 👋 처음 오셨네요!")
            st.caption("좋아하는 아티스트를 알려주시면 추천이 훨씬 정확해져요. (3명 권장, 쉼표로 구분)")
            with st.form("onboarding_form", clear_on_submit=True):
                artists_input = st.text_input(
                    "좋아하는 아티스트",
                    placeholder="예: IU, 검정치마, 새소년",
                    label_visibility="collapsed",
                )
                col_submit, col_skip = st.columns([3, 1])
                submitted = col_submit.form_submit_button("등록하기", use_container_width=True, type="primary")
                skipped = col_skip.form_submit_button("나중에", use_container_width=True)

            if submitted and artists_input.strip():
                from src.memory.profile import UserProfileManager
                mgr = UserProfileManager()
                names = [a.strip() for a in artists_input.split(",") if a.strip()]
                for name in names:
                    mgr.add_liked_artist(st.session_state.session_id, name)
                st.session_state.profile = load_profile()
                st.rerun()
            if skipped:
                st.session_state.onboarding_skipped = True
                st.rerun()

    # ── Quick-start chips (온보딩 완료 또는 건너뛴 사용자) ──────────────────
    elif not st.session_state.chat_history:
        st.markdown("**빠른 시작 ↓**")
        chip_cols = st.columns(4)
        chips = [
            ("☀️ 오늘 기분 좋은 음악",         "오늘 기분이 좋아서 신나는 음악 추천해줘"),
            ("🌧️ 집중해서 공부할 때",           "공부할 때 들으면 좋은 집중력 올려주는 음악"),
            ("🏃 운동할 때 들을 에너지 넘치는 곡", "운동할 때 들을 에너지 넘치는 곡 추천해줘"),
            ("🌙 잠들기 전 잔잔한 음악",         "잠들기 전에 들을 잔잔한 음악 추천해줘"),
        ]
        for col, (label, msg) in zip(chip_cols, chips):
            if col.button(label, use_container_width=True):
                st.session_state.pending_input = msg
                st.rerun()

    # ── Chat history ────────────────────────────────────────────────────────
    for entry_idx, entry in enumerate(st.session_state.chat_history):
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("recs"):
                render_recommendation_cards(entry["recs"], key_prefix=f"hist{entry_idx}")

    # ── Handle button feedback (👍/👎) ──────────────────────────────────────
    pending_button = st.session_state.pop("pending_button", None)
    if pending_button:
        track, fb_type = pending_button
        ack = "👍 좋아요" if fb_type == "like" else "👎 별로"
        user_msg = f"{ack}: {track.get('track_name')} — {track.get('artist_name')}"
        with st.chat_message("user"):
            st.markdown(user_msg)
        st.session_state.chat_history.append({"role": "user", "content": user_msg, "recs": []})

        with st.chat_message("assistant"):
            with st.spinner("피드백 반영 중..."):
                try:
                    result = run_agent_button(track, fb_type)
                    response = result.get("response", "")
                    recs = result.get("recommendations") or []
                except Exception as e:
                    response = f"오류가 발생했습니다: {e}"
                    recs = []
            st.markdown(response)
            if recs:
                render_recommendation_cards(recs, key_prefix="live")

        st.session_state.chat_history.append({
            "role": "assistant", "content": response, "recs": recs,
        })
        st.session_state.last_recs = recs
        st.session_state.profile = load_profile()
        st.rerun()

    # ── Handle quick-chip input ─────────────────────────────────────────────
    pending = st.session_state.pop("pending_input", None)

    # ── Chat input ──────────────────────────────────────────────────────────
    user_input = st.chat_input("어떤 음악이 듣고 싶으세요? (ex: 비 오는 날 카페에서 듣기 좋은 음악)")
    if user_input is None and pending:
        user_input = pending

    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input, "recs": []})

        # Run agent with spinner
        with st.chat_message("assistant"):
            with st.spinner("음악 찾는 중..."):
                try:
                    result = run_agent(user_input)
                    response = result.get("response", "죄송해요, 잠시 후 다시 시도해 주세요.")
                    recs = result.get("recommendations") or []
                except Exception as e:
                    response = f"오류가 발생했습니다: {e}"
                    recs = []

            st.markdown(response)
            if recs:
                render_recommendation_cards(recs, key_prefix="live")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "recs": recs,
        })
        st.session_state.last_recs = recs

        # Update profile display
        st.session_state.profile = load_profile()
        st.rerun()


if __name__ == "__main__":
    main()
