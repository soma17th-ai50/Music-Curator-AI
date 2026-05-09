"""
LangGraph nodes for the music recommendation agent.

Flow:
  analyze_input → [route_intent]
    "recommend" → load_profile → search_music → generate_response
    "feedback"  → load_profile → update_profile → search_music → generate_response
    "setup"     → load_profile → update_profile → generate_response
    "chat"      → generate_response
"""
import os
from typing import List, Literal, Optional

from langchain_upstage import ChatUpstage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..memory.profile import UserProfileManager
from ..music.dataset import MusicDataset
from .state import AgentState

# ---------------------------------------------------------------------------
# LangSmith tracing — enabled automatically when env vars are set
# ---------------------------------------------------------------------------
os.environ.setdefault("LANGSMITH_TRACING_V2",  os.getenv("LANGSMITH_TRACING_V2", "false"))
os.environ.setdefault("LANGSMITH_ENDPOINT",    os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"))
os.environ.setdefault("LANGSMITH_PROJECT",     os.getenv("LANGSMITH_PROJECT", "music-recommendation-agent"))

# ---------------------------------------------------------------------------
# LLM — Upstage Solar Pro 2
# ---------------------------------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "solar-pro2")
_llm = ChatUpstage(model=LLM_MODEL)


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------
class IntentResult(BaseModel):
    intent: Literal["recommend", "feedback", "setup", "chat"] = Field(
        description="recommend=음악 추천 요청, feedback=이전 추천 반응, setup=초기 취향 설정, chat=일반 대화"
    )
    mood: Optional[str] = Field(None, description="감정/분위기 키워드 (한 단어)")
    activity: Optional[str] = Field(None, description="활동 키워드 (한 단어)")
    context: Optional[str] = Field(None, description="추가 맥락 (장소, 날씨 등)")
    feedback_type: Optional[Literal["like", "dislike", "neutral"]] = Field(
        None, description="이전 추천에 대한 반응"
    )
    liked_artists: Optional[List[str]] = Field(
        None, description="사용자가 언급한 좋아하는 아티스트 목록"
    )


class RecommendationItem(BaseModel):
    index: int = Field(description="후보 목록 번호 (1부터 시작)")
    reason: str = Field(description="왜 이 곡인지 1~2문장 설명")


class RecommendationOutput(BaseModel):
    intro: str = Field(description="추천 전 짧은 인삿말")
    items: List[RecommendationItem]
    outro: str = Field(description="추천 후 피드백 유도 문장")


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------
_intent_llm = _llm.with_structured_output(IntentResult)


def _recent_messages(state: AgentState, n: int = 6) -> list:
    return state.get("messages", [])[-n:]


# ---------------------------------------------------------------------------
# Node 1: analyze_input
# ---------------------------------------------------------------------------
def analyze_input(state: AgentState) -> dict:
    """Classify user intent and extract mood / activity / context."""
    # Button-click feedback: skip LLM, intent/feedback_type 이미 state에 세팅됨
    if state.get("button_event"):
        return {
            "intent":        state.get("intent"),
            "feedback_type": state.get("feedback_type"),
            "messages":      [HumanMessage(content=state["user_input"])],
        }

    system = SystemMessage(content="""
당신은 음악 추천 에이전트의 의도 분류기입니다.
사용자 메시지를 분석해 intent, mood, activity, context, feedback_type, liked_artists를 추출하세요.

- intent 분류 기준:
  recommend : "추천해줘", "들을 게 없어", "~한 음악", "뭐 들을까" 등
  feedback  : "좋아", "별로야", "이건 별로", "좋았어", "다시 추천", "비슷한 걸로" 등
  setup     : 처음 인사이거나 좋아하는 아티스트/장르를 알려줄 때
  chat      : 그 외 일반 대화

- mood 예시: happy, sad, energetic, calm, focused, party, sleep, angry, romantic, chill 등
- activity 예시: working, studying, exercising, relaxing, cooking, commuting, sleeping, driving 등
""")

    messages = [system] + _recent_messages(state) + [
        HumanMessage(content=state["user_input"])
    ]
    result: Optional[IntentResult] = _intent_llm.invoke(messages)

    # LLM이 스키마에 맞는 응답을 못 주면 None 반환 → chat으로 폴백
    if result is None:
        return {
            "intent": "chat",
            "messages": [HumanMessage(content=state["user_input"])],
        }

    return {
        "intent":              result.intent,
        "mood":                result.mood,
        "activity":            result.activity,
        "context":             result.context,
        "feedback_type":       result.feedback_type,
        "liked_artists_input": result.liked_artists,
        "messages": [HumanMessage(content=state["user_input"])],
    }


# ---------------------------------------------------------------------------
# Node 2: load_profile
# ---------------------------------------------------------------------------
def load_profile(state: AgentState) -> dict:
    manager = UserProfileManager()
    profile = manager.get_profile(state["session_id"])

    # Track mood history
    if state.get("mood"):
        manager.record_mood(state["session_id"], state["mood"])
        profile = manager.get_profile(state["session_id"])

    return {"user_profile": profile}


# ---------------------------------------------------------------------------
# Node 3: update_profile
# ---------------------------------------------------------------------------
def update_profile(state: AgentState) -> dict:
    """Handle both initial setup and like/dislike feedback."""
    manager = UserProfileManager()
    session = state["session_id"]
    intent = state.get("intent")
    profile = state.get("user_profile", {})

    if intent == "setup" and state.get("liked_artists_input"):
        for artist in state["liked_artists_input"]:
            manager.add_liked_artist(session, artist)

    elif intent == "feedback":
        fb = state.get("feedback_type")
        target_id = state.get("target_track_id")
        target_artist = state.get("target_track_artist")
        if target_id:
            # 버튼 클릭: 정확한 곡이 지정됨
            if fb == "like":
                manager.add_liked_track(session, target_id)
                if target_artist:
                    manager.add_liked_artist(session, target_artist)
            elif fb == "dislike":
                manager.add_disliked_track(session, target_id)
        else:
            # 텍스트 발화: 직전 추천 첫 곡으로 폴백
            recs = state.get("recommendations") or []
            if recs:
                first = recs[0]
                if fb == "like":
                    manager.add_liked_track(session, first["track_id"])
                    manager.add_liked_artist(session, first["artist_name"])
                elif fb == "dislike":
                    manager.add_disliked_track(session, first["track_id"])

    updated = manager.get_profile(session)
    return {"user_profile": updated}


# ---------------------------------------------------------------------------
# Node 4: search_music
# ---------------------------------------------------------------------------
def search_music(state: AgentState) -> dict:
    dataset = MusicDataset()
    profile = state.get("user_profile") or {}

    candidates = dataset.search(
        mood=state.get("mood"),
        activity=state.get("activity"),
        context=state.get("context"),
        liked_artists=profile.get("liked_artists", []),
        disliked_tracks=profile.get("disliked_tracks", []),
        liked_tracks=profile.get("liked_tracks", []),
        clicked_tracks=profile.get("clicked_tracks", []),
        top_k=20,
    )
    return {"candidates": candidates}


# ---------------------------------------------------------------------------
# Node 5: generate_response
# ---------------------------------------------------------------------------
_rec_llm = _llm.with_structured_output(RecommendationOutput)


def generate_response(state: AgentState) -> dict:
    intent     = state.get("intent", "chat")
    candidates = state.get("candidates") or []
    profile    = state.get("user_profile") or {}

    # ── 공통 컨텍스트 구성 ────────────────────────────────────────────────
    ctx_parts = []
    if profile.get("liked_artists"):
        ctx_parts.append("좋아하는 아티스트: " + ", ".join(profile["liked_artists"][:5]))
    if state.get("mood"):
        ctx_parts.append(f"기분/분위기: {state['mood']}")
    if state.get("activity"):
        ctx_parts.append(f"활동: {state['activity']}")
    if state.get("context"):
        ctx_parts.append(f"맥락: {state['context']}")

    # 버튼 이벤트: user_input을 자연어 컨텍스트로 변환
    if state.get("button_event"):
        target_name = state.get("target_track_name") or "?"
        target_artist = state.get("target_track_artist") or "?"
        fb_type = state.get("feedback_type")
        if fb_type == "like":
            user_content = (
                f"[버튼 피드백] 사용자가 방금 '{target_name}' — {target_artist} 곡에 👍 좋아요를 눌렀습니다. "
                "이 곡이 마음에 들었을 만한 이유를 한 문장으로 짧게 짚어주고, "
                "비슷한 분위기의 곡을 새로 추천해 주세요."
            )
        elif fb_type == "dislike":
            user_content = (
                f"[버튼 피드백] 사용자가 방금 '{target_name}' — {target_artist} 곡에 👎 별로를 눌렀습니다. "
                "어떤 면이 마음에 들지 않았을지 한 문장으로 짧게 추측해 주고, "
                "다른 스타일의 곡을 새로 추천해 주세요."
            )
        else:
            user_content = state["user_input"]
    else:
        user_content = state["user_input"]
    if ctx_parts:
        user_content += "\n\n[컨텍스트]\n" + "\n".join(ctx_parts)

    # ── 추천 요청: 인덱스 기반 구조화 출력 ───────────────────────────────
    if candidates and intent in ("recommend", "feedback"):
        pool = candidates[:12]
        track_lines = "\n".join(
            f"{i+1}. {t['track_name']} — {t['artist_name']} "
            f"(에너지:{t['energy']:.2f}, 분위기:{t['valence']:.2f}, 댄서빌리티:{t['danceability']:.2f})"
            for i, t in enumerate(pool)
        )
        system = SystemMessage(content="""
당신은 친근하고 음악을 잘 아는 큐레이터 AI입니다.
항상 한국어로 자연스럽고 따뜻하게 대화하세요.

규칙:
- 아래 후보 목록에서 3~5곡을 골라 추천하세요.
- items의 index 필드에 후보 목록의 번호(1~12)를 그대로 입력하세요.
- 각 곡의 reason 필드에 추천 이유를 1~2문장으로 작성하세요.
- intro에 짧은 인삿말, outro에 피드백 유도 문장을 작성하세요.
""")
        user_content += f"\n\n[후보 목록]\n{track_lines}"

        messages = [system] + _recent_messages(state) + [HumanMessage(content=user_content)]
        result: RecommendationOutput = _rec_llm.invoke(messages)

        # 인덱스로 candidates 직접 조회 → 텍스트와 카드가 동일한 곡
        recs = []
        for item in result.items:
            idx = item.index - 1
            if 0 <= idx < len(pool):
                recs.append({**pool[idx], "reason": item.reason})

        # 텍스트 응답 조합
        lines = [result.intro, ""]
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. **{rec['track_name']}** — {rec['artist_name']}")
            lines.append(f"   {rec['reason']}")
            lines.append("")
        lines.append(result.outro)
        response_text = "\n".join(lines)

    # ── setup / chat: 일반 텍스트 응답 ───────────────────────────────────
    else:
        system = SystemMessage(content="""
당신은 친근하고 음악을 잘 아는 큐레이터 AI입니다.
항상 한국어로 자연스럽고 따뜻하게 대화하세요.
""")
        if intent == "setup":
            user_content += "\n\n취향을 등록했습니다. 친근하게 환영하고 음악 추천을 제안해 주세요."

        messages = [system] + _recent_messages(state) + [HumanMessage(content=user_content)]
        reply = _llm.invoke(messages)
        response_text = reply.content
        recs = []

    return {
        "response":        response_text,
        "recommendations": recs,
        "messages": [AIMessage(content=response_text)],
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------
def route_intent(state: AgentState) -> str:
    intent = state.get("intent", "chat")
    if intent in ("recommend", "feedback", "setup"):
        return intent
    return "chat"


def route_after_load(state: AgentState) -> str:
    intent = state.get("intent", "chat")
    if intent == "recommend":
        return "search"
    if intent in ("feedback", "setup"):
        return "update"
    return "respond"


def route_after_update(state: AgentState) -> str:
    """After updating profile: re-search for feedback, go straight to respond for setup."""
    if state.get("intent") == "feedback":
        return "search"
    return "respond"
