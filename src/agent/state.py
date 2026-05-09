import operator
from typing import Annotated, Optional, List, Dict, Any
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # Conversation history (auto-merge via operator.add)
    messages: Annotated[list, operator.add]
    user_input: str
    session_id: str

    # Intent classification results
    intent: Optional[str]             # recommend | feedback | setup | chat
    mood: Optional[str]               # happy, sad, energetic, calm, focused, ...
    activity: Optional[str]           # working, exercising, relaxing, ...
    context: Optional[str]            # extra context (weather, place, time)
    feedback_type: Optional[str]      # like | dislike | neutral
    liked_artists_input: Optional[List[str]]  # artists mentioned by user

    # Persistent data
    user_profile: Optional[Dict[str, Any]]

    # Recommendation pipeline
    candidates: Optional[List[Dict[str, Any]]]
    recommendations: Optional[List[Dict[str, Any]]]

    # Button-driven feedback (👍/👎 click) — bypasses intent classification
    button_event: Optional[bool]
    target_track_id: Optional[str]
    target_track_name: Optional[str]
    target_track_artist: Optional[str]

    # Final output
    response: Optional[str]
