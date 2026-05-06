"""
SQLite-based user preference profile manager.
Stores liked/disliked artists, tracks, and mood preferences per session.
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class UserProfileManager:
    def __init__(self, db_path: str = "data/user_profiles.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    session_id    TEXT PRIMARY KEY,
                    liked_artists TEXT NOT NULL DEFAULT '[]',
                    liked_tracks  TEXT NOT NULL DEFAULT '[]',
                    disliked_tracks TEXT NOT NULL DEFAULT '[]',
                    liked_genres  TEXT NOT NULL DEFAULT '[]',
                    mood_history  TEXT NOT NULL DEFAULT '[]',
                    extra         TEXT NOT NULL DEFAULT '{}',
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL
                )
            """)
            conn.commit()

    # ------------------------------------------------------------------
    def get_profile(self, session_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM profiles WHERE session_id = ?", (session_id,)
            ).fetchone()

        if row is None:
            return self._empty_profile()

        return {
            "liked_artists":    json.loads(row["liked_artists"]),
            "liked_tracks":     json.loads(row["liked_tracks"]),
            "disliked_tracks":  json.loads(row["disliked_tracks"]),
            "liked_genres":     json.loads(row["liked_genres"]),
            "mood_history":     json.loads(row["mood_history"]),
            **json.loads(row["extra"]),
        }

    def save_profile(self, session_id: str, profile: Dict[str, Any]):
        now = datetime.utcnow().isoformat()
        extra_keys = set(profile.keys()) - {
            "liked_artists", "liked_tracks", "disliked_tracks",
            "liked_genres", "mood_history",
        }
        extra = {k: profile[k] for k in extra_keys}

        with self._connect() as conn:
            conn.execute("""
                INSERT INTO profiles
                    (session_id, liked_artists, liked_tracks, disliked_tracks,
                     liked_genres, mood_history, extra, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    liked_artists   = excluded.liked_artists,
                    liked_tracks    = excluded.liked_tracks,
                    disliked_tracks = excluded.disliked_tracks,
                    liked_genres    = excluded.liked_genres,
                    mood_history    = excluded.mood_history,
                    extra           = excluded.extra,
                    updated_at      = excluded.updated_at
            """, (
                session_id,
                json.dumps(profile.get("liked_artists", []),  ensure_ascii=False),
                json.dumps(profile.get("liked_tracks", []),   ensure_ascii=False),
                json.dumps(profile.get("disliked_tracks", []), ensure_ascii=False),
                json.dumps(profile.get("liked_genres", []),   ensure_ascii=False),
                json.dumps(profile.get("mood_history", []),   ensure_ascii=False),
                json.dumps(extra,                             ensure_ascii=False),
                now, now,
            ))
            conn.commit()

    # ------------------------------------------------------------------
    def add_liked_artist(self, session_id: str, artist: str):
        profile = self.get_profile(session_id)
        artists = profile.get("liked_artists", [])
        if artist not in artists:
            artists.append(artist)
            profile["liked_artists"] = artists
            self.save_profile(session_id, profile)

    def add_liked_track(self, session_id: str, track_id: str):
        profile = self.get_profile(session_id)
        tracks = profile.get("liked_tracks", [])
        if track_id not in tracks:
            tracks.append(track_id)
            profile["liked_tracks"] = tracks[-100:]   # keep last 100
            self.save_profile(session_id, profile)

    def add_disliked_track(self, session_id: str, track_id: str):
        profile = self.get_profile(session_id)
        tracks = profile.get("disliked_tracks", [])
        if track_id not in tracks:
            tracks.append(track_id)
            profile["disliked_tracks"] = tracks[-200:]
            self.save_profile(session_id, profile)

    def record_mood(self, session_id: str, mood: str):
        profile = self.get_profile(session_id)
        history: List[str] = profile.get("mood_history", [])
        history.append(mood)
        profile["mood_history"] = history[-50:]
        self.save_profile(session_id, profile)

    # ------------------------------------------------------------------
    @staticmethod
    def _empty_profile() -> Dict[str, Any]:
        return {
            "liked_artists":   [],
            "liked_tracks":    [],
            "disliked_tracks": [],
            "liked_genres":    [],
            "mood_history":    [],
        }
