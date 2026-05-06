"""
Music dataset loader and feature-based search.
Uses the local final_dataset.csv with Spotify audio features.
"""
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

CURRENT_YEAR = 2026

# --- Mood / activity → target audio feature vectors ---
FEATURE_COLS = [
    "danceability", "energy", "speechiness",
    "acousticness", "instrumentalness", "valence",
]

MOOD_PROFILES: Dict[str, Dict[str, float]] = {
    "happy":       {"valence": 0.8,  "energy": 0.65, "danceability": 0.60, "acousticness": 0.25},
    "sad":         {"valence": 0.2,  "energy": 0.30, "acousticness":  0.60, "instrumentalness": 0.30},
    "energetic":   {"energy":  0.9,  "danceability": 0.75, "valence":   0.65},
    "calm":        {"energy":  0.3,  "acousticness": 0.70, "valence":   0.55, "instrumentalness": 0.35},
    "focused":     {"instrumentalness": 0.60, "speechiness": 0.04, "energy": 0.50, "valence": 0.50},
    "party":       {"danceability": 0.85, "energy": 0.85, "valence": 0.75},
    "sleep":       {"energy":  0.15, "acousticness": 0.85, "instrumentalness": 0.70, "valence": 0.45},
    "angry":       {"energy":  0.90, "valence": 0.20},
    "romantic":    {"valence": 0.60, "acousticness": 0.65, "energy": 0.35},
    "melancholic": {"valence": 0.25, "energy": 0.35, "acousticness": 0.60},
    "nostalgic":   {"valence": 0.50, "acousticness": 0.55, "energy": 0.40},
    "relaxed":     {"energy":  0.30, "acousticness": 0.65, "valence": 0.60},
    "excited":     {"energy":  0.85, "valence": 0.80, "danceability": 0.75},
    "chill":       {"energy":  0.35, "valence": 0.55, "acousticness": 0.50, "danceability": 0.50},
}

ACTIVITY_PROFILES: Dict[str, Dict[str, float]] = {
    "working":    {"instrumentalness": 0.45, "speechiness": 0.04, "energy": 0.50},
    "studying":   {"instrumentalness": 0.60, "speechiness": 0.04, "energy": 0.40, "valence": 0.50},
    "exercising": {"energy": 0.90, "danceability": 0.70},
    "relaxing":   {"energy": 0.30, "acousticness": 0.65},
    "cooking":    {"danceability": 0.60, "energy": 0.60, "valence": 0.70},
    "commuting":  {"energy": 0.60, "danceability": 0.55},
    "sleeping":   {"energy": 0.15, "acousticness": 0.85, "instrumentalness": 0.70},
    "party":      {"danceability": 0.85, "energy": 0.85, "valence": 0.75},
    "driving":    {"energy": 0.70, "danceability": 0.60, "valence": 0.65},
    "cafe":       {"energy": 0.40, "acousticness": 0.55, "instrumentalness": 0.30},
}

# ── 파생변수 보정에 쓸 기분/활동 그룹 ────────────────────────────────────
# energy_danceability 부스트 대상: 활기차고 댄서블한 분위기
_DANCE_MOODS = {"party", "energetic", "excited", "dance", "happy"}
# acoustic_instrumental_diff 양수 부스트: 어쿠스틱+보컬
_ACOUSTIC_VOCAL_MOODS = {"romantic", "calm", "relaxed", "nostalgic",
                         "sad", "melancholic", "chill", "cafe"}
# acoustic_instrumental_diff 음수 부스트: 연주곡 중심
_INSTRUMENTAL_MOODS = {"focused", "working", "studying", "sleep", "sleeping"}
# mood_index 방향 부스트
_POSITIVE_MOODS = {"happy", "energetic", "excited", "party", "romantic"}
_NEGATIVE_MOODS = {"sad", "melancholic", "angry"}


def _find_csv() -> Path:
    """Locate final_dataset.csv — checks env var, then common relative paths."""
    env_path = os.getenv("MUSIC_DATA_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    candidates = [
        Path(__file__).parent / "data" / "final_dataset.csv",
        Path(__file__).parent / "final_dataset.csv",
        Path(__file__).parents[3] / "data" / "final_dataset.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "final_dataset.csv를 찾을 수 없습니다. "
        "MUSIC_DATA_PATH 환경 변수를 설정하거나 data/ 폴더에 파일을 복사해 주세요."
    )


class MusicDataset:
    _instance: Optional["MusicDataset"] = None
    _df: Optional[pd.DataFrame] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._df is None:
            self._load()

    def _load(self):
        path = _find_csv()
        df = pd.read_csv(path)
        df = df.dropna(subset=FEATURE_COLS)

        # Normalise loudness to [0, 1]  (typical range –60 … 0 dB)
        df["loudness_norm"] = (df["loudness"].clip(-60, 0) + 60) / 60

        # Normalise tempo to [0, 1]  (50 … 250 BPM)
        df["tempo_norm"] = (df["tempo"].clip(50, 250) - 50) / 200

        # ── CSV 파생변수 결측값만 보정 (재계산 없이 원본 사용) ───────────
        df["energy_danceability"]        = df["energy_danceability"].fillna(0)
        df["acoustic_instrumental_diff"] = df["acoustic_instrumental_diff"].fillna(0)
        df["mood_index"]                 = df["mood_index"].fillna(0.5)

        # ── 신규 파생변수 #1: track_age (발매 경과 연수) ─────────────────
        release_year = pd.to_datetime(
            df["release_date"], errors="coerce"
        ).dt.year.fillna(2000).astype(int)
        df["track_age"] = (CURRENT_YEAR - release_year).clip(lower=0)

        # ── 신규 파생변수 #2: popularity_ratio (트랙/아티스트 인기도 비율) ─
        df["popularity_ratio"] = (
            df["track_popularity"] /
            df["artist_popularity"].replace(0, np.nan)
        ).fillna(1.0).clip(0, 5)

        # Build feature matrix (only the 6 core features used for cosine search)
        self._feature_matrix = df[FEATURE_COLS].values.astype(float)
        MusicDataset._df = df

    # ------------------------------------------------------------------
    def _build_target_vector(
        self,
        mood: Optional[str],
        activity: Optional[str],
    ) -> Optional[np.ndarray]:
        target: Dict[str, float] = {col: 0.5 for col in FEATURE_COLS}

        matched = False
        if mood:
            key = mood.lower().strip()
            for k, v in MOOD_PROFILES.items():
                if k in key or key in k:
                    for feat, val in v.items():
                        if feat in target:
                            target[feat] = val
                    matched = True
                    break

        if activity:
            key = activity.lower().strip()
            for k, v in ACTIVITY_PROFILES.items():
                if k in key or key in k:
                    for feat, val in v.items():
                        if feat in target:
                            # Blend with mood if already set
                            target[feat] = (target[feat] + val) / 2 if matched else val
                    break

        return np.array([target[c] for c in FEATURE_COLS])

    # ------------------------------------------------------------------
    def _derived_boost(
        self,
        df: "pd.DataFrame",
        mood: Optional[str],
        activity: Optional[str],
    ) -> "pd.Series":
        """
        파생변수 3종을 기분/활동에 맞춰 점수에 가산.

        energy_danceability      → 댄서블·활기찬 분위기 부스트
        acoustic_instrumental_diff → 어쿠스틱+보컬 or 순수 연주 부스트
        mood_index               → 전반적 무드 방향성 부스트
        """
        boost = pd.Series(0.0, index=df.index)

        # 기분·활동 키워드를 정규화해서 그룹 집합과 비교
        keys: set = set()
        for raw in [mood, activity]:
            if not raw:
                continue
            kw = raw.lower().strip()
            for group in (_DANCE_MOODS | _ACOUSTIC_VOCAL_MOODS |
                          _INSTRUMENTAL_MOODS | _POSITIVE_MOODS | _NEGATIVE_MOODS):
                if group in kw or kw in group:
                    keys.add(group)
            keys.add(kw)

        # ① energy_danceability: 활기차고 댄서블한 분위기
        if keys & _DANCE_MOODS:
            boost += df["energy_danceability"] * 0.25

        # ② acoustic_instrumental_diff
        #    양수(어쿠스틱+보컬) → romantic/calm/sad 등
        #    음수(연주곡)       → focused/study/sleep 등
        if keys & _ACOUSTIC_VOCAL_MOODS:
            boost += df["acoustic_instrumental_diff"].clip(lower=0) * 0.20
        if keys & _INSTRUMENTAL_MOODS:
            boost += (-df["acoustic_instrumental_diff"]).clip(lower=0) * 0.20

        # ③ mood_index: 긍정 무드 → 높은 값, 부정 무드 → 낮은 값 선호
        if keys & _POSITIVE_MOODS:
            boost += df["mood_index"] * 0.15
        elif keys & _NEGATIVE_MOODS:
            boost += (1 - df["mood_index"]) * 0.15

        # ④ track_age: 기본적으로 신곡 소폭 우대 (최대 +0.05)
        max_age = df["track_age"].max() or 1
        boost += (1 - df["track_age"] / max_age) * 0.05

        # ⑤ popularity_ratio: 아티스트 대비 인기 높은 곡(숨겨진 명곡) 소폭 우대
        #    ratio > 1 이면 아티스트 평균보다 많이 들리는 곡 → +0.05 이하
        boost += (df["popularity_ratio"].clip(0, 2) / 2) * 0.05

        return boost

    # ------------------------------------------------------------------
    # 코사인·유클리디안 가중치
    _COSINE_W = 0.6
    _EUCLIDEAN_W = 0.4

    def _combined_similarity(
        self,
        target: np.ndarray,
        feature_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        코사인(분위기·방향) + 유클리디안(수치 근접도) 가중합으로 통합 유사도 산출.
        두 지표를 각각 [0,1]로 정규화한 뒤 합산.
        """
        cos_sim  = cosine_similarity([target], feature_matrix)[0]

        dist     = euclidean_distances([target], feature_matrix)[0]
        euc_sim  = 1.0 / (1.0 + dist)   # 거리 → 유사도 변환

        return self._COSINE_W * cos_sim + self._EUCLIDEAN_W * euc_sim

    # ------------------------------------------------------------------
    def search(
        self,
        mood: Optional[str] = None,
        activity: Optional[str] = None,
        context: Optional[str] = None,
        liked_artists: Optional[List[str]] = None,
        disliked_tracks: Optional[List[str]] = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        df = self._df.copy()

        # ① 별로인 곡 제외
        if disliked_tracks:
            df = df[~df["track_id"].isin(disliked_tracks)]

        # ② 통합 유사도 (코사인 0.6 + 유클리디안 0.4)
        target = self._build_target_vector(mood, activity)
        feature_matrix = df[FEATURE_COLS].values
        if target is not None:
            sim = self._combined_similarity(target, feature_matrix)
        else:
            sim = np.ones(len(df))

        df = df.copy()
        df["_score"] = sim

        # ③ 인기도 보너스
        df["_score"] += df["track_popularity"].fillna(0) / 200.0

        # ④ 파생변수 보정 5종
        df["_score"] += self._derived_boost(df, mood, activity)

        # ⑤ 선호 아티스트 가중치
        if liked_artists:
            liked_lower = [a.lower() for a in liked_artists]
            artist_boost = df["artist_name"].str.lower().apply(
                lambda n: 0.15 if any(l in n or n in l for l in liked_lower) else 0.0
            )
            df["_score"] += artist_boost

        df = df.sort_values("_score", ascending=False)
        top = df.head(top_k)

        return top[[
            "track_name", "track_id", "artist_name", "album_name",
            "track_popularity", "artist_popularity", "release_date",
            "danceability", "energy", "valence",
            "acousticness", "instrumentalness", "speechiness",
            "tempo", "mood_index", "track_age", "popularity_ratio",
            "_score",
        ]].to_dict(orient="records")

    # ------------------------------------------------------------------
    def get_artist_tracks(self, artist_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        df = self._df
        mask = df["artist_name"].str.lower().str.contains(artist_name.lower(), na=False)
        return df[mask].nlargest(top_k, "track_popularity")[
            ["track_name", "track_id", "artist_name", "album_name",
             "track_popularity", "danceability", "energy", "valence"]
        ].to_dict(orient="records")
