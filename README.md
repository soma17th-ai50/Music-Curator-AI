# 🎵 Music Curator AI

상황·기분에 맞는 음악을 추천하고 *왜 그 곡인지* 설명해 주는 LangGraph 기반 대화형 큐레이터.

## 주요 기능

- **자연어 추천**: "비 오는 날 카페에서 듣기 좋은 음악" 같은 문장으로 검색
- **취향 학습**: 👍/👎 버튼, 좋아하는 아티스트 등록, Spotify 클릭 시그널을 프로필에 반영
- **의도 라우팅**: 한 그래프에서 `recommend / feedback / setup / chat` 4가지 의도를 분기 처리
- **이유 설명**: 단순 리스트가 아니라 각 곡을 추천한 근거를 함께 제시

## 기술 스택

- **에이전트**: LangGraph + LangChain
- **LLM**: Upstage Solar (`solar-pro2`)
- **데이터**: 로컬 CSV (`src/music/final_dataset.csv`)
- **프로필 저장**: SQLite (`data/user_profiles.db`)
- **UI**: Streamlit

## 디렉토리 구조

```
.
├── main.py                       # Streamlit 진입점
├── requirements.txt
├── data/
│   └── user_profiles.db          # 사용자 프로필 (자동 생성)
└── src/
    ├── agent/
    │   ├── graph.py              # LangGraph 워크플로우
    │   ├── nodes.py              # 노드 구현 (analyze/load/search/...)
    │   └── state.py              # AgentState 정의
    ├── memory/
    │   └── profile.py            # UserProfileManager
    └── music/
        ├── dataset.py            # MusicDataset 로더
        └── final_dataset.csv     # 곡 메타데이터
```

## 그래프 토폴로지

```
START
  └─► analyze_input
        ├── (recommend) ──► load_profile ──► search_music ──► generate_response ──► END
        ├── (feedback)  ──► load_profile ──► update_profile ──► search_music ──► generate_response ──► END
        ├── (setup)     ──► load_profile ──► update_profile ──► generate_response ──► END
        └── (chat)      ──► generate_response ──► END
```

## 실행 방법

### 1. 의존성 설치

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 환경변수 설정

`.env.example`을 복사해서 `.env`를 만들고 키를 채워 넣습니다.

```
UPSTAGE_API_KEY=up_...
LANGSMITH_API_KEY=ls__...      # (선택) 트레이싱용
LANGSMITH_TRACING_V2=true
LANGSMITH_PROJECT=music-recommendation-agent
```

### 3. 실행

```powershell
streamlit run main.py
```

브라우저가 자동으로 열리며, 첫 진입 시 좋아하는 아티스트 3명을 등록하면 추천 품질이 향상됩니다.
