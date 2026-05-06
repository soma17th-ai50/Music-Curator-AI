"""
LangGraph 워크플로우를 그림으로 저장하고 출력합니다.

실행:  python visualize_graph.py
결과:  graph.png 저장 + Mermaid 소스 출력
"""
from dotenv import load_dotenv
load_dotenv()

from src.agent.graph import create_graph

graph = create_graph()
compiled = graph.get_graph()

# ── PNG 저장 ───────────────────────────────────────────────────────────────
png_bytes = compiled.draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)
print("[OK] Graph saved -> graph.png")

# ── Mermaid 소스 출력 ──────────────────────────────────────────────────────
print("\n--- Mermaid Source ---")
mermaid_src = compiled.draw_mermaid()
# safe print for Windows cp949 console
print(mermaid_src.encode("utf-8").decode("ascii", errors="replace"))
