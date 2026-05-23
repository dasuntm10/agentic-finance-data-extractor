"""Generate langgraph_flow.png using LangGraph's built-in draw_mermaid_png().

Uses the Mermaid.ink API (one HTTP call; only needed at diagram-generation time,
not at pipeline runtime). Requires internet access when run.

Output: langgraph_flow.png in the project root.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore")

from afde.orchestrator import build_graph  # noqa: E402

graph = build_graph()
png = graph.get_graph().draw_mermaid_png()

out = Path(__file__).parent.parent / "langgraph_flow.png"
out.write_bytes(png)
print(f"Saved {out}  ({len(png):,} bytes)")
