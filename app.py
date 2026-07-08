import streamlit as st
from src.rag_query import answer

st.title("10-K / 10-Q Analyst")

with st.sidebar:
    graph_mode = st.selectbox("Knowledge graph", ["Auto", "On", "Off"])
use_graph = {"Auto": None, "On": True, "Off": False}[graph_mode]

question = st.text_input("Question", placeholder="What are Apple's main risk factors?")

if st.button("Ask") and question:
    with st.spinner("Searching filings..."):
        result = answer(question, use_graph=use_graph)

    st.markdown("### Answer")
    st.write(result["answer"])

    if result.get("graph_facts"):
        with st.expander("Knowledge Graph Facts"):
            for intent, rows in result["graph_facts"].items():
                st.markdown(f"**{intent.capitalize()}**")
                for r in rows:
                    if intent in ("board", "executives"):
                        title = f" — {r['title']}" if r.get("title") else ""
                        st.write(f"- {r['name']}{title} ({r['org']})")
                    elif intent == "headquarters":
                        st.write(f"- {r['org']}: {r['address']}")
                    else:
                        st.write(f"- {r['name']} ({r['org']})")

    with st.expander("Sources"):
        for s in result["sources"]:
            st.write(f"**{s['filing_type']}** · chunk {s['chunk_index']} · score {s['score']}")
            st.caption(s["source"])
