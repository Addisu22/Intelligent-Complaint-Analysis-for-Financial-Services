import streamlit as st
from RAG_Pipeline import rag_pipeline

# Page config
st.set_page_config(page_title="CrediTrust Complaint Assistant", layout="wide")
st.title(" CrediTrust Complaint Q&A System")
st.write("Ask a question about consumer complaints. The assistant will retrieve relevant cases and generate an evidence-based answer.")

# User input
question = st.text_input(" Enter your question", placeholder="e.g., What are common complaints about BNPL services?")

# Submit button
if st.button("Ask") and question.strip():
    with st.spinner(" Thinking..."):
        result = rag_pipeline(question)
        st.markdown(" Answer")
        st.success(result["answer"])

        st.markdown(" Retrieved Sources")
        for i, source in enumerate(result["retrieved_sources"], 1):
            st.markdown(f"**Source {i}** *(Product: {source['product']})*")
            st.write(source["original_text"][:500] + "...")


# Clear button (Streamlit workaround)
if st.button("Clear"):
    st.experimental_rerun()
