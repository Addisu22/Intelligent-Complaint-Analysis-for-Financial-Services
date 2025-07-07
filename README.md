
Task 1: Exploratory Data Analysis and Data Preprocessing
========================================================
Objective
Prepare the CFPB complaint dataset for semantic search by:
- Performing Exploratory Data Analysis (EDA)
- Filtering relevant complaints
- Cleaning and preprocessing narrative texts

 Steps Performed
1. **Loaded CFPB complaints dataset** using `pandas`.
2. **Performed EDA**:
   - Count of complaints per product
   - Length distribution of complaint narratives
   - Presence of narratives vs. empty entries
3. **Filtered** for 5 relevant product categories:
   - Credit card
   - Personal loan
   - Buy Now, Pay Later (BNPL)
   - Savings account
   - Money transfers
4. **Cleaned narratives** by:
   - Lowercasing text
   - Removing special characters and boilerplate
   - Normalizing whitespace

Output
- Cleaned and filtered dataset saved to: `Data/filtered_complaints.csv`
- Jupyter Notebook or Python script: `notebooks/EDA_Preprocessing.ipynb`

Key Findings
- The majority of complaints centered around credit cards and money transfers.
- Many narratives were short or missing, so filtering significantly improved dataset quality.
- Cleaning helped standardize content for better downstream text embedding.

Tools
- `pandas`, `matplotlib`, `seaborn`, `re`

Task 2: Text Chunking, Embedding, and Vector Store Indexing
===========================================================

Objective
Convert cleaned complaint narratives into vector embeddings for efficient semantic search using FAISS.

Steps Performed
1. **Loaded cleaned data** from `Data/filtered_complaints.csv`.
2. **Chunked long texts** using `LangChain's RecursiveCharacterTextSplitter`:
   - `chunk_size = 1000000`
   - `chunk_overlap = 100000`
3. **Generated embeddings** using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Indexed vectors** using `FAISS` and saved metadata (complaint ID, product).

Output
- FAISS vector index: `Vector_Store/FAISS_Index/faiss.index`
- Metadata: `Vector_Store/FAISS_Index/metadata.pkl`
- Script: `src/Embed_Index.py`

Model Choice Justification
- **MiniLM-L6-v2** was chosen for:
  - High performance on semantic similarity tasks
  - Lightweight and fast inference
  - Multilingual capability

Tools
- `LangChain`, `FAISS`, `sentence-transformers`, `pickle`, `pandas`

Task 3: Building the RAG Core Logic and Evaluation
==================================================

Objective
Build the logic that powers the RAG pipeline and evaluate its performance on real-world queries.

Steps Performed
1. **Retriever**:
   - Embedded user queries using MiniLM
   - Searched top-k results in FAISS vector store
2. **Prompt Engineering**:
   - Designed prompt to instruct the LLM using retrieved chunks
3. **Generator**:
   - Used HuggingFace `text-generation` pipeline with GPT-2
   - Combined prompt and context to produce answers
4. **Evaluation**:
   - Ran 5–10 questions through the pipeline
   - Recorded outputs in markdown evaluation table

Output
- Python module: `src/RAG_Pipeline.py`
- Evaluation report: `Report/Evaluation.md`

Evaluation Structure
| Question | Retrieved Sources | Answer | Score | Comments |
|----------|-------------------|--------|-------|----------|

Tools
- `sentence-transformers`, `FAISS`, `transformers`, `pickle`, `numpy`

Task 4: Creating an Interactive Chat Interface
==============================================

 Objective
Build an easy-to-use web interface for non-technical users to interact with the RAG system.

Features
- Input box to ask a question
- Submit and clear buttons
- Displays AI-generated answer
- Displays source complaint excerpts (for trust & verification)

How It Works
- User types a question → `rag_pipeline()` is triggered
- Retrieves top-5 chunks from FAISS vector store
- Constructs prompt and generates response
- Shows final answer + sources to user

Files
- Interface: `app.py`
- Screenshot: `Report/Screenshot.png` or `.gif`

To Run
```bash
streamlit run app.py
