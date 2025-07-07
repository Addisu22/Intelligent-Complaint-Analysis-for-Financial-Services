
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
- Cleaned and filtered dataset saved to: `data/filtered_complaints.csv`
- Jupyter Notebook or Python script: `notebooks/task1_eda_preprocessing.ipynb`

Key Findings
- The majority of complaints centered around credit cards and money transfers.
- Many narratives were short or missing, so filtering significantly improved dataset quality.
- Cleaning helped standardize content for better downstream text embedding.

Tools
- `pandas`, `matplotlib`, `seaborn`, `re`
