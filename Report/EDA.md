Exploratory Data Analysis Summary
=================================

We analyzed the CFPB complaint dataset to understand the distribution and quality of consumer complaints. The initial dataset contained over X entries with varying levels of completeness. We focused on narrative complaints and found that many entries were missing text or contained very short inputs. Most complaints were centered around products like credit cards, personal loans, and money transfers.

After filtering for five key products of interest and removing entries without narratives, we performed basic text cleaning to prepare the data for downstream language processing and embedding. Narrative length ranged widely, with a small portion being extremely short or overly long. The cleaned and filtered dataset is now ready for use in the Retrieval-Augmented Generation (RAG) pipeline.
