# llm-rag

Large language models (LLMs) have achieved strong results at answering questions assessing general knowlegdge[^1] as well as knowledge of many specific subject areas, such as math, and hard sciences[^2]. However, there remain many subject areas where an LLM's knowledge is incomplete. With this project, I try to improve one LLM's ability to correctly answer questions in a specific subject area, by using retrieval augmented generation (RAG) techniques to include relevant context when posing questions to the LLM.

For the LLM, I use Llama 3.1 8B. This model was released in July 2024, it is small enough to easily run on a laptop, and it performs well on many benchmarks[^3].

For the subject area, I focus on a topic from structural engineering, structural concrete. I wrote a set of 20+ questions and answers on this topic. All of these questions have short answers, do not require any calculations, and can be answered by referencing a relevant code document. For such a code, I used ACI 318-08, the American Concrete Institute Building Code Requirements for Structural Concrete document from 2008, when writing the questions.

The questions are open ended. When I initially wrote the questions, I liked the idea that this would elicit a free form response from the LLM. After having gone through the exercise, I found that evaluating free form responses accurately and automatically can be tricky. I used heuristics to evaluate the responses, which proved to be a little noisy. As an example, consider this question:

> When designing a structural concrete member subject to flexure or axial load, should the tensile strength of the concrete be considered?

The simple yes/no answer is "no, do not consider tensile strength of concrete." However, the more nuanced answer is "generally (and conservatively), no, do not consider tensile strength of concrete. But, there can be exceptions to this when designing particular types of concrete structures such as prestressed concrete members, plain concrete, etc. and when specific conditions are met." The heuristics that I used did not fully capture the range of possible correct answers. In comparison, many benchmarks use multiple choices questions, which have the nice property of being straight forward to score.

For the RAG techniques, when posing questions to the LLM, I retrieved relevant sections of ACI 318-08 and included them in the prompt with the question. The document itself is a large pdf with code and commentary sections. I split it by section to divide it into smaller chunks. And, I did this manually because for one document, that was quick and accurate. For retrieval, I tried two different approaches. One, I did full text search with Solr. Two, I did semantic search using an embeddings model.

## Results

### Retrieval

As an intermediate result, I checked how well the retrieval approaches worked with the question and source document data sets. When writing each question, I noted the relevant code section. I assessed the performance of the retrieval, varying the number of items retrieved, with precision, recall, and F<sub>1</sub> score. Precision was the ratio of relevant items retrieved to total items retrieved. Recall was the ratio of relevant items retrieved to total relevant items. And, F<sub>1</sub> score was the harmonic mean of precision and recall.

|retrieval method|precision|recall|F<sub>1</sub> score|
|---|---|---|---|
|full text (top_k=1)|55%|55%|55%|
|full text (top_k=3)|23%|68%|34%|
|full text (top_k=5)|15%|77%|26%|
|full text (top_k=7)|11%|68%|19%|
|embeddings (top_k=1)|55%|55%|55%|
|embeddings (top_k=3)|23%|68%|34%|
|embeddings (top_k=5)|16%|82%|27%|
|embeddings (top_k=7)|12%|82%|20%|
|full text (top_k=1) & embeddings (top_k=1)|39%|77%|52%|
|full text (top_k=3) & embeddings (top_k=3)|14%|86%|25%|
|full text (top_k=5) & embeddings (top_k=5)|10%|95%|17%|
|full text (top_k=7) & embeddings (top_k=7)|7%|95%|13%|

Both full text and embeddings search are capable of retrieving relevant code sections given a question as a query. With this dataset, there is only one relevant section per question. As the number of items retrieved increases, recall increases, precision decreases, and the F<sub>1</sub> score decreases.

I also tried combining the results of both a full text and embedding search. This approach yielded the highest recall, as well as lowest precision.


### Answering Questions

For answering questions, I compared the success rate of the LLM at answering the question set with and without RAG.

|RAG approach|success rate|
|---|---|
|no RAG|45%|
|RAG with full text retrieval (top_k=1)|64%|
|RAG with full text retrieval (top_k=3)|45%|
|RAG with full text retrieval (top_k=5)|45%|
|RAG with full text retrieval (top_k=7)|45%|
|RAG with embeddings retrieval (top_k=1)|55%|
|RAG with embeddings retrieval (top_k=3)|68%|
|RAG with embeddings retrieval (top_k=5)|64%|
|RAG with embeddings retrieval (top_k=7)|64%|
|RAG with full text (top_k=1) & embeddings retrieval (top_k=1)|64%|
|RAG with full text (top_k=3) & embeddings retrieval (top_k=3)|68%|
|RAG with full text (top_k=5) & embeddings retrieval (top_k=5)|45%|
|RAG with full text (top_k=7) & embeddings retrieval (top_k=7)|50%|

All of the RAG approaches are capable of enabling the LLM to answer the question set with a higher success rate than the LLM by itself. Also, extent to which RAG helps is sensitive to how the RAG approach is configured. In this exercise, configurations where the retrieval F<sub>1</sub> score was higher generally performed better than when F<sub>1</sub> score was lower. Also, embedding retrieval performed better than full text retrieval.


### Set up

#### python virtual environment
```
python3 -m venv virtual_environment_path
source virtual_environment_path/bin/activate
pip install -r requirements.txt
```

#### solr
```
brew install solr
# To start solr now and restart at login:
  brew services start solr
# Or, if you don't want/need a background service you can just run:
  /opt/homebrew/opt/solr/bin/solr start -f -s /opt/homebrew/var/lib/solr
# To run in cloud mode
  /opt/homebrew/opt/solr/bin/solr start -f -s /opt/homebrew/var/lib/solr -c
# To restart in cloud mode
  /opt/homebrew/opt/solr/bin/solr start -f -s /opt/homebrew/var/lib/solr -c
# To check whether solr is running
  /opt/homebrew/opt/solr/bin/solr status
```

#### ollama
```
ollama run llama3.1
```

#### ACI 318-08
ACI 318-08 is the short name for the American Concrete Institute Building Code Requirements for Structural Concrete document from 2008. This document is updated every few years, so the 2008 version is now superceded. I am not including a copy of it in the repository, but newer versions of it are available from ACI directly.

The electronic version of the document is a pdf file. For this project, I manually extracted the text of the code portion of the document and formatted it with one line for each chapter section of the document.
```
1.1 — Scope 1.1.1 — This Code provides minimum requirements...
1.2 — Drawings and specifications 1.2.1 — Copies of design ...
1.3 — Inspection 1.3.1 — Concrete construction shall be ins...
...
```
(I mention the code portion specifically, because the document also contains commentary text, which I excluded for this project.)

Also, I evaluated parsing the document automatically, but the initial approaches I tried did not give reliable results. So, manual processing seemed more expedient.


[^1]: MMLU [ref](https://arxiv.org/abs/2009.03300)
[^2]: GPQA [ref](https://arxiv.org/abs/2311.12022)
[^3]: https://ai.meta.com/blog/meta-llama-3-1/
