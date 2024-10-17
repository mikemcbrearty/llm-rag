import numpy as np
from numpy.linalg import norm
import torch
from transformers import AutoModel, AutoTokenizer


class EmbeddingsSearch:
    """
    Based on https://www.pinecone.io/learn/series/rag/embedding-models-rundown/
    """

    device = "cpu"
    model_id = "intfloat/e5-base-v2"

    def __init__(self):
        # initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

        self.sections = self.__readDocSections()
        self.sections_arr = np.array(self.sections)
        self.index = self.__buildIndex(self.sections)


    def __readDocSections(self) -> list[str]:
        with open("documents/318_08_sections.txt", 'r') as f:
            sections = [line for line in f]
        return sections


    def __buildIndex(self, sections: list[str]):
        batch_size = 256
        for i in range(0, len(sections), batch_size):
            i_end = min(len(sections), i+batch_size)
            sections_batch = sections[i:i_end]
            embed_batch = self.__embed(sections_batch)
            if i == 0:
                arr = embed_batch.copy()
            else:
                arr = np.concatenate([arr, embed_batch.copy()])
        return arr


    def __embed(self, docs: list[str]):
        docs = [f"passage: {d}" for d in docs]
        # tokenize
        tokens = self.tokenizer(
            docs,
            padding=True,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            # process with model for token-level embeddings
            out = self.model(**tokens)
            # mask padding tokens
            last_hidden = out.last_hidden_state.masked_fill(
                ~tokens["attention_mask"][..., None].bool(), 0.0
            )
            # create mean pooled embeddings
            doc_embeds = last_hidden.sum(dim=1) / tokens["attention_mask"].sum(dim=1)[..., None]
        return doc_embeds.numpy()


    def query(self, text: str, top_k: int=3) -> list[str]:
        arr = self.index

        # create query embedding
        xq = self.__embed([f"query: {text}"])[0]

        # calculate cosine similarities
        sim = np.dot(arr, xq.T) / (norm(arr, axis=1)*norm(xq.T))

        # get indices of top_k records
        idx = np.argpartition(sim, -top_k)[-top_k:]
        docs = self.sections_arr[idx]
        return docs.tolist()
