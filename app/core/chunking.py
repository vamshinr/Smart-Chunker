from app.core.embeddings import embed_sentences, cosine_sim
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

class SmartChunker:
    def __init__(self, method="semantic", chunk_size=512, overlap=50):
        self.method = method
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str):
        if self.method == "fixed":
            return self.fixed_chunking(text)
        elif self.method == "header":
            return self.header_chunking(text)
        else:
            return self.semantic_chunking(text)

    def fixed_chunking(self, text):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append({"text": text[start:end], "start": start, "end": end})
            start = end - self.overlap
        return chunks

    def header_chunking(self, text):
        lines = text.split("\n")
        chunks = []
        chunk, start = "", 0
        for line in lines:
            if line.strip().lower().startswith(("section", "chapter", "##")):
                if chunk:
                    chunks.append({"text": chunk.strip(), "start": start, "end": start + len(chunk)})
                    start += len(chunk)
                    chunk = ""
            chunk += line + "\n"
        if chunk:
            chunks.append({"text": chunk.strip(), "start": start, "end": start + len(chunk)})
        return chunks

    def semantic_chunking(self, text):
        sentences = sent_tokenize(text)
        embeddings = embed_sentences(sentences)

        chunks, current_chunk, current_embeds = [], [], []
        start_idx, start_char = 0, 0
        for i, sent in enumerate(sentences):
            current_chunk.append(sent)
            current_embeds.append(embeddings[i])
            if len(" ".join(current_chunk)) > self.chunk_size:
                mean_vec = sum(current_embeds) / len(current_embeds)
                chunk_text = " ".join(current_chunk)
                end_char = start_char + len(chunk_text)
                chunks.append({"text": chunk_text, "start": start_char, "end": end_char})
                start_char = end_char - self.overlap
                current_chunk, current_embeds = [], []
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({"text": chunk_text, "start": start_char, "end": start_char + len(chunk_text)})
        return chunks
