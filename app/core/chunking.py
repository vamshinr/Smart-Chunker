from app.core.embeddings import embed_sentences, cosine_sim
from app.core.utils import extract_text_and_elements_from_pdf
import nltk
import re
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize

class SmartChunker:
    def __init__(self, method="semantic", chunk_size=512, overlap=50):
        self.method = method
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, visual_elements: list = None):
        if self.method == "fixed":
            return self.fixed_chunking(text, visual_elements)
        elif self.method == "header":
            return self.header_chunking(text, visual_elements)
        elif self.method == "swarax":
            return self.semantic_window_chunking(text, visual_elements)
        else:
            return self.semantic_chunking(text, visual_elements)

    def fixed_chunking(self, text, visual_elements):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_metadata = self._match_visual_elements(start, end, visual_elements)
            chunks.append({"text": text[start:end], "start": start, "end": end, "metadata": chunk_metadata})
            start = end - self.overlap
        return chunks

    def header_chunking(self, text, visual_elements):
        pattern = re.compile(r'(Section \d+\:|Chapter \d+\:|Part \d+\:)', re.IGNORECASE)
        matches = list(pattern.finditer(text))

        chunks = []
        for i in range(len(matches)):
            start_idx = matches[i].start()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk_text = text[start_idx:end_idx].strip()
            chunk_metadata = self._match_visual_elements(start_idx, end_idx, visual_elements)
            chunks.append({"text": chunk_text, "start": start_idx, "end": end_idx, "metadata": chunk_metadata})
        return chunks

    def semantic_chunking(self, text, visual_elements):
        sentences = sent_tokenize(text)
        embeddings = embed_sentences(sentences)

        chunks, current_chunk, current_embeds = [], [], []
        start_char = 0
        for i, sent in enumerate(sentences):
            current_chunk.append(sent)
            current_embeds.append(embeddings[i])
            if len(" ".join(current_chunk)) > self.chunk_size:
                chunk_text = " ".join(current_chunk)
                end_char = start_char + len(chunk_text)
                chunk_metadata = self._match_visual_elements(start_char, end_char, visual_elements)
                chunks.append({"text": chunk_text, "start": start_char, "end": end_char, "metadata": chunk_metadata})
                start_char = end_char - self.overlap
                current_chunk, current_embeds = [], []
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({"text": chunk_text, "start": start_char, "end": start_char + len(chunk_text), "metadata": self._match_visual_elements(start_char, start_char + len(chunk_text), visual_elements)})
        return chunks

    def semantic_window_chunking(self, text, visual_elements):
        sentences = sent_tokenize(text)
        embeddings = embed_sentences(sentences)

        chunks = []
        start_char = 0
        max_tokens = self.chunk_size
        threshold = 0.75

        i = 0
        while i < len(sentences):
            current_chunk = [sentences[i]]
            current_length = len(sentences[i])
            j = i + 1
            while j < len(sentences):
                sim = cosine_sim(embeddings[j-1], embeddings[j])
                if sim < threshold or (current_length + len(sentences[j])) > max_tokens:
                    break
                current_chunk.append(sentences[j])
                current_length += len(sentences[j])
                j += 1
            chunk_text = " ".join(current_chunk)
            end_char = start_char + len(chunk_text)
            chunk_metadata = self._match_visual_elements(start_char, end_char, visual_elements)
            chunks.append({"text": chunk_text, "start": start_char, "end": end_char, "metadata": chunk_metadata})
            start_char = end_char - self.overlap
            i = j

        return chunks

    def _match_visual_elements(self, start: int, end: int, visual_elements: list):
        if not visual_elements:
            return {}
        metadata = {
            "contains_image": False,
            "contains_table": False,
            "contains_code": False
        }
        for el in visual_elements:
            if el["start"] >= start and el["start"] < end:
                metadata[f"contains_{el['type']}"] = True
        return metadata
