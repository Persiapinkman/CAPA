import re
from typing import Any, List
from llama_index.core.node_parser import TextSplitter


class RecursiveCharacterTextSplitter2(TextSplitter):
    """Implementation of splitting text that looks at characters.

    Recursively tries to split by different characters to find one
    that works.
    """
    _separator = "\n\n"
    _pattern1 = r'[一二三四五六七八九十百千万零〇]+、'  # Pattern for splitting Chinese numbered lists (e.g., 一、二、)
    _pattern2 = [rf'\n第[一二三四五六七八九十百千万零〇]+{suffix}' for suffix in '章节条款项']  # Patterns for chapters and articles
    _pattern3 = r'\n（[一二三四五六七八九十百千万零〇]+）'  # Pattern for splitting based on Chinese numbered parentheses (e.g., （一）)
    chunk_size = 128

    def __init__(self, chunk_size=128, **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self.chunk_size = chunk_size

    def split_text(self, text: str, force_chunk: int = 300) -> List[str]:
        # Initial split by separator
        texts = text.split(self._separator)

        # Apply each split pattern in sequence
        chunks = self.split1(self._pattern1, texts)
        chunks = self.split2(self._pattern2, chunks)
        chunks = self.split3(self._pattern3, chunks)

        # If chunks exceed max size, split further
        chunks = self._split_large_chunks(chunks, force_chunk)

        # Merge chunks with fewer than chunk_size characters with the previous chunk
        return self._merge_small_chunks(chunks)

    def split1(self, pattern: str, texts: List[str]) -> List[str]:
        # Split based on numbered lists like "一、二、三..."
        result = []
        for text in texts:
            split_text = re.split(pattern, text)
            result.extend(split_text)
        return result

    def split2(self, patterns: List[str], chunks: List[str]) -> List[str]:
        # Split based on chapters or articles like "第x章" or "第x条"
        result = []
        for chunk in chunks:
            for pattern in patterns:
                chunk = re.sub(pattern, f"\n{pattern}", chunk)  # Add separator before pattern
            result.append(chunk)
        return result

    def split3(self, pattern: str, chunks: List[str]) -> List[str]:
        # Split based on numbered parentheses (e.g., "（一）")
        result = []
        for chunk in chunks:
            split_text = re.split(pattern, chunk)
            result.extend(split_text)
        return result

    def _split_large_chunks(self, chunks: List[str], force_chunk: int) -> List[str]:
        # If chunks are too large, split them further into smaller chunks
        result = []
        for chunk in chunks:
            if len(chunk) > force_chunk:
                result.extend(self._split_text_by_size(chunk, force_chunk))
            else:
                result.append(chunk)
        return result

    @staticmethod
    def _split_text_by_size(text: str, max_size: int) -> List[str]:
        # Split text into smaller chunks by size
        chunks = []
        while len(text) > max_size:
            split_point = text.rfind(' ', 0, max_size)  # Find the last space within the max_size limit
            if split_point == -1:  # No space found, just cut at max_size
                split_point = max_size
            chunks.append(text[:split_point])
            text = text[split_point:].strip()  # Remaining text
        if text:
            chunks.append(text)  # Add the final part
        return chunks

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge chunks with fewer than chunk_size characters with the previous chunk."""
        result = []
        for i in range(len(chunks)):
            if len(chunks[i]) < self.chunk_size and result:
                result[-1] += chunks[i]  # Merge with previous chunk
            else:
                result.append(chunks[i])
        return result


if __name__ == '__main__':
    # Example usage
    splitter = RecursiveCharacterTextSplitter2(chunk_size=128)
    text = "这是第一部分内容。\n\n一、内容1\n第二部分内容。\n\n第六章 内容2\n（五）章节内容。"
    chunks = splitter.split_text(text)
    for chunk in chunks:
        print(chunk)