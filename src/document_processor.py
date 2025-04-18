import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict

class DocumentProcessor:
    def __init__(self):
        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Max tokens per chunk
            chunk_overlap=100,  # Overlap for context
            separators=["\n\n", "\n", ". ", " ", ""],  # Split by paragraphs, sentences, etc.
            length_function=len
        )

    def extract_text_from_txt(self, txt_path: str) -> List[Dict[str, str]]:
        """Extract text from a TXT file and split into chunks with metadata."""
        if not os.path.isfile(txt_path):
            raise FileNotFoundError(f"The file {txt_path} does not exist.")

        chunks = []
        current_section = "Unknown"

        # Read the entire TXT file
        with open(txt_path, "r", encoding="utf-8") as file:
            full_text = file.read()

        # Split text into chunks using LangChain
        documents = self.text_splitter.create_documents([full_text])
        for doc in documents:
            chunk_text = doc.page_content
            # Heuristic: Detect section headings (e.g., all caps or starting with '#')
            lines = chunk_text.split("\n")
            for line in lines:
                if line.strip().startswith("#") or line.isupper():
                    current_section = line.strip()
                    break  # Use first heading-like line as section
            chunks.append({
                "text": chunk_text.strip(),
                "metadata": {"section": current_section}
            })

        return chunks

if __name__ == "__main__":
    processor = DocumentProcessor()
    # Replace with path to your sample TXT file
    chunks = processor.extract_text_from_txt("documents/TELIA for Coil - Dashboard User Guide.txt")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"Text: {chunk['text'][:100]}...")
        print(f"Metadata: {chunk['metadata']}\n")