from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import json
import logging
import os
from getpass import getpass

OPENAI_API_KEY = getpass()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# Set up logging
logging.basicConfig(level=logging.INFO, filename="llm_generator.log", format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LLMGenerator:
    def __init__(self):
        llm = OpenAI()
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Define prompt template for generating query-response pairs
        prompt = PromptTemplate.from_template(
            """You are an expert in warehouse management software. Given the following section of a dashboard documentation, generate exactly 3 relevant questions that a user might ask about the content, along with accurate responses based solely on the provided text. Ensure responses are concise, accurate, and actionable. Output the result as a valid JSON list of query-response pairs.

            Documentation: {text}
            Section: {section}

            Output format (valid JSON):
            [
              {{"query": "Question?", "response": "Response."}},
              {{"query": "Question?", "response": "Response."}},
              {{"query": "Question?", "response": "Response."}}
            ]"""
        )
        self.chain = prompt | llm
    

    def generate_pairs(self, chunk: Dict[str, str]) -> List[Dict[str, str]]:
        """Generate query-response pairs from a document chunk."""
        text = chunk["text"]
        section = chunk["metadata"].get("section", "Unknown")

        # Generate pairs using LLM
        try:
            print("Generating pairs...")
            generated = self.chain.invoke({"text": text, "section": section})
            logger.info(f"Raw LLM output: {generated}")  # Log the raw output from the LLM

            # Parse generated text as JSON
            pairs = json.loads(generated.strip())
            if not isinstance(pairs, list):
                raise ValueError("Generated output is not a list")
            logger.info(f"Generated {len(pairs)} pairs for chunk: {text[:50]}...")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e} - Generated text: {generated}")
            return []
        except Exception as e:
            logger.error(f"Error generating pairs: {e}")
            return []

        return pairs

    

if __name__ == "__main__":
    from document_processor import DocumentProcessor
    processor = DocumentProcessor()
    generator = LLMGenerator()
    chunks = processor.extract_text_from_txt("documents/TELIA for Coil - Dashboard User Guide.txt")
    for chunk in chunks[:2]:  
        pairs = generator.generate_pairs(chunk)
        print(pairs.__len__())
        print(f"Chunk: {chunk['text'][:50]}...")
        print("Generated Pairs:")
        for pair in pairs:
            print(f"Query: {pair['query']}")
            print(f"Response: {pair['response']}\n")