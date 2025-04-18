import os
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, filename="supabase_client.log", format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SupabaseClient:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
        
        # Initialize Supabase client
        self.client: Client = create_client(url, key)
        self.table_name = "knowledge_base"

    def insert_pairs(self, pairs: List[Dict[str, str]]) -> None:
        """Insert query-response pairs into the knowledge_base table."""
        if not pairs:
            logger.warning("No pairs to insert")
            return
        
        # Format pairs for Supabase
        data = [
            {
                "query": pair["query"],
                "response": pair["response"]
            }
            for pair in pairs
        ]
        
        try:
            response = self.client.table(self.table_name).insert(data).execute()
            logger.info(f"Inserted {len(response.data)} pairs into {self.table_name}")
        except Exception as e:
            logger.error(f"Error inserting pairs: {e}")
            raise

    def get_pairs(self) -> List[Dict[str, str]]:
        """Retrieve all query-response pairs."""
        try:
            response = self.client.table(self.table_name).select("*").execute()
            return response.data
        except Exception as e:
            logger.error(f"Error retrieving pairs: {e}")
            return []

if __name__ == "__main__":
    from document_processor import DocumentProcessor
    from llm_generator import LLMGenerator

    # Initialize components
    processor = DocumentProcessor()
    generator = LLMGenerator()
    supabase_client = SupabaseClient()

    # Process document and generate pairs
    chunks = processor.extract_text_from_txt("documents/TELIA for Coil - System Specifications_cleaned.txt")
    for chunk in chunks:
        pairs = generator.generate_pairs(chunk)
        print(f"Inserting {len(pairs)} pairs")
        supabase_client.insert_pairs(pairs)