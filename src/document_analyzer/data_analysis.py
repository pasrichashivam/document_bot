import sys
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from prompt.prompt_library import PROMPT_REGISTRY

class DocumentAnalyzer:
    """
    Analyzes documents using a pre-trained model.
    Automatically logs all actions and supports session-based organization.
    """
    def __init__(self):
        self.log = CustomLogger().get_logger(__name__)
        try:
            self.loader=ModelLoader()
            self.llm=self.loader.load_llm()
            # This is used to parse the output from the LLM into a structured format using Pydantic models
            self.parser = JsonOutputParser(pydantic_object=Metadata) 
            # This is used to fix the output from the LLM if it is not in the expected format
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm) 
            self.prompt = PROMPT_REGISTRY["document_analysis"]
            
            self.log.info("DocumentAnalyzer initialized successfully")
        except Exception as e:
            self.log.error(f"Error initializing DocumentAnalyzer: {e}")
            raise DocumentPortalException("Error in DocumentAnalyzer initialization", sys)
    
    def analyze_document(self, document_text:str)-> dict:
        """
        Analyze a document's text and extract structured metadata & summary.
        """
        try:
            # This creates a chain of operations: prompt -> LLM -> output parser which is first we pass the prompt to the LLM 
            # and then parse the output using the fixing parser
            chain = self.prompt | self.llm | self.fixing_parser
            self.log.info("Meta-data analysis chain initialized")
            # Here we pass the format instructions and the document text to the chain format_instructions is used to 
            # tell the LLM what format we expect the output to be in and document_text is the text of the document we want to analyze
            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text
            }) 
            self.log.info("Metadata extraction successful", keys=list(response.keys()))
            return response

        except Exception as e:
            self.log.error("Metadata analysis failed", error=str(e))
            raise DocumentPortalException("Metadata extraction failed",sys)
