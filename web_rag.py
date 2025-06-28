"""TO Build a web based rag system on a website using Langchain.
This web based RAG System is built on the ChaiCode Docs Site with enhanced explanatory answers and source URLs"""

from pathlib import Path
import os
import sys
from dotenv import load_dotenv
from urllib.parse import urlparse
import re
import logging
from typing import List, Tuple, Any, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from getpass import getpass
from langchain_community.document_loaders import WebBaseLoader
from langchain.load import dumps, loads

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def setup_environment():
    """Setup environment variables with proper validation"""
    if not os.getenv("QDRANT_URL"):
        qdrant = getpass("Enter your Qdrant URL: ")
        os.environ["QDRANT_URL"] = qdrant

    if not os.getenv("QDRANT_API_KEY") and not os.getenv("QDRANT_API_KEY_FILE"):
        api_key = getpass("Enter your Qdrant API KEY: ")
        os.environ["QDRANT_API_KEY"] = api_key

    if not os.getenv("GOOGLE_API_KEY"):
        api_key = getpass("Enter your Google API KEY: ") 
        os.environ["GOOGLE_API_KEY"] = api_key

# List of URLs to load
urls = [
    "https://docs.chaicode.com/", 
    "https://docs.chaicode.com/youtube/chai-aur-html/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-html/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-html/emmit-crash-course/",
    "https://docs.chaicode.com/youtube/chai-aur-html/html-tags/",
    "https://docs.chaicode.com/youtube/chai-aur-git/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-git/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-git/terminology/",
    "https://docs.chaicode.com/youtube/chai-aur-git/behind-the-scenes/",
    "https://docs.chaicode.com/youtube/chai-aur-git/branches/",
    "https://docs.chaicode.com/youtube/chai-aur-git/diff-stash-tags/",
    "https://docs.chaicode.com/youtube/chai-aur-git/managing-history/",
    "https://docs.chaicode.com/youtube/chai-aur-git/github/",
    "https://docs.chaicode.com/youtube/chai-aur-c/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-c/hello-world/",
    "https://docs.chaicode.com/youtube/chai-aur-c/variables-and-constants/",
    "https://docs.chaicode.com/youtube/chai-aur-c/data-types/",
    "https://docs.chaicode.com/youtube/chai-aur-c/operators/",
    "https://docs.chaicode.com/youtube/chai-aur-c/control-flow/",
    "https://docs.chaicode.com/youtube/chai-aur-c/loops/",
    "https://docs.chaicode.com/youtube/chai-aur-c/functions/",
    "https://docs.chaicode.com/youtube/chai-aur-django/getting-started/",
    "https://docs.chaicode.com/youtube/chai-aur-django/jinja-templates/",
    "https://docs.chaicode.com/youtube/chai-aur-django/tailwind/",
    "https://docs.chaicode.com/youtube/chai-aur-django/models/",
    "https://docs.chaicode.com/youtube/chai-aur-django/relationships-and-forms/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/postgres/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/normalization/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/database-design-exercise/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/joins-and-keys/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/joins-exercise/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/setup-vpc/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/setup-nginx/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/nginx-rate-limiting/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/nginx-ssl-setup/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/node-nginx-vps/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-docker/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-vps/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/node-logger/"
]

def load_and_split_documents():
    """Load documents from URLs and split them into chunks"""
    try:
        loader = WebBaseLoader(urls)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents from {len(urls)} URLs")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
        split_docs = text_splitter.split_documents(documents=docs)
        logger.info(f"Split into {len(split_docs)} chunks")
        
        return split_docs
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []

def get_collection_name(url):
    """Generate a valid collection name from URL"""
    parsed = urlparse(url)
    path = parsed.path.strip('/')
    if not path:
        return "chaicode_main"
    
    collection_name = re.sub(r'[^a-zA-Z0-9_]', '_', path)
    collection_name = collection_name.strip('_')
    
    if collection_name and collection_name[0].isdigit():
        collection_name = f"col_{collection_name}"
    
    return collection_name[:50] if collection_name else "chaicode_default"

def create_embeddings_single_store(split_docs):
    """Create embeddings using a single vector store with multiple collections"""
    logger.info("Creating Embeddings")
    
    try:
        embedders = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            api_key=os.getenv("GOOGLE_API_KEY"),
        )
        
        # Group documents by their source URL
        docs_by_url = {}
        for doc in split_docs:
            source_url = doc.metadata.get('source', 'unknown')
            collection_name = get_collection_name(source_url)
            
            if collection_name not in docs_by_url:
                docs_by_url[collection_name] = []
            docs_by_url[collection_name].append(doc)
        
        # Create the first collection to initialize the vector store
        first_collection = list(docs_by_url.keys())[0]
        first_docs = docs_by_url[first_collection]
        
        logger.info(f"Initializing with collection: {first_collection}")
        vector_store = QdrantVectorStore.from_documents(
            documents=first_docs,
            embedding=embedders,
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=first_collection
        )
        
        # Add remaining collections
        for collection_name, docs in list(docs_by_url.items())[1:]:
            logger.info(f"Adding {len(docs)} documents to collection: {collection_name}")
            try:
                QdrantVectorStore.from_documents(
                    documents=docs,
                    embedding=embedders,
                    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                    api_key=os.getenv("QDRANT_API_KEY"),
                    collection_name=collection_name
                )
                logger.info(f"Successfully created collection: {collection_name}")
            except Exception as e:
                logger.error(f"Error creating collection {collection_name}: {e}")
        
        return vector_store
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        return None

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system"""
        setup_environment()
        
        # Load and process documents
        self.split_docs = load_and_split_documents()
        if not self.split_docs:
            raise ValueError("Failed to load documents")
        
        # Create vector store
        self.vector_store = create_embeddings_single_store(self.split_docs)
        if not self.vector_store:
            raise ValueError("Failed to create vector store")
        
        # Initialize retriever - FIXED TYPO
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15}
        )
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-1.5-flash',
            temperature=0.1,
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        logger.info("RAG System initialized successfully")

    def generate_multiple_questions(self, question: str) -> List[str]:
        """Generate multiple related questions based on the input question"""
        try:
            system_prompt = f"""
            You are a helpful AI assistant specialized in generating similar queries.
            
            Given the user question: "{question}"
            
            Generate 4 additional questions that are:
            1. Related to the same topic
            2. Different perspectives on the same subject
            3. More specific or general versions of the original question
            4. Questions that could help gather comprehensive information
            
            Return only the 4 questions, one per line, without numbering or bullet points.
            """
            
            response = self.llm.invoke(system_prompt)
            
            # Parse the response to extract questions
            generated_questions = [q.strip() for q in response.content.split('\n') if q.strip()]
            
            # Include original question and filter out empty responses
            all_questions = [question] + generated_questions[:4]  # Limit to 4 additional questions
            
            logger.info(f"Generated {len(all_questions)} questions total")
            return all_questions
            
        except Exception as e:
            logger.error(f"Error generating multiple questions: {e}")
            return [question]  # Return at least the original question

    def get_available_collections(self) -> List[str]:
        """Get all available collections from the vector store"""
        try:
            # Get all collection names based on your URL structure
            collections = [
                "chaicode_main",
                "youtube_chai_aur_html_welcome",
                "youtube_chai_aur_html_introduction", 
                "youtube_chai_aur_html_emmit_crash_course",
                "youtube_chai_aur_html_html_tags",
                "youtube_chai_aur_git_welcome",
                "youtube_chai_aur_git_introduction",
                "youtube_chai_aur_git_terminology",
                "youtube_chai_aur_git_behind_the_scenes",
                "youtube_chai_aur_git_branches",
                "youtube_chai_aur_git_diff_stash_tags",
                "youtube_chai_aur_git_managing_history",
                "youtube_chai_aur_git_github",
                "youtube_chai_aur_c_introduction",
                "youtube_chai_aur_c_hello_world",
                "youtube_chai_aur_c_variables_and_constants",
                "youtube_chai_aur_c_data_types",
                "youtube_chai_aur_c_operators",
                "youtube_chai_aur_c_control_flow",
                "youtube_chai_aur_c_loops",
                "youtube_chai_aur_c_functions",
                "youtube_chai_aur_django_getting_started",
                "youtube_chai_aur_django_jinja_templates",
                "youtube_chai_aur_django_tailwind",
                "youtube_chai_aur_django_models",
                "youtube_chai_aur_django_relationships_and_forms",
                "youtube_chai_aur_sql_introduction",
                "youtube_chai_aur_sql_postgres",
                "youtube_chai_aur_sql_normalization",
                "youtube_chai_aur_sql_database_design_exercise",
                "youtube_chai_aur_sql_joins_and_keys",
                "youtube_chai_aur_sql_joins_exercise",
                "youtube_chai_aur_devops_setup_vpc",
                "youtube_chai_aur_devops_setup_nginx",
                "youtube_chai_aur_devops_nginx_rate_limiting",
                "youtube_chai_aur_devops_nginx_ssl_setup",
                "youtube_chai_aur_devops_node_nginx_vps",
                "youtube_chai_aur_devops_postgresql_docker",
                "youtube_chai_aur_devops_postgresql_vps",
                "youtube_chai_aur_devops_node_logger"
            ]
            return collections
        except Exception as e:
            logger.error(f"Error getting collections: {e}")
            return []

    def identify_relevant_collections(self, question: str) -> List[str]:
        """Identify which collections are most relevant to the question using LLM"""
        try:
            available_collections = self.get_available_collections()
            
            # Create a mapping of topics to collections for better LLM understanding
            collection_topics = {
                "HTML": ["youtube_chai_aur_html_welcome", "youtube_chai_aur_html_introduction", 
                        "youtube_chai_aur_html_emmit_crash_course", "youtube_chai_aur_html_html_tags"],
                "Git": ["youtube_chai_aur_git_welcome", "youtube_chai_aur_git_introduction", 
                       "youtube_chai_aur_git_terminology", "youtube_chai_aur_git_behind_the_scenes",
                       "youtube_chai_aur_git_branches", "youtube_chai_aur_git_diff_stash_tags",
                       "youtube_chai_aur_git_managing_history", "youtube_chai_aur_git_github"],
                "C Programming": ["youtube_chai_aur_c_introduction", "youtube_chai_aur_c_hello_world",
                                 "youtube_chai_aur_c_variables_and_constants", "youtube_chai_aur_c_data_types",
                                 "youtube_chai_aur_c_operators", "youtube_chai_aur_c_control_flow",
                                 "youtube_chai_aur_c_loops", "youtube_chai_aur_c_functions"],
                "Django": ["youtube_chai_aur_django_getting_started", "youtube_chai_aur_django_jinja_templates",
                          "youtube_chai_aur_django_tailwind", "youtube_chai_aur_django_models",
                          "youtube_chai_aur_django_relationships_and_forms"],
                "SQL": ["youtube_chai_aur_sql_introduction", "youtube_chai_aur_sql_postgres",
                       "youtube_chai_aur_sql_normalization", "youtube_chai_aur_sql_database_design_exercise",
                       "youtube_chai_aur_sql_joins_and_keys", "youtube_chai_aur_sql_joins_exercise"],
                "DevOps": ["youtube_chai_aur_devops_setup_vpc", "youtube_chai_aur_devops_setup_nginx",
                          "youtube_chai_aur_devops_nginx_rate_limiting", "youtube_chai_aur_devops_nginx_ssl_setup",
                          "youtube_chai_aur_devops_node_nginx_vps", "youtube_chai_aur_devops_postgresql_docker",
                          "youtube_chai_aur_devops_postgresql_vps", "youtube_chai_aur_devops_node_logger"],
                "General": ["chaicode_main"]
            }
            
            system_prompt = f"""
            You are an expert at analyzing questions and identifying relevant topics.
            
            Question: "{question}"
            
            Available topics and their collections:
            {chr(10).join([f"{topic}: {', '.join(collections)}" for topic, collections in collection_topics.items()])}
            
            Instructions:
            1. Analyze the question to identify which programming topics/technologies it's about
            2. Select the most relevant collections that could contain information to answer this question
            3. Return ONLY the collection names, one per line, without any explanations
            4. If the question is general or about multiple topics, include relevant collections for each
            5. Always include "chaicode_main" if the question might need general information
            6. Maximum 8 collections to keep search focused
            
            Example responses:
            - For HTML questions: youtube_chai_aur_html_introduction
            - For Git questions: youtube_chai_aur_git_terminology
            - For programming questions: youtube_chai_aur_c_introduction
            
            Relevant collections:
            """
            
            response = self.llm.invoke(system_prompt)
            
            # Parse the response to extract collection names
            identified_collections = []
            for line in response.content.strip().split('\n'):
                collection_name = line.strip()
                if collection_name in available_collections:
                    identified_collections.append(collection_name)
            
            # Fallback: if no valid collections identified, use a broader search
            if not identified_collections:
                logger.warning("No specific collections identified, using broader search")
                # Add general collection and try to match keywords
                identified_collections = ["chaicode_main"]
                
                question_lower = question.lower()
                if any(word in question_lower for word in ['html', 'tag', 'element', 'css']):
                    identified_collections.extend(collection_topics["HTML"][:2])
                elif any(word in question_lower for word in ['git', 'version', 'commit', 'branch']):
                    identified_collections.extend(collection_topics["Git"][:2])
                elif any(word in question_lower for word in ['c', 'programming', 'variable', 'function']):
                    identified_collections.extend(collection_topics["C Programming"][:2])
                elif any(word in question_lower for word in ['django', 'python', 'web', 'framework']):
                    identified_collections.extend(collection_topics["Django"][:2])
                elif any(word in question_lower for word in ['sql', 'database', 'query', 'join']):
                    identified_collections.extend(collection_topics["SQL"][:2])
                elif any(word in question_lower for word in ['devops', 'nginx', 'docker', 'server']):
                    identified_collections.extend(collection_topics["DevOps"][:2])
            
            # Remove duplicates and limit to 8 collections
            identified_collections = list(set(identified_collections))[:8]
            
            logger.info(f"Identified {len(identified_collections)} relevant collections: {identified_collections}")
            return identified_collections
            
        except Exception as e:
            logger.error(f"Error identifying relevant collections: {e}")
            return ["chaicode_main"]  # Fallback to general collection

    def find_sources(self, question: str) -> Tuple[List[Any], List[str]]:
        """Find relevant documents by first identifying relevant collections, then searching within them"""
        try:
            logger.info(f"Finding relevant documents for: {question}")
            
            # Step 1: Identify relevant collections based on the question
            relevant_collections = self.identify_relevant_collections(question)
            
            if not relevant_collections:
                logger.warning("No relevant collections identified")
                return [], []
            
            logger.info(f"Searching in collections: {relevant_collections}")
            
            # Step 2: Search for documents within each relevant collection
            all_relevant_docs = []
            
            for collection_name in relevant_collections:
                try:
                    # Create a retriever for this specific collection
                    collection_retriever = QdrantVectorStore(
                        client=self.vector_store.client,
                        collection_name=collection_name,
                        embedding=GoogleGenerativeAIEmbeddings(
                            model="models/embedding-001",
                            api_key=os.getenv("GOOGLE_API_KEY")
                        )
                    ).as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 10}  # Get top 10 from each collection
                    )
                    
                    # Get relevant documents from this collection
                    collection_docs = collection_retriever.get_relevant_documents(question)
                    
                    # Add collection info to metadata for tracking
                    for doc in collection_docs:
                        doc.metadata['collection_used'] = collection_name
                    
                    all_relevant_docs.extend(collection_docs)
                    logger.info(f"Found {len(collection_docs)} documents in collection: {collection_name}")
                    
                except Exception as e:
                    logger.warning(f"Error searching in collection {collection_name}: {e}")
                    continue
            
            logger.info(f"Total found {len(all_relevant_docs)} relevant documents from {len(relevant_collections)} collections")
            return all_relevant_docs, relevant_collections
            
        except Exception as e:
            logger.error(f"Error finding relevant documents: {e}")
            return [], []

    def reciprocal_rank_fusion(self, results: List[List[Any]], k: int = 60) -> List[Tuple[Any, float]]:
        """Apply Reciprocal Rank Fusion to combine multiple ranked lists"""
        try:
            fused_scores = {}
            
            for docs in results:
                for rank, doc in enumerate(docs):
                    doc_str = dumps(doc)
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0
                    fused_scores[doc_str] += 1 / (rank + k)
            
            # Sort by fused scores in descending order
            reranked_results = [
                (loads(doc), score)
                for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            ]
            
            logger.info(f"Reranked {len(reranked_results)} unique documents")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in reciprocal rank fusion: {e}")
            return []

    def extract_source_urls(self, reranked_results: List[Tuple[Any, float]]) -> List[str]:
        """Extract unique source URLs from the reranked results"""
        try:
            source_urls = set()
            
            for doc, score in reranked_results[:10]:  # Get URLs from top 10 documents
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    source_urls.add(doc.metadata['source'])
                elif isinstance(doc, dict) and 'metadata' in doc and 'source' in doc['metadata']:
                    source_urls.add(doc['metadata']['source'])
            
            return list(source_urls)
            
        except Exception as e:
            logger.error(f"Error extracting source URLs: {e}")
            return []

    def format_urls_for_display(self, urls: List[str]) -> str:
        """Format URLs for better display with titles"""
        if not urls:
            return ""
        
        formatted_urls = []
        for url in urls:
            # Extract a readable title from the URL
            title = url.replace("https://docs.chaicode.com/", "").replace("/", " > ").replace("-", " ").title()
            if not title or title == "":
                title = "ChaiCode Docs Home"
            
            formatted_urls.append(f"📖 {title}: {url}")
        
        return "\n".join(formatted_urls)

    def generate_final_answer(self, original_question: str, reranked_results: List[Tuple[Any, float]]) -> Tuple[str, List[str]]:
        """Generate the final answer using the original question and reranked documents"""
        try:
            # Extract document content for context (using more documents for better context)
            context_docs = []
            for doc, score in reranked_results[:8]:  # Use top 8 documents for better coverage
                if hasattr(doc, 'page_content'):
                    context_docs.append(f"Document (Relevance Score: {score:.3f}):\n{doc.page_content[:1500]}...")
                elif isinstance(doc, dict) and 'page_content' in doc:
                    context_docs.append(f"Document (Relevance Score: {score:.3f}):\n{doc['page_content'][:1500]}...")
            
            context = "\n\n" + "="*50 + "\n\n".join(context_docs)
            
            # Extract source URLs
            source_urls = self.extract_source_urls(reranked_results)
            
            system_prompt = f"""
            You are an expert AI assistant specialized in providing comprehensive, detailed, and educational answers based on programming documentation.
            
            Question: {original_question}
            
            Context from ChaiCode documentation:
            {context}
            
            Instructions for generating a comprehensive answer:
            1. **Provide a detailed, step-by-step explanation** - Break down complex concepts into digestible parts
            2. **Include practical examples** - Show code snippets, commands, or practical applications where relevant
            3. **Explain the "why" behind concepts** - Don't just state what something is, explain why it's important or how it works
            4. **Use clear structure** - Organize your response with headers, bullet points, or numbered lists when appropriate
            5. **Include best practices** - Mention tips, common mistakes to avoid, or recommended approaches
            6. **Make it educational** - Assume the user wants to learn, not just get a quick answer
            7. **Reference the context** - Use specific information from the provided documentation
            8. **Be comprehensive but focused** - Cover the topic thoroughly while staying relevant to the question
            
            If the context doesn't contain sufficient information, clearly state what information is missing and provide what you can based on the available context.
            
            Format your response as:
            ## Answer
            [Your comprehensive answer here]
            
            ## Key Points
            [Bullet points of the most important takeaways]
            
            ## Additional Context
            [Any relevant background information or related concepts]
            
            Answer:
            """
            
            response = self.llm.invoke(system_prompt)
            logger.info("Enhanced final answer generated successfully")
            return response.content, source_urls
            
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            return "I apologize, but I encountered an error while generating the answer.", []

    def rag_main_chain(self, question: str) -> Tuple[str, List[Tuple[Any, float]], List[str], List[str]]:
        """Main RAG chain that processes the question and returns the final answer with source URLs"""
        try:
            logger.info(f"Starting RAG chain for question: {question}")
            
            # Step 1: Generate multiple questions
            multiple_questions = self.generate_multiple_questions(question)
            logger.info(f"Generated {len(multiple_questions)} questions")
            
            # Step 2: Find relevant documents for each question
            all_relevant_docs = []
            all_collections = set()
            
            for q in multiple_questions:
                relevant_docs, relevant_collections = self.find_sources(q)
                if relevant_docs:  # Only add if we found documents
                    all_relevant_docs.append(relevant_docs)
                    all_collections.update(relevant_collections)
            
            if not all_relevant_docs:
                return "I couldn't find any relevant documents for your question.", [], [], []
            
            # Step 3: Apply reciprocal rank fusion
            reranked_results = self.reciprocal_rank_fusion(all_relevant_docs)
            
            if not reranked_results:
                return "I couldn't rank the retrieved documents.", [], [], []
            
            # Step 4: Generate final answer and extract source URLs
            final_answer, source_urls = self.generate_final_answer(question, reranked_results)
            
            logger.info("RAG chain completed successfully")
            return final_answer, reranked_results, list(all_collections), source_urls
            
        except Exception as e:
            logger.error(f"Error in RAG main chain: {e}")
            return f"An error occurred while processing your question: {str(e)}", [], [], []

def main():
    """Main function to run the RAG system"""
    try:
        # Initialize RAG system
        print("🚀 Initializing Enhanced RAG System...")
        rag_system = RAGSystem()
        print("✅ RAG System initialized successfully!")
        print("🌐 You can view your collection at: http://localhost:6333/dashboard")
        print("📚 This system provides detailed explanations with source links for better learning!")
        
        # Interactive loop
        while True:
            user_question = input("\n🗣️ Ask your question (or type 'exit' to quit): ").strip()

            if user_question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if user_question:
                try:
                    print("🔍 Processing your question...")
                    final_answer, reranked_results, relevant_collections, source_urls = rag_system.rag_main_chain(user_question)
                    
                    print(f"\n{'='*60}")
                    print(f"📝 DETAILED ANSWER:")
                    print(f"{'='*60}")
                    print(final_answer)
                    
                    # Display source URLs if available
                    if source_urls:
                        print(f"\n{'='*60}")
                        print("🔗 RELEVANT SOURCES:")
                        print(f"{'='*60}")
                        formatted = rag_system.format_urls_for_display(source_urls)
                        print(formatted)
                    else:
                        print("\n⚠️ No source URLs could be retrieved.")
                
                except Exception as e:
                    print(f"❌ Error during processing: {e}")
    
    except Exception as main_error:
        print(f"❌ Critical Error: {main_error}")


if __name__ == "__main__":
    main()
