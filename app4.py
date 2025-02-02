import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
import logging
import re
import pickle
import base64
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from groq import Groq
from PIL import Image
import io
from dotenv import load_dotenv

# Configure logging for application monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables and set constants
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATASET_PATH = r"D:\sih-recommendation\synthetic_bail_decision_data.csv"

class EnhancedLegalCaseAnalyzer:
    # Main class for analyzing legal cases using ML models and LLMs
    def __init__(self,
                 cache_dir: str = "cache",
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 llm_model: str = "llama-3.3-70b-versatile",
                 vision_model: str = "llama-3.2-11b-vision-preview"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.dataset_path = DATASET_PATH
        self.embedding_cache_path = self.cache_dir / "embeddings.pkl"
        self.vision_model = vision_model

        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        self.groq_client = Groq()

        try:
            self.embedder = SentenceTransformer(embedding_model)
            self.llm = self._initialize_llm(llm_model)
            self.df = self._load_and_preprocess_dataset()
            self.index, self.embeddings = self._load_or_create_embeddings()
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def _initialize_llm(self, model_name: str) -> Any:
        return ChatGroq(
            model=model_name,
            temperature=0.2,
            max_tokens=500,
            timeout=90,
            max_retries=3
        )

    def _load_and_preprocess_dataset(self) -> pd.DataFrame:
        # Load and combine relevant fields for similarity search
        try:
            df = pd.read_csv(self.dataset_path)
            df['description'] = (
                "Legal Classification: " + df['Act_Section'].fillna('') + " " +
                "Offence Type: " + df['Offence_Type'].fillna('') + " " +
                "Severity: " + df['Crime_Severity'].fillna('Medium') + " " +
                "Prior Convictions: " + df['Prior_Convictions'].fillna('0').astype(str) + " " +
                "Mitigating Factors: " + df['Mitigating_Circumstances'].fillna('None') + " " +
                "Public Impact: " + df['Public_Impact'].fillna('Minimal') + " " +
                "Judicial Insights: " + df['Judge_Comments'].fillna('')
            )
            return df
        except Exception as e:
            logger.error(f"Dataset loading error: {e}")
            raise

    def _encode_image(self, image_data: bytes) -> str:
        try:
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Image encoding error: {e}")
            raise

    def process_document_ocr(self, image_data: bytes) -> str:
        # Extract text from document images using vision model
        try:
            base64_image = self._encode_image(image_data)
            
            completion = self.groq_client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract and provide all text content from this legal document image."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None
            )
            
            extracted_text = completion.choices[0].message.content
            logger.info("Successfully extracted text from image")
            return extracted_text
            
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            raise

    def analyze_document_with_ocr(self, image_data: bytes) -> Dict[str, Any]:
        try:
            extracted_text = self.process_document_ocr(image_data)
            analysis = self.analyze_query(extracted_text)
            
            return {
                'extracted_text': extracted_text,
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Document analysis error: {e}")
            return {'error': str(e)}

    def _load_or_create_embeddings(self) -> Tuple[faiss.Index, np.ndarray]:
        # Load cached embeddings or compute new ones if needed
        try:
            if self.embedding_cache_path.exists():
                logger.info("Loading cached embeddings...")
                with open(self.embedding_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                if (cache_data['dataset_path'] == self.dataset_path and
                    cache_data['dataset_modified'] == os.path.getmtime(self.dataset_path)):
                    embeddings = cache_data['embeddings']
                    logger.info("Successfully loaded cached embeddings")
                else:
                    logger.info("Cache invalid or outdated, computing new embeddings...")
                    embeddings = self._compute_and_cache_embeddings()
            else:
                logger.info("No cache found, computing embeddings...")
                embeddings = self._compute_and_cache_embeddings()

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            return index, embeddings

        except Exception as e:
            logger.error(f"Error in loading/creating embeddings: {e}")
            raise

    def _compute_and_cache_embeddings(self) -> np.ndarray:
        embeddings = self.embedder.encode(
            self.df['description'].tolist(),
            convert_to_tensor=False,
            show_progress_bar=True
        )

        cache_data = {
            'embeddings': embeddings,
            'dataset_path': self.dataset_path,
            'dataset_modified': os.path.getmtime(self.dataset_path)
        }

        with open(self.embedding_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        logger.info("Embeddings computed and cached successfully")
        return embeddings

    def analyze_query(self, query: str) -> Dict[str, Any]:
        # Main analysis function for legal queries
        try:
            query_embedding = self.embedder.encode([query])
            similar_cases = self.search_similar_cases(query_embedding)
            
            if not similar_cases:
                response = self.llm.invoke([("human", f"Analyze this legal query: {query}")])
                return {'response': response.content.strip()}

            case_summaries = []
            for case in similar_cases[:3]:
                prompt = f"Summarize this case:\n{case['description']}"
                summary = self.llm.invoke([("human", prompt)])
                case_summaries.append({
                    'Case_ID': case['Case_ID'],
                    'summary': summary.content.strip(),
                    'bail_status': case['bail_status']
                })

            recommendation = self.llm.invoke([
                ("human", f"Based on these cases: {case_summaries}, what's your recommendation for: {query}")
            ])

            return {
                'similar_cases': case_summaries,
                'recommendation': recommendation.content.strip()
            }

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {'error': str(e)}

    def search_similar_cases(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        # Find similar cases using FAISS similarity search
        try:
            distances, indices = self.index.search(query_embedding, top_k)
            similar_cases = []

            for idx in indices[0]:
                case = self.df.iloc[idx]
                similar_cases.append({
                    'Case_ID': case['Case_ID'],
                    'description': case['description'],
                    'bail_status': case['Bail_Status']
                    
                })

            return similar_cases
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

def init_speech_recognizer():
    return sr.Recognizer()

def record_audio() -> Optional[str]:
    # Record and transcribe audio input
    recognizer = init_speech_recognizer()
    
    with sr.Microphone() as source:
        st.write("🎤 Listening... Speak your query")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.write("Processing speech...")
            return recognizer.recognize_google(audio)
        except sr.WaitTimeoutError:
            st.error("No speech detected")
        except sr.UnknownValueError:
            st.error("Could not understand audio")
        except sr.RequestError:
            st.error("Speech recognition service error")
    return None

def text_to_speech(text: str) -> None:
    # Convert analysis results to speech
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            st.audio(fp.name)
        os.unlink(fp.name)
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")

def display_analysis_results(result: dict):
    # Display analysis results in the Streamlit interface
    if 'error' in result:
        st.error(f"Error: {result['error']}")
        return

    if 'similar_cases' in result:
        st.subheader("📋 Similar Cases")
        for i, case in enumerate(result['similar_cases'], 1):
            with st.expander(f"Case {i}"):
                st.info(f"Case_ID: {case['Case_ID']}")
                st.write(case['summary'])
                st.info(f"Bail Status: {case['bail_status']}")

        st.subheader("💡 Recommendation")
        st.write(result['recommendation'])
        
        if st.session_state.get('enable_voice', False):
            text_to_speech(result['recommendation'])
    else:
        st.write(result['response'])
        if st.session_state.get('enable_voice', False):
            text_to_speech(result['response'])

def main():
    # Main application setup and interface
    st.set_page_config(
        page_title="Legal Case Analyzer",
        page_icon="⚖️",
        layout="wide"
    )

    st.title("⚖️ Legal Case Analyzer")
    st.markdown("---")

    if 'analyzer' not in st.session_state:
        try:
            st.session_state.analyzer = EnhancedLegalCaseAnalyzer()
            st.success("System initialized successfully!")
        except Exception as e:
            st.error(f"Initialization error: {e}")
            return

    with st.sidebar:
        st.header("Settings")
        st.checkbox("Enable Voice Features", key="enable_voice")

    input_type = st.radio(
        "Choose Input Method:",
        ["Text", "Voice", "Document Image"],
        horizontal=True
    )

    if input_type == "Text":
        query = st.text_area("Enter your legal query:", height=100)
        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                result = st.session_state.analyzer.analyze_query(query)
                display_analysis_results(result)

    elif input_type == "Voice":
        if not st.session_state.get('enable_voice', False):
            st.warning("Please enable voice features in the sidebar.")
            return
            
        if st.button("Start Recording"):
            query = record_audio()
            if query:
                st.info(f"Query: {query}")
                with st.spinner("Analyzing..."):
                    result = st.session_state.analyzer.analyze_query(query)
                    display_analysis_results(result)

    else:
        uploaded_file = st.file_uploader(
            "Upload a document image (JPG, PNG, etc.):",
            type=["jpg", "jpeg", "png", "bmp"]
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Document", use_column_width=True)
            
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    result = st.session_state.analyzer.analyze_document_with_ocr(uploaded_file.getvalue())
                    
                    if 'error' not in result:
                        st.subheader("📄 Extracted Text")
                        st.write(result['extracted_text'])
                        
                        st.subheader("🔍 Analysis")
                        display_analysis_results(result['analysis'])
                    else:
                        st.error(f"Error processing document: {result['error']}")

    st.markdown("---")
    st.caption("*Built with Streamlit and Enhanced Legal Case Analyzer*")

if __name__ == "__main__":
    main()