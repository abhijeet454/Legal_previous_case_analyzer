# Legal Case Analyzer 🏛️

A sophisticated legal case analysis system that leverages AI to process, analyze, and provide insights on legal cases. The system supports text, voice, and document image inputs to assist legal professionals in case analysis and decision-making.

## 🌟 Features

- **Multi-Modal Input Support**
  - Text-based queries
  - Voice input with speech-to-text conversion
  - Document image processing with OCR capabilities

- **Advanced Analysis Capabilities**
  - Case similarity search using FAISS
  - Natural language processing with LLM integration
  - Document OCR with vision model support
  - Intelligent case recommendations

- **Interactive User Interface**
  - Built with Streamlit for seamless interaction
  - Voice output capabilities
  - Expandable case summaries
  - Real-time processing feedback

## 🔧 Technical Stack

- **Core Technologies**
  - Python 3.8+
  - Streamlit for UI
  - FAISS for similarity search
  - Sentence Transformers for embeddings
  - Groq for LLM integration
  - Speech Recognition for voice input
  - gTTS for text-to-speech

- **Key Libraries**
  - `streamlit`
  - `faiss-cpu`
  - `sentence-transformers`
  - `langchain-groq`
  - `speech_recognition`
  - `gtts`
  - `PIL`
  - `pandas`
  - `numpy`

## 📋 Prerequisites

1. Python 3.8 or higher
2. Groq API key
3. Required Python packages (listed in requirements.txt)
4. Microphone access (for voice features)
5. Internet connection for API services

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/legal-case-analyzer.git
cd legal-case-analyzer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

## 🚀 Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the interface through your web browser (typically http://localhost:8501)

3. Choose your preferred input method:
   - Text: Type your legal query directly
   - Voice: Enable voice features and speak your query
   - Document Image: Upload legal document images for analysis

## 💡 Key Features Explained

### Case Similarity Search
- Uses FAISS for efficient similarity searching
- Embeddings are cached for improved performance
- Automatically finds relevant similar cases

### Document OCR
- Processes uploaded document images
- Extracts text using advanced vision models
- Analyzes extracted content for insights

### Voice Interface
- Speech-to-text for query input
- Text-to-speech for result readout
- Interactive voice commands

## 🔒 Security Considerations

- API keys are stored securely in environment variables
- Local data caching for improved performance
- Error handling for sensitive operations

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

- Groq for LLM capabilities
- Sentence Transformers team
- FAISS developers
- Streamlit community

## 📞 Support

For support, please open an issue in the GitHub repository or contact [your-email@domain.com].

---

⭐ **Star this repository if you find it helpful!** ⭐
