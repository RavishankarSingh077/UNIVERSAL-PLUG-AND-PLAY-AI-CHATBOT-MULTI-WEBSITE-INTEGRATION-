# Universal Website Chatbot (Multi-Website Integration)

A professional AI-powered chatbot that can be integrated into any website. It provides intelligent responses about a company's services and content using **Mistral AI / Groq** + **ChromaDB** (Retrieval Augmented Generation).

## 🚀 Features

- **Multi-LLM Support**: Prioritizes **Mistral AI** (`mistral-medium-latest`) with automatic fallback to **Groq** (`llama-3.1-8b-instant`).
- **RAG System**: Scrapes and learns from website content using ChromaDB for high-accuracy responses.
- **Branding-Aware**: Dynamically extracts company info and adapts its voice to the website.
- **Mobile Responsive**: Fully optimized for desktop and mobile browsers.
- **Docker Ready**: Includes a Dockerfile for easy deployment on Hugging Face Spaces or other cloud providers.

## 📁 Project Structure

```
.
├── app_chromadb.py         # Main FastAPI backend (RAG + LLM)
├── chatbot-widget.html     # All-in-one widget for easy integration
├── frontend/               # Dedicated frontend files
│   └── chatbot.html        # Enhanced chat interface
├── requirements.txt        # Python dependencies
├── Dockerfile              # Deployment configuration
└── env.example             # Environment variable template
```

## 🛠️ Local Setup Instructions

### 1. Prerequisites
- Python 3.10 or higher
- Mistral AI and/or Groq API Keys

### 2. Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/RavishankarSingh077/UNIVERSAL-PLUG-AND-PLAY-AI-CHATBOT-MULTI-WEBSITE-INTEGRATION-.git
   cd UNIVERSAL-PLUG-AND-PLAY-AI-CHATBOT-MULTI-WEBSITE-INTEGRATION-
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file from `env.example`:
   ```bash
   cp env.example .env
   ```
   Edit `.env` and add your `MISTRAL_API_KEY` or `GROQ_API_KEY`, and set your `WEBSITE_URL`.

### 3. Running the Application
1. **Start the Backend**:
   ```bash
   uvicorn app_chromadb:app --reload
   ```
   The API will be live at `http://127.0.0.1:8000`.

2. **Run the Frontend**:
   Simply open `chatbot-widget.html` or `frontend/chatbot.html` in your web browser.

---

## ☁️ Deployment (Hugging Face / Cloud)

This project is Docker-ready. For Hugging Face Spaces:
1. Create a new **Docker Space**.
2. Connect this repository.
3. Configure the `MISTRAL_API_KEY` and other variables in the Space settings.

---

**Universal AI Chatbot** - Powered by Mistral, Groq, and ChromaDB. 🚀
