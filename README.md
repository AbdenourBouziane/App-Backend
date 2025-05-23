# Islamic Finance Educational App - Backend

A Python FastAPI backend that powers the Islamic Finance Educational App, providing AI-enhanced explanations, feedback, and answers about Islamic Finance Standards.

## 🛠️ Technology Stack

- **FastAPI**: High-performance Python web framework
- **Together AI**: AI model integration for explanations and feedback
- **Python-dotenv**: Environment variable management
- **Uvicorn**: ASGI server
- **Requests**: HTTP client for API calls
- **JSON**: Data storage format

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Together AI API key (sign up at [Together AI](https://www.together.ai/))

### Setup

1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/yourusername/islamic-finance-app.git
   cd islamic-finance-app/backend
   \`\`\`

2. Create a virtual environment:
   \`\`\`bash
   python -m venv venv
   \`\`\`

3. Activate the virtual environment:
   - On Windows:
     \`\`\`bash
     venv\Scripts\activate
     \`\`\`
   - On macOS/Linux:
     \`\`\`bash
     source venv/bin/activate
     \`\`\`

4. Install the required packages:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

5. Create a `.env` file in the backend directory with your Together AI API key:
   \`\`\`
   TOGETHER_API_KEY=your_together_ai_api_key_here
   \`\`\`

6. Populate the data files (if they don't exist):
   \`\`\`bash
   python populate_data.py
   \`\`\`

7. Run the backend server:
   \`\`\`bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   \`\`\`

## 📁 Project Structure

\`\`\`
backend/
├── data/                # JSON data files
│   ├── standards.json   # Islamic finance standards
│   ├── examples.json    # Example scenarios for each standard
│   └── glossary.json    # Glossary terms and definitions
├── main.py              # FastAPI server and API endpoints
├── populate_data.py     # Data initialization script
├── requirements.txt     # Python dependencies
└── .env.example         # Example environment variables file
\`\`\`

## 📚 API Documentation

The backend provides the following RESTful API endpoints:

| Endpoint | Method | Description | Request Body | Response |
|----------|--------|-------------|--------------|----------|
| `/` | GET | Health check | None | `{"message": "Islamic Finance Standards API"}` |
| `/api/standards` | GET | Get all Islamic finance standards | None | Array of standards |
| `/api/examples` | GET | Get all example scenarios | None | Array of examples |
| `/api/glossary` | GET | Get glossary terms and definitions | None | Array of glossary terms |
| `/api/explanation` | POST | Get AI explanation for a standard and scenario | `{"standard_id": "FAS 4", "scenario": "...", "language": "English"}` | `{"explanation": "..."}` |
| `/api/feedback` | POST | Get feedback on a user's solution | `{"standard_id": "FAS 4", "user_solution": "...", "language": "English"}` | `{"feedback": "...", "expert_solution": "..."}` |
| `/api/ask` | POST | Ask a custom question about Islamic finance | `{"question": "...", "language": "English"}` | `{"answer": "..."}` |

## 🧠 AI Integration

This backend uses Together AI's language models to provide intelligent responses. It's configured to use the "mistralai/Mistral-7B-Instruct-v0.2" model by default, which is available through the serverless API.

If you want to use a different model:

1. Update the model name in the `TogetherAIClient` class in `main.py`:
   ```python
   def __init__(self, api_key: str, model: str = "your-preferred-model"):
   \`\`\`

2. Make sure the model is available through Together AI's serverless API or create a dedicated endpoint for models that require it.

## 🔄 Data Files

The backend uses three JSON files to store data:

1. **standards.json**: Contains information about Islamic finance standards
2. **examples.json**: Contains example scenarios for each standard
3. **glossary.json**: Contains glossary terms and their definitions

If these files don't exist, the backend will create empty files. You can populate them with sample data using the `populate_data.py` script.

## 🔧 Troubleshooting

- **404 Not Found errors**: Make sure the data files exist and contain valid data
- **500 Internal Server errors**: Check the console logs for detailed error messages
- **AI model errors**: Verify your Together AI API key and check if the model is available
