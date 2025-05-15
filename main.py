from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()

# Set up Together AI API key
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise HTTPException(status_code=500, detail="Together AI API key not found")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define data models
class ExplanationRequest(BaseModel):
    standard_id: str
    scenario: str
    language: str = "English"

class FeedbackRequest(BaseModel):
    standard_id: str
    user_solution: str
    language: str = "English"

class QuestionRequest(BaseModel):
    question: str
    language: str = "English"

# Together AI API client
class TogetherAIClient:
    def __init__(self, api_key: str, model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.together.xyz/v1/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_text(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Generate text using Together AI API"""
        data = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            print(f"Sending request to Together AI with model: {self.model}")
            print(f"Prompt: {prompt[:100]}...")
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=data
            )
            
            if response.status_code != 200:
                error_msg = f"Error from Together AI API: {response.text}"
                print(error_msg)
                return f"Error: {response.status_code}"
            
            result = response.json()
            print(f"Response received. First few characters: {result['choices'][0]['text'][:100]}...")
            return result["choices"][0]["text"]
        except Exception as e:
            print(f"Exception when calling Together AI API: {str(e)}")
            return f"Error: {str(e)}"

# Load standards and examples data
try:
    with open("data/standards.json", "r", encoding="utf-8") as f:
        standards = json.load(f)

    with open("data/examples.json", "r", encoding="utf-8") as f:
        examples = json.load(f)

    with open("data/glossary.json", "r", encoding="utf-8") as f:
        glossary = json.load(f)
except FileNotFoundError:
    # Create empty data structures if files don't exist
    standards = []
    examples = []
    glossary = []
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Create empty JSON files
    with open("data/standards.json", "w", encoding="utf-8") as f:
        json.dump(standards, f)
    
    with open("data/examples.json", "w", encoding="utf-8") as f:
        json.dump(examples, f)
    
    with open("data/glossary.json", "w", encoding="utf-8") as f:
        json.dump(glossary, f)

# Initialize Together AI client
ai_client = TogetherAIClient(api_key=TOGETHER_API_KEY)

class IslamicFinanceStandardsExplainer:
    def __init__(self, ai_client: TogetherAIClient):
        self.ai_client = ai_client
    
    def get_explanation(self, standard: str, standard_title: str, scenario: str, language: str = "English") -> str:
        """Get AI explanation for a specific standard and scenario"""
        if language == "English":
            prompt = f"""You are an expert in Islamic Finance standards, particularly AAOIFI standards.
Explain the accounting treatment for the given scenario in simple terms that a non-specialist can understand.
Use step-by-step explanations and include the journal entries where appropriate.

Standard: {standard_title}

Scenario: {scenario}

Please explain:
1. What this standard is about
2. How to account for this transaction step-by-step
3. The proper journal entries
4. Why this method complies with Islamic finance principles
"""
        else:  # Arabic
            prompt = f"""أنت خبير في معايير التمويل الإسلامي، خاصة معايير هيئة المحاسبة والمراجعة للمؤسسات المالية الإسلامية.
قم بشرح المعالجة المحاسبية للسيناريو المعطى بمصطلحات بسيطة يمكن لغير المتخصص فهمها.
استخدم شرحًا خطوة بخطوة وقم بتضمين قيود اليومية حيثما كان ذلك مناسبًا.

المعيار: {standard_title}

السيناريو: {scenario}

يرجى شرح:
1. ما هو هذا المعيار
2. كيفية المحاسبة عن هذه المعاملة خطوة بخطوة
3. قيود اليومية المناسبة
4. لماذا تتوافق هذه الطريقة مع مبادئ التمويل الإسلامي
"""
        
        return self.ai_client.generate_text(prompt, max_tokens=2048, temperature=0.5)
    
    def get_feedback(self, scenario: str, user_solution: str, expert_solution: str, language: str = "English") -> str:
        """Get feedback on user's solution"""
        if language == "English":
            prompt = f"""You are an expert in Islamic Finance standards. Compare the user's solution to an expert solution and provide feedback.

Scenario: {scenario}

User's solution:
{user_solution}

Expert solution:
{expert_solution}

Provide feedback on the user's solution. Highlight what they got correct and what needs improvement.
Rate their understanding on a scale of 1-10.
"""
        else:  # Arabic
            prompt = f"""أنت خبير في معايير التمويل الإسلامي. قارن حل المستخدم بحل الخبير وقدم تعليقات.

السيناريو: {scenario}

حل المستخدم:
{user_solution}

حل الخبير:
{expert_solution}

قدم تعليقات على حل المستخدم. سلط الضوء على ما أصابوه بشكل صحيح وما يحتاج إلى تحسين.
قيّم فهمهم على مقياس من 1 إلى 10.
"""
        
        return self.ai_client.generate_text(prompt, max_tokens=1024, temperature=0.5)
    
    def ask_custom_question(self, question: str, language: str = "English") -> str:
        """Answer a custom question about Islamic finance"""
        if language == "English":
            prompt = f"""You are an expert in Islamic Finance standards, particularly AAOIFI standards.
The user has asked the following question about Islamic Finance:

{question}

Provide a clear, detailed answer using your knowledge of Islamic finance principles and standards.
Reference specific AAOIFI standards when relevant. Make your explanation easy for non-specialists to understand.
"""
        else:  # Arabic
            prompt = f"""أنت خبير في معايير التمويل الإسلامي، خاصة معايير هيئة المحاسبة والمراجعة للمؤسسات المالية الإسلامية.
طرح المستخدم السؤال التالي حول التمويل الإسلامي:

{question}

قدم إجابة واضحة ومفصلة باستخدام معرفتك بمبادئ ومعايير التمويل الإسلامي.
أشر إلى معايير هيئة المحاسبة والمراجعة للمؤسسات المالية الإسلامية المحددة عندما يكون ذلك ذا صلة. اجعل شرحك سهلاً لغير المتخصصين لفهمه.
"""
        
        return self.ai_client.generate_text(prompt, max_tokens=1536, temperature=0.7)

# Initialize the explainer
explainer = IslamicFinanceStandardsExplainer(ai_client)

# API endpoints
@app.get("/")
def read_root():
    return {"message": "Islamic Finance Standards API"}

@app.get("/api/standards")
def get_standards():
    return standards

@app.get("/api/examples")
def get_examples():
    return examples

@app.get("/api/glossary")
def get_glossary():
    return glossary

@app.post("/api/explanation")
def get_explanation(request: ExplanationRequest):
    try:
        # Find the standard
        standard = next((s for s in standards if s["id"] == request.standard_id), None)
        if not standard:
            raise HTTPException(status_code=404, detail="Standard not found")
        
        # Find the example
        example = next((e for e in examples if e["standard_id"] == request.standard_id), None)
        if not example:
            raise HTTPException(status_code=404, detail="Example not found")
        
        # Get the title based on language
        title_key = "title_en" if request.language == "English" else "title_ar"
        
        # Get explanation
        explanation = explainer.get_explanation(
            standard=request.standard_id,
            standard_title=standard[title_key],
            scenario=request.scenario,
            language=request.language
        )
        
        return {"explanation": explanation}
    
    except Exception as e:
        print(f"Error in explanation endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback")
def get_feedback(request: FeedbackRequest):
    try:
        # Find the standard
        standard = next((s for s in standards if s["id"] == request.standard_id), None)
        if not standard:
            raise HTTPException(status_code=404, detail="Standard not found")
        
        # Find the example
        example = next((e for e in examples if e["standard_id"] == request.standard_id), None)
        if not example:
            raise HTTPException(status_code=404, detail="Example not found")
        
        # Get the scenario based on language
        scenario_key = "scenario_en" if request.language == "English" else "scenario_ar"
        title_key = "title_en" if request.language == "English" else "title_ar"
        
        # Get expert solution first
        expert_solution = explainer.get_explanation(
            standard=request.standard_id,
            standard_title=standard[title_key],
            scenario=example[scenario_key],
            language=request.language
        )
        
        # Get feedback
        feedback = explainer.get_feedback(
            scenario=example[scenario_key],
            user_solution=request.user_solution,
            expert_solution=expert_solution,
            language=request.language
        )
        
        return {
            "feedback": feedback,
            "expert_solution": expert_solution
        }
    
    except Exception as e:
        print(f"Error in feedback endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask")
def ask_question(request: QuestionRequest):
    try:
        # Get answer
        answer = explainer.ask_custom_question(
            question=request.question,
            language=request.language
        )
        
        return {"answer": answer}
    
    except Exception as e:
        print(f"Error in ask endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
