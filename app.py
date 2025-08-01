# D:\medical_ai_project\app.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from medical_ai_system import MedicalRecommendationSystem # Import your main system
import uvicorn
import os

# Initialize the FastAPI app
app = FastAPI(
    title="Medical Product Recommendation API",
    description="API for recommending over-the-counter medical products based on symptoms using LLM and RAG.",
    version="1.0.0"
)

# --- Pydantic Models for Request and Response ---
# These define the expected data structures for your API.
class SymptomInput(BaseModel):
    symptoms: str

class ProductDetail(BaseModel):
    product_name: str
    use_for: str
    side_effects: str
    relevance_score: float

class RecommendationOutput(BaseModel):
    recommendation: str
    confidence_score: float
    retrieved_products: list[ProductDetail] # List of ProductDetail objects

# --- Global Initialization ---
# This variable will hold our MedicalRecommendationSystem instance.
# It's initialized once when the application starts up.
medical_system: MedicalRecommendationSystem = None

@app.on_event("startup")
async def startup_event():
    """
    Event handler that runs when the FastAPI application starts up.
    Initializes the MedicalRecommendationSystem (loading LLM and building FAISS index).
    This ensures the heavy models are loaded only once.
    """
    global medical_system
    print("Starting up: Initializing MedicalRecommendationSystem...")
    try:
        # Pass the path to your medical_products.json (should be in the same directory)
        medical_system = MedicalRecommendationSystem(data_path="medical_products.json")
        print("MedicalRecommendationSystem initialized successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize MedicalRecommendationSystem: {e}")
        # In a production environment, you might want to log this extensively
        # and perhaps exit the application if it cannot function without the AI system.
        # For this test, we'll raise an HTTPException which will prevent server from fully starting.
        raise HTTPException(status_code=500, detail=f"Failed to initialize AI system: {e}. Check server logs for details.")

# --- API Endpoints ---

# Mount a static files directory (for your HTML, CSS, JS)
# This serves files from the current directory under the /static/ path.
# We're making index.html available directly from the root though.
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serves the index.html file as the main frontend for the application.
    """
    # Ensure index.html is in the same directory as app.py
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found. Make sure it's in the root directory.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving frontend: {e}")


@app.post("/recommend", response_model=RecommendationOutput)
async def recommend_product(input: SymptomInput):
    """
    Endpoint to get medical product recommendations based on user symptoms.
    - **symptoms**: A string describing the user's current symptoms.
    Returns:
    - **recommendation**: The AI-generated recommendation text.
    - **confidence_score**: A numerical score indicating the system's confidence in the recommendation (0.0-1.0).
    - **retrieved_products**: A list of detailed information for products identified as relevant by the RAG system.
    """
    if medical_system is None:
        raise HTTPException(status_code=503, detail="AI system not yet initialized. Please wait a moment or check server logs.")

    try:
        # Call the generate_recommendation method from your MedicalRecommendationSystem
        recommendation_result = medical_system.generate_recommendation(input.symptoms)
        
        # Convert the list of dicts from medical_system to a list of Pydantic ProductDetail models
        product_details = []
        for p in recommendation_result.get('retrieved_products', []):
            try:
                product_details.append(
                    ProductDetail(
                        product_name=p.get('product_name', 'N/A'),
                        use_for=p.get('use_for', 'N/A'),
                        side_effects=p.get('side_effects', 'N/A'),
                        relevance_score=p.get('relevance_score', 0.0)
                    )
                )
            except Exception as product_parse_error:
                print(f"Warning: Could not parse retrieved product details: {p} - {product_parse_error}")
                # Optionally, skip this product or include a placeholder

        return RecommendationOutput(
            recommendation=recommendation_result['recommendation'],
            confidence_score=recommendation_result['confidence_score'],
            retrieved_products=product_details
        )
    except Exception as e:
        print(f"Error during recommendation generation for symptoms '{input.symptoms}': {e}")
        # Return a generic 500 error to the client, but log the specific error on the server
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during recommendation. Error: {e}")

# --- Main entry point to run the FastAPI app ---
if __name__ == "__main__":
    # To run the FastAPI app:
    #   Open your terminal, navigate to the project directory, and run:
    #   uvicorn app:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=10000)