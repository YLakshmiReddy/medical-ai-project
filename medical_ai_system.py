# D:\medical_ai_project\medical_ai_system.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from rag_system import MedicalProductRAG # Import your RAG system
import torch

class MedicalRecommendationSystem:
    def __init__(self, llm_model_name="google/gemma-2b", data_path="medical_products.json"): # CHANGED LLM MODEL NAME
        """
        Initializes the medical recommendation system with an LLM and RAG.
        Args:
            llm_model_name (str): Hugging Face model ID for the LLM.
            data_path (str): Path to the medical product JSON data.
        """
        print(f"Initializing LLM: {llm_model_name}...")
        
        # Determine device (GPU if available, else CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        # Ensure pad_token is set for generation. Many models use eos_token as pad_token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the LLM model
        try:
            # Use torch.float16 for GPU to save VRAM. Use float32 for CPU.
            # device_map="auto" attempts to load model parts across available GPUs or CPU.
            # REMOVED load_in_8bit=True here as Gemma-2B is smaller
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" # This helps with memory management on GPUs
            )
            # If device_map="auto" fails or if on CPU, ensure model is on the correct device
            if self.device == "cpu" and self.model.device.type != "cpu":
                 self.model.to(self.device) # Explicitly move to CPU if it landed elsewhere
            print(f"LLM '{llm_model_name}' loaded successfully on {self.device}.")
        except Exception as e:
            print(f"Error loading LLM '{llm_model_name}' on {self.device}: {e}")
            print("Falling back to CPU only loading without auto-mapping...")
            try:
                # REMOVED load_in_8bit=True here (fallback)
                self.model = AutoModelForCausalLM.from_pretrained(
                    llm_model_name,
                    torch_dtype=torch.float32, # Always use float32 for CPU
                    device_map="cpu" # Force CPU
                )
                self.device = "cpu" # Confirm device is CPU
                print(f"LLM '{llm_model_name}' loaded successfully on CPU as fallback.")
            except Exception as e_cpu:
                print(f"Failed to load LLM on CPU as well: {e_cpu}")
                raise RuntimeError(f"Could not load the LLM model {llm_model_name}. "
                                   "Check your internet connection, model name, and system memory/GPU.")

        # Initialize the RAG system
        self.rag_system = MedicalProductRAG(data_path=data_path)

    def generate_recommendation(self, user_symptoms, k=3):
        """
        Generates a medical product recommendation based on user symptoms using RAG and LLM.
        Args:
            user_symptoms (str): User's input symptoms.
            k (int): Number of top products to retrieve from RAG.
        Returns:
            dict: A dictionary containing the recommendation text, confidence score, and retrieved products.
        """
        print(f"\nProcessing user query: '{user_symptoms}'")

        # 1. Retrieve relevant products using RAG
        retrieved_products = self.rag_system.retrieve_products(user_symptoms, k=k)

        if not retrieved_products:
            print("No relevant products found by RAG system. Returning default message.")
            return {
                "recommendation": "I'm sorry, I couldn't find any relevant over-the-counter products for your symptoms in my database. Please consult a medical professional if your symptoms persist or worsen.",
                "confidence_score": 0.0,
                "retrieved_products": []
            }

        # 2. Prepare context for the LLM -- NEW, SIMPLIFIED FORMAT FOR LLM
        context_parts = []
        for i, product in enumerate(retrieved_products):
            context_parts.append(
                f"PRODUCT NAME: {product.get('product_name', 'N/A')}\n"
                f"USE CASE: {product.get('use_for', 'N/A')}\n"
                f"SIDE EFFECTS: {product.get('side_effects', 'No known side effects')}\n"
            )
        context = "\n---\n".join(context_parts) # Separate products with --- for clarity


        # 3. Create the prompt for the LLM -- NEW, MORE DIRECTIVE PROMPT
        prompt = (
            f"You are a helpful AI assistant specialized in recommending over-the-counter medical products. "
            f"Your task is to provide a concise and direct recommendation. "
            f"You MUST only use information from the 'Available Products (Context)' section below. "
            f"Do NOT add any external knowledge, disclaimers, or conversational filler. "
            f"Just the direct recommendation(s) in a numbered list.\n\n"
            f"--- Available Products (Context) ---\n"
            f"{context}\n\n"
            f"--- User Symptoms ---\n"
            f"Symptoms: '{user_symptoms}'\n\n"
            f"--- Recommendation ---\n"
            f"Based on the user's symptoms and the provided product information, recommend the most appropriate "
            f"over-the-counter medical product(s). For each recommended product, clearly state its name, its primary use case, and its potential side effects."
            f"Example: 1. Product Name: [Name], Use: [Use Case], Side Effects: [Side Effects]\n"
            f"Recommendation:"
        )

        # 4. Generate response using the LLM -- UPDATED GENERATION PARAMETERS
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(self.device)

        output_tokens = self.model.generate(
            **inputs,
            max_new_tokens=150,     # Drastically reduced to prevent rambling
            num_beams=1,            # Use greedy search (deterministic)
            do_sample=False,        # Do not sample (deterministic)
            temperature=0.7,        # Will be ignored, but good practice to keep
            top_k=50,               # Will be ignored, but good practice to keep
            no_repeat_ngram_size=2, # Helps avoid repetitive phrases
            early_stopping=True,    # Will be ignored, but good practice to keep
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id, # Explicitly tell model what end-of-sentence token is
        )

        generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # Post-process: Extract only the LLM's recommendation part -- REFINED CLEANUP
        llm_recommendation = generated_text.strip()
        
        # Look for the last occurrence of "Recommendation:" and extract everything after it.
        start_marker = "\nRecommendation:"
        if start_marker in llm_recommendation:
            llm_recommendation = llm_recommendation.rpartition(start_marker)[2].strip()
        
        # Aggressive cleanup for common Phi-2 repetitions/runaways
        if "--- Available Products (Context) ---" in llm_recommendation:
            llm_recommendation = llm_recommendation.split("--- Available Products (Context) ---")[0].strip()
        if "--- User Symptoms ---" in llm_recommendation:
            llm_recommendation = llm_recommendation.split("--- User Symptoms ---")[0].strip()
        if "--- Recommendation ---" in llm_recommendation: # Catch partial repetitions of this marker
            llm_recommendation = llm_recommendation.split("--- Recommendation ---")[0].strip()
        
        # Fallback message (if generation failed to produce meaningful output)
        if not llm_recommendation or len(llm_recommendation.split()) < 5: 
            llm_recommendation = "I'm sorry, I couldn't generate a specific recommendation based on the provided information. Please consult a medical professional if your symptoms persist or worsen."


        # Calculate a simple confidence score (still based on retrieved products)
        confidence_score = 0.0
        if retrieved_products:
            # Take the relevance score of the top retrieved product as a proxy for confidence
            confidence_score = retrieved_products[0].get('relevance_score', 0.0)

        print(f"LLM Recommendation Generated. Confidence: {confidence_score:.2f}")

        return {
            "recommendation": llm_recommendation,
            "confidence_score": confidence_score,
            "retrieved_products": retrieved_products # Include retrieved products for debugging/info
        }

if __name__ == "__main__":
    # This block runs only when medical_ai_system.py is executed directly
    print("--- Testing MedicalRecommendationSystem ---")
    
    # Initialize the full system. This will load the LLM and build the RAG index.
    # This step can take a while on the first run as the LLM is downloaded.
    system = MedicalRecommendationSystem()

    test_cases = [
        "I have a fever and body pain.",
        "I'm sneezing a lot and have a runny nose.",
        "I feel chest congestion and sore throat.",
        "My stomach hurts and I have heartburn.",
        "I have a minor cut on my finger and it's bleeding a little.",
        "I feel very tired and dehydrated because of diarrhea.",
        "My skin is itchy and red from an insect bite.",
        "I have a dry cough and my throat is scratchy.",
        "My nose is completely blocked and I can't breathe."
    ]

    for symptoms in test_cases:
        result = system.generate_recommendation(symptoms)
        print(f"\n--- Symptoms: {symptoms} ---")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Confidence Score: {result['confidence_score']:.2f}")
        print("\nRetrieved Products (for reference):")
        if result['retrieved_products']:
            for p in result['retrieved_products']:
                print(f"  - {p.get('product_name', 'N/A')} (Score: {p.get('relevance_score', 0.0):.2f})")
                print(f"    Use: {p.get('use_for', 'N/A')}")
                print(f"    Side Effects: {p.get('side_effects', 'N/A')}")
        else:
            print("  No products were retrieved for this query.")
        print("-" * 80) # Separator for readability