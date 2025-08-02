# D:\medical_ai_project\medical_ai_system.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from rag_system import MedicalProductRAG # Import your RAG system
import torch

class MedicalRecommendationSystem:
    # Using TinyLlama as it's accessible without authentication
    def __init__(self, llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", data_path="medical_products.json"):
        """
        Initializes the medical recommendation system with an LLM and RAG.
        Args:
            llm_model_name (str): Hugging Face model ID for the LLM.
            data_path (str): Path to the medical product JSON data.
        """
        print(f"Initializing LLM: {llm_model_name}...")
        
        # Check if CUDA (GPU) is available, otherwise use CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the LLM model
        try:
            # Use float16 for CUDA to save VRAM, float32 for CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" # Let accelerate manage device mapping
            )
            # Ensure model is on the correct device if device_map="auto" puts it on meta/cpu and device is CPU
            if self.device == "cpu" and self.model.device.type != "cpu":
                 self.model.to(self.device)
            print(f"LLM '{llm_model_name}' loaded successfully on {self.device}.")
        except Exception as e:
            print(f"Error loading LLM '{llm_model_name}' on {self.device}: {e}")
            print("Falling back to CPU only loading without auto-mapping (if applicable) and explicitly float32...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    llm_model_name,
                    torch_dtype=torch.float32, # Always use float32 for CPU fallback
                    device_map="cpu" # Force CPU in fallback
                )
                self.device = "cpu" # Confirm device is CPU for fallback
                print(f"LLM '{llm_model_name}' loaded successfully on CPU as fallback.")
            except Exception as e_cpu:
                print(f"Failed to load LLM on CPU as well: {e_cpu}")
                raise RuntimeError(f"Could not load the LLM model {llm_model_name}. "
                                   "Check your internet connection, model name, and system memory/GPU. "
                                   "Consider a smaller model (e.g., 'TinyLlama/TinyLlama-1.1B-Chat-v1.0') or more RAM if this persists. "
                                   "If running on a free Colab GPU, ensure the runtime is GPU and sometimes restarting runtime helps.")

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

        # 2. Prepare context for the LLM
        context_parts = []
        for i, product in enumerate(retrieved_products):
            # Make context extremely clear and concise for the LLM
            context_parts.append(
                f"Product {i+1}:\n"
                f"NAME: {product.get('product_name', 'N/A')}\n"
                f"USE CASE: {product.get('use_for', 'N/A')}\n"
                f"SIDE EFFECTS: {product.get('side_effects', 'No known side effects')}"
            )
        context = "\n\n".join(context_parts) # Separate products with double newline for clarity


        # 3. Create the prompt for the LLM
        # IMPORTANT: Make the prompt even MORE directive and restrictive
        # Adding roles helps some models understand intent better.
        prompt = (
            f"You are a highly precise medical AI assistant that strictly follows instructions. "
            f"Your task is to recommend over-the-counter medical products based ONLY on the 'AVAILABLE PRODUCTS' below. "
            f"You MUST format your output as a numbered list of recommendations. "
            f"For each recommended product, state its exact 'NAME', 'USE CASE', and 'SIDE EFFECTS' from the provided context. "
            f"Do NOT include any other text, disclaimers, conversational filler, or external information. "
            f"If no product is suitable, state 'No suitable product found.'.\n\n"
            f"--- AVAILABLE PRODUCTS ---\n"
            f"{context}\n\n"
            f"--- USER SYMPTOMS ---\n"
            f"Symptoms: '{user_symptoms}'\n\n"
            f"--- RECOMMENDATION (Numbered List ONLY) ---\n"
        )

        # 4. Generate response using the LLM
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(self.device)

        # Generate output tokens
        output_tokens = self.model.generate(
            **inputs,
            max_new_tokens=150,     # Reduced slightly to try and prevent rambling
            num_beams=1,            # Use greedy search (deterministic)
            do_sample=False,        # Do not sample (deterministic)
            temperature=0.0,        # Set temperature to 0.0 for maximum determinism
            top_k=0,                # Set top_k to 0 (ignored for greedy search but explicit)
            top_p=1.0,              # Set top_p to 1.0 (ignored for greedy search but explicit)
            no_repeat_ngram_size=2, # Helps avoid repetitive phrases
            early_stopping=True,    # Stop when EOS token is generated
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode the generated text and extract ONLY the part after the last prompt marker
        generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        # Post-process: Extract only the LLM's recommendation part
        # Find the start of the "--- RECOMMENDATION (Numbered List ONLY) ---" section in the *original prompt*,
        # and extract everything after it from the *generated text*.
        start_marker_in_prompt = "--- RECOMMENDATION (Numbered List ONLY) ---\n"
        prompt_length_in_output = generated_text.rfind(start_marker_in_prompt)
        
        if prompt_length_in_output != -1:
            llm_recommendation = generated_text[prompt_length_in_output + len(start_marker_in_prompt):].strip()
        else:
            # Fallback if marker not found, take everything after the last "---" section.
            # This is a less reliable fallback for TinyLlama.
            last_marker_index = generated_text.rfind("---")
            if last_marker_index != -1:
                llm_recommendation = generated_text[last_marker_index + 3:].strip()
            else:
                llm_recommendation = generated_text.strip() # As a last resort, take all text

        # Final check for emptiness or non-compliance (e.g., if LLM still rambles)
        if not llm_recommendation or len(llm_recommendation.split()) < 5 or \
           "Note:" in llm_recommendation or "Please consult" in llm_recommendation or \
           "I'm sorry" in llm_recommendation:
            llm_recommendation = "I'm sorry, I couldn't generate a specific recommendation in the requested format based on the provided information. Please consult a medical professional if your symptoms persist or worsen."
        elif not llm_recommendation.strip().startswith("1.") and not llm_recommendation.strip().startswith("No suitable product found"):
            # Further refinement to enforce numbered list if not starting with "1."
            llm_recommendation = "I was unable to format the recommendation as a numbered list. Here's what I found:\n" + llm_recommendation


        # Calculate a simple confidence score
        confidence_score = 0.0
        if retrieved_products:
            # The relevance score from RAG is already between 0 and 1
            confidence_score = retrieved_products[0].get('relevance_score', 0.0)

        print(f"LLM Recommendation Generated. Confidence: {confidence_score:.2f}")

        return {
            "recommendation": llm_recommendation,
            "confidence_score": confidence_score,
            "retrieved_products": retrieved_products
        }

if __name__ == "__main__":
    print("--- Testing MedicalRecommendationSystem ---")
    
    try:
        system = MedicalRecommendationSystem(llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    except RuntimeError as e:
        print(f"Initialization failed: {e}")
        print("Exiting test. Please adjust 'llm_model_name' or ensure sufficient resources.")
        exit()

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
        print("-" * 80)