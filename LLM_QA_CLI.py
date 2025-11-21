import google.generativeai as genai
import re
import os

# --- CONFIGURATION ---
def configure_genai():
    """Configures the AI model using the GEMINI_API_KEY environment variable."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found. Set it in your environment variables!")
        return None
    
    genai.configure(api_key=api_key)
    # Using the standard Gemini Flash model for speed
    model = genai.GenerativeModel('gemini-2.5-flash')
    return model

def preprocess_input(text):
    """
    Performs basic NLP preprocessing:
    1. Lowercasing
    2. Removing punctuation
    3. Basic Tokenization (splitting by whitespace)
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return text, tokens

def get_ai_response(model, prompt):
    """Sends the processed prompt to the LLM."""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"API Error: {str(e)}"

def main():
    print("="*50)
    print("NLP Q&A SYSTEM - CLI MODE")
    print("="*50)
    
    model = configure_genai()
    if not model:
        return

    while True:
        user_input = input("\nAsk a question (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting system...")
            break
        
        processed_text, tokens = preprocess_input(user_input)
        print(f"\n[Debug] Preprocessed: {processed_text}")
        print(f"[Debug] Tokens: {tokens}")
        print("-" * 30)
        print("Thinking...")

        answer = get_ai_response(model, user_input)
        print("\n>> AI Answer:")
        print(answer)
        print("="*50)

if __name__ == "__main__":
    main()
