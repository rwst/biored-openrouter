import requests
import os

# usage: export OPENROUTER_API_KEY="your_actual_key_here"
API_KEY = os.getenv("OPENROUTER_API_KEY")

def get_openrouter_models():
    if not API_KEY:
        print("Error: OPENROUTER_API_KEY environment variable not set.")
        return

    url = "https://openrouter.ai/api/v1/models"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Raise an error for bad status codes
        
        data = response.json()
        models = data.get("data", [])

        print(f"Found {len(models)} models:\n")
        for model in models:
            # 'id' is the string used to call the model (e.g., 'openai/gpt-4')
            # 'name' is the readable display name
            print(f"ID: {model['id']}")
            print(f"Name: {model['name']}")
            print("-" * 30)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    get_openrouter_models()
