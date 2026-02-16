import requests
import json
import time

# Wait for server to start (model loading can take 30-60 seconds)
print("Waiting for server to start (this may take 30-60 seconds for model loading)...")
max_retries = 12
retry_delay = 5

for i in range(max_retries):
    try:
        # Try health check first
        health_response = requests.get("http://127.0.0.1:8000/", timeout=2)
        if health_response.status_code == 200:
            print(f"✓ Server is running! (attempt {i+1})")
            break
    except:
        if i < max_retries - 1:
            print(f"Waiting... ({i+1}/{max_retries})")
            time.sleep(retry_delay)
        else:
            print("ERROR: Server did not start after waiting. Please check for errors.")
            exit(1)

# Test the chatbot
url = "http://127.0.0.1:8000/chat"
payload = {
    "message": "hello",
    "session_id": "test123"
}

try:
    print("\n=== Testing Chatbot ===")
    print(f"Sending request to: {url}")
    print(f"Message: {payload['message']}")
    print()
    
    response = requests.post(url, json=payload, timeout=30)
    
    print(f"Status Code: {response.status_code}")
    print("\n" + "="*50)
    print("CHATBOT RESPONSE")
    print("="*50)
    result = response.json()
    print(f"Reply: {result.get('reply', 'N/A')}")
    print(f"Sources: {result.get('sources', [])}")
    print(f"Website URL: {result.get('website_url', 'N/A')}")
    print("\n" + "="*50)
    print("\nFull JSON Response:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
except requests.exceptions.ConnectionError:
    print("ERROR: Server is not running. Please start the server first.")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

