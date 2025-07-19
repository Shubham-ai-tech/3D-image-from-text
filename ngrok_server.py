import os
import time
from pyngrok import ngrok, conf # Import conf for setting ngrok path if needed

# --- Configuration ---
# The local port your server is running on (e.g., from simple_http_server)
LOCAL_SERVER_PORT = 8000

# --- 1. Authenticate ngrok ---
# It's highly recommended to use Colab Secrets for your ngrok auth token.
# 1. Go to ngrok.com and sign up/log in to get your authtoken.
# 2. In Colab, on the left sidebar, click the "key" icon (Secrets).
# 3. Click "Add new secret". Name it "NGROK_AUTH_TOKEN" and paste your token as the value.
# 4. Make sure "Notebook access" is toggled ON for this notebook.

try:
    NGROK_AUTH_TOKEN = "305gRwBfUJbIL1KwcKg6hXrgAF8_7dijtF8cpjkFojPn4SXrT"
    if not NGROK_AUTH_TOKEN:
        raise ValueError("NGROK_AUTH_TOKEN not found in Colab Secrets.")
    
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    print("ngrok authenticated using Colab Secrets.")
except Exception as e:
    print(f"Error authenticating ngrok: {e}")
    print("Please ensure 'NGROK_AUTH_TOKEN' is set in Colab Secrets and enabled for this notebook.")
    print("Alternatively, for testing, you can paste your token directly (less secure):")
    print('ngrok.set_auth_token("YOUR_AUTH_TOKEN_HERE")')
    # Exit if authentication fails, as tunneling won't work
    exit(1)

# --- 2. Download ngrok binary (pyngrok handles this automatically) ---
# pyngrok will automatically download the ngrok binary if it's not found.
# You can optionally specify a custom path if you downloaded it manually:
# conf.get_default().ngrok_path = "/path/to/your/ngrok/executable"

# --- 3. Establish the ngrok tunnel ---
print(f"\nAttempting to establish ngrok tunnel to http://localhost:{LOCAL_SERVER_PORT}...")
try:
    # Connect ngrok to your local server port
    public_url = ngrok.connect(LOCAL_SERVER_PORT)
    print(f"\n✨ Ngrok Tunnel established! ✨")
    print(f"Your local server is now publicly accessible at:")
    print(f"➡️  {public_url}")
    print("\nKeep this Colab cell running to maintain the tunnel.")
    print("To stop the tunnel, interrupt this cell (e.g., click the stop button in Colab).")

    # Keep the Python script running indefinitely to maintain the tunnel
    while True:
        time.sleep(10) # Sleep for a bit to prevent excessive CPU usage

except Exception as e:
    print(f"\n❌ Failed to establish ngrok tunnel: {e}")
    print(f"Please ensure your server is running on port {LOCAL_SERVER_PORT} in a separate Colab cell or process.")
    print("Also, double-check your ngrok authentication token and internet connection.")
finally:
    # Disconnect ngrok tunnels when the script exits
    print("\nDisconnecting ngrok tunnels...")
    ngrok.disconnect(public_url) # Disconnect the specific tunnel
    # ngrok.kill() # Kills all ngrok processes (use if multiple tunnels or issues)
    print("Ngrok tunnels disconnected.")

