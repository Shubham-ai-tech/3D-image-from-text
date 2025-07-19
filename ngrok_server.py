import os
import time
from pyngrok import ngrok, conf

LOCAL_SERVER_PORT = 8000



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

    exit(1)



print(f"\nAttempting to establish ngrok tunnel to http://localhost:{LOCAL_SERVER_PORT}...")
try:

    public_url = ngrok.connect(LOCAL_SERVER_PORT)
    print(f"\n✨ Ngrok Tunnel established! ✨")
    print(f"Your local server is now publicly accessible at:")
    print(f"➡️  {public_url}")
    print("\nKeep this Colab cell running to maintain the tunnel.")
    print("To stop the tunnel, interrupt this cell (e.g., click the stop button in Colab).")


    while True:
        time.sleep(10)

except Exception as e:
    print(f"\n❌ Failed to establish ngrok tunnel: {e}")
    print(f"Please ensure your server is running on port {LOCAL_SERVER_PORT} in a separate Colab cell or process.")
    print("Also, double-check your ngrok authentication token and internet connection.")
finally:

    print("\nDisconnecting ngrok tunnels...")
    ngrok.disconnect(public_url)

    print("Ngrok tunnels disconnected.")

