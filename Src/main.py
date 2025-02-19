import subprocess

if __name__ == "__main__":
    try:
        print("Starting Flask app...")
        subprocess.run(["python", "app.py"])
        subprocess.run(["python", "app2.py"])
    except KeyboardInterrupt:
        print("\nFlask app stopped.")
