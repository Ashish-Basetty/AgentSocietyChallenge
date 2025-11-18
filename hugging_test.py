import requests
from transformers import pipeline

def main():
    try:
        r = requests.get("https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion/resolve/main/config.json", timeout=5)
        print("Status code:", r.status_code)
        emotion_classifier = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-emotion",
            top_k=5,
            device=-1
        )
    except Exception as e:
        print("Error:", e)
    print(emotion_classifier("I am so happy today!"))

if __name__ == "__main__":
    main()

    
