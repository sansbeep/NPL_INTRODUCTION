import requests
from config import HF_API_KEY
MODEL_ID = "facebook/bart-large-mnli"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
TOPICS = ["Sports", "Technology", "Business", "Politics", "Health"]
def ask_hf(headline):
    payload = {"inputs": headline, "parameters": {"candidate_labels": TOPICS}}
    r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
    if not r.ok:
        return []
    return r.json()
def best_topic(preds):
    best = max(preds, key=lambda x: x["score"])
    return best["label"], best["score"]
def bar(score):
    pct = score * 100
    blocks = int(pct // 10)
    return "#" * blocks + "-" * (10 - blocks)
def show(headline, preds):
    top_label, top_score = best_topic(preds)
    print("News Topic Classifier")
    print("Headline:", headline)
    print("Best topic:", top_label)
    print("Confidence:", round(top_score*100,1), "%", bar(top_score))
    print("Top 3:")
    top3 = sorted(preds, key=lambda x: x["score"], reverse=True)[:3]
    for i, p in enumerate(top3, 1):
        print(i, p['label'], round(p['score']*100,1), "%", bar(p['score']))
def main():
    print("Type a news headline. Type exit to stop.")
    while True:
        headline = input("Headline: ").strip()
        if headline.lower() == "exit":
            break
        if not headline:
            continue
        try:
            preds = ask_hf(headline)
            if isinstance(preds, list) and preds and "label" in preds[0]:
                show(headline, preds)
            else:
                print("Unexpected reply.")
        except:
            print("Error.")
if __name__ == "__main__":
    main()