# lead_scout.py (Google Places + Sentiment)
import os
import time
import re
import argparse
import requests
import pandas as pd
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

analyzer = SentimentIntensityAnalyzer()
EMAIL_REGEX = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')

# ---------- Helpers ----------
def safe_get(url, headers=None, timeout=6):
    try:
        r = requests.get(url, headers=headers or {"User-Agent": "LeadScoutBot/1.0"}, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception:
        return None


# ---------- Google Places API ----------
def google_places_search(query, location, limit=20, pagetoken=None):
    endpoint = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{query} in {location}",
        "key": GOOGLE_API_KEY,
    }
    if pagetoken:
        params["pagetoken"] = pagetoken

    resp = requests.get(endpoint, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])
    next_page = data.get("next_page_token")
    return results, next_page


def google_place_details(place_id):
    endpoint = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,website,formatted_address,rating,user_ratings_total,reviews",
        "key": GOOGLE_API_KEY,
    }
    resp = requests.get(endpoint, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json().get("result", {})


# ---------- Website checks ----------
def extract_website_info(url):
    text = safe_get(url)
    if text is None:
        return {"reachable": False, "has_meta_desc": False, "contact_email": None, "has_ldjson": False}

    soup = BeautifulSoup(text, "html.parser")
    meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    has_meta = bool(meta and meta.get("content"))
    emails = EMAIL_REGEX.findall(text)
    contact_email = emails[0] if emails else None
    has_ldjson = bool(soup.find("script", attrs={"type": "application/ld+json"}))
    return {"reachable": True, "has_meta_desc": has_meta, "contact_email": contact_email, "has_ldjson": has_ldjson}


# ---------- Scoring ----------
def compute_ai_lead_score(features):
    f_has_website = 1 if features.get("has_website") else 0
    rating = features.get("rating") or 5.0
    f_rating = 1 - (rating / 5.0)
    review_count = features.get("review_count") or 0
    f_review_count = 1 - min(review_count, 200) / 200

    # Sentiment: convert compound (-1..1) to [0..1] where higher => more negative
    compound = features.get("avg_review_sentiment")
    f_sent = 1 - ((compound + 1) / 2) if compound is not None else 0

    f_missing_meta = 1 if (features.get("has_meta_desc") is False) else 0

    w = {"missing_website": 0.25, "rating": 0.20, "review_count": 0.15, "sentiment": 0.25, "missing_meta": 0.15}
    score_raw = (
        w["missing_website"] * (1 - f_has_website) +
        w["rating"] * f_rating +
        w["review_count"] * f_review_count +
        w["sentiment"] * f_sent +
        w["missing_meta"] * f_missing_meta
    )

    score = int(round(score_raw * 100))
    reasons = []
    if not f_has_website:
        reasons.append("No website found")
    if f_rating > 0.3:
        reasons.append(f"Low rating: {rating}")
    if review_count < 20:
        reasons.append(f"Low review count: {review_count}")
    if f_sent > 0.25:
        reasons.append("Negative review sentiment")
    if f_missing_meta:
        reasons.append("Missing meta description")

    return max(0, min(100, score)), reasons


# ---------- Main Pipeline ----------
def scout_leads(category, location, target_n=50, sleep_between_requests=1.5):
    all_biz = []
    next_page = None

    while len(all_biz) < target_n:
        batch, next_page = google_places_search(category, location, limit=min(20, target_n - len(all_biz)), pagetoken=next_page)
        if not batch:
            break
        all_biz.extend(batch)
        if not next_page:
            break
        time.sleep(sleep_between_requests)

    print(f"Found {len(all_biz)} candidate businesses from Google Places")

    rows = []
    for biz in all_biz:
        name = biz.get("name")
        place_id = biz.get("place_id")
        rating = biz.get("rating")
        review_count = biz.get("user_ratings_total", 0)

        # Fetch details including reviews
        try:
            details = google_place_details(place_id)
        except Exception:
            details = {}

        website = details.get("website")
        website_info = extract_website_info(website) if website else {"reachable": False, "has_meta_desc": False, "contact_email": None, "has_ldjson": False}

        # Sentiment analysis of reviews (snippets)
        reviews = details.get("reviews", [])
        compound_scores = []
        for rv in reviews:
            text = (rv.get("text") or "")[:1000]
            compound_scores.append(analyzer.polarity_scores(text)["compound"])
        avg_compound = sum(compound_scores)/len(compound_scores) if compound_scores else None

        features = {
            "has_website": bool(website),
            "rating": rating,
            "review_count": review_count,
            "avg_review_sentiment": avg_compound,
            "has_meta_desc": website_info.get("has_meta_desc")
        }

        score, reasons = compute_ai_lead_score(features)
        rows.append({
            "name": name,
            "place_id": place_id,
            "rating": rating,
            "review_count": review_count,
            "website": website or "",
            "avg_sentiment": avg_compound,
            "ai_lead_score": score,
            "reasons": "; ".join(reasons)
        })
        time.sleep(sleep_between_requests)

    df = pd.DataFrame(rows).sort_values("ai_lead_score", ascending=False)
    return df


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True, help="Business category, e.g., 'gym'")
    parser.add_argument("--location", type=str, required=True, help="Location, e.g., 'San Diego, CA'")
    parser.add_argument("--num", type=int, default=50, help="Target number of leads")
    args = parser.parse_args()

    if not GOOGLE_API_KEY:
        print("ERROR: Set GOOGLE_API_KEY in .env before running")
        return

    df = scout_leads(args.category, args.location, target_n=args.num)
    print(df.head(25).to_string(index=False))
    out_csv = f"scout_{args.category}_{args.location.replace(',', '').replace(' ', '_')}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")


if __name__ == "__main__":
    main()
