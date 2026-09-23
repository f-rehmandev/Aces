import logging
import requests

logger = logging.getLogger("osm_search")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"


def search_businesses(keyword: str, location: str, max_results: int = 20) -> list[dict]:
    """
    Free, no-API-key business search via OpenStreetMap's Overpass API.
    Returns real businesses matching `keyword` within `location`, with
    name, address, phone, website (or None if absent — the lead-qualifying signal), email.
    """
    query = f"""
    [out:json][timeout:25];
    area["name"="{location}"]->.searchArea;
    (
      node["name"~"{keyword}",i](area.searchArea);
      node["cuisine"~"{keyword}",i](area.searchArea);
      node["shop"~"{keyword}",i](area.searchArea);
    );
    out body {max_results};
    """

    logger.info(f"Querying OpenStreetMap for '{keyword}' in '{location}'")
    headers = {
        "User-Agent": "ACES-Project/1.0 (student portfolio project; contact: your_email@example.com)",
        "Referer": "https://github.com/your-username/aces",
        "Accept": "*/*",
        "Content-Type": "text/plain",
    }
    
    response = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=30)
    if response.status_code != 200:
        print("OVERPASS ERROR RESPONSE:", response.text[:1000])
    response.raise_for_status()
    data = response.json()

    results = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue

        address_parts = [
            tags.get("addr:housenumber", ""),
            tags.get("addr:street", ""),
            tags.get("addr:city", ""),
        ]
        address = " ".join(p for p in address_parts if p).strip() or None

        results.append({
            "business_name": name,
            "phone": tags.get("phone") or tags.get("contact:phone"),
            "address": address,
            "email": tags.get("email") or tags.get("contact:email"),
            "website_url": tags.get("website") or tags.get("contact:website"),
            "has_website": bool(tags.get("website") or tags.get("contact:website")),
        })

    logger.info(f"Found {len(results)} businesses")
    return results


if __name__ == "__main__":
    results = search_businesses(keyword="pizza", location="Lahore", max_results=15)
    for r in results:
        print(r)