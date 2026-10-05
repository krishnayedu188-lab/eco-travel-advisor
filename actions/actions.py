"""
Eco-Travel Advisor - custom actions (Step 3)
- Form validation (cleans user answers)
- Quick-reply buttons generated in Python
- Carbon calculation: Climatiq API with an offline fallback
"""
import json
import difflib
import logging
import math
import os
import re
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Text

import requests
from dotenv import load_dotenv
from rasa_sdk import Action, FormValidationAction, Tracker
from rasa_sdk.events import ActiveLoop, FollowupAction, SlotSet
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict

load_dotenv()  # reads CLIMATIQ_API_KEY from .env - keys are never written in code
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
CLIMATIQ_URL = "https://api.climatiq.io/data/v1/estimate"
CLIMATIQ_DATA_VERSION = "^21"

# Emission factors found with the Climatiq search endpoint (available on the free plan).
# The distance-based Travel endpoint needs a paid plan, so we send our own distance.
CLIMATIQ_FACTORS = {
    "rail": {"activity_id": "passenger_train-route_type_na-fuel_source_na",
             "region": "DE", "year": 2020, "per_passenger": True},
    "bus": {"activity_id": "passenger_vehicle-vehicle_type_coach-fuel_source_diesel-engine_size_na-vehicle_age_na-vehicle_weight_na",
            "region": "NL", "year": 2023, "per_passenger": True},
    "car": {"activity_id": "passenger_vehicle-vehicle_type_car-fuel_source_na-engine_size_na-vehicle_age_na-vehicle_weight_na",
            "region": "US", "year": 2024, "per_passenger": False},
    "air": {"activity_id": "passenger_flight-route_type_domestic-aircraft_type_na-distance_na-class_na-rf_included-distance_uplift_included",
            "region": "GB", "year": 2025, "per_passenger": True},
}
CLIMATIQ_KEY = os.getenv("CLIMATIQ_API_KEY", "").strip()
API_TIMEOUT = 2.5  # seconds - supports the "< 3 seconds" latency requirement

FEATURED_DESTINATIONS = ["Amsterdam", "Vienna", "Copenhagen", "Lisbon"]

# Approximate city coordinates, used only for the offline fallback distance
CITY_COORDS = {
    "amsterdam": (52.3676, 4.9041),
    "barcelona": (41.3874, 2.1686),
    "berlin": (52.5200, 13.4050),
    "cologne": (50.9375, 6.9603),
    "copenhagen": (55.6761, 12.5683),
    "hamburg": (53.5511, 9.9937),
    "lisbon": (38.7223, -9.1393),
    "london": (51.5072, -0.1276),
    "munich": (48.1351, 11.5820),
    "paris": (48.8566, 2.3522),
    "prague": (50.0755, 14.4378),
    "rome": (41.9028, 12.4964),
    "vienna": (48.2082, 16.3738),
}

# Fallback emission factors in kg CO2e per passenger-km.
# Approximate values based on the UK Government (DESNZ/DEFRA) GHG conversion
# factors - check the latest table and cite it in the report.
FALLBACK_FACTORS = {
    "rail": 0.035,
    "bus": 0.027,
    "car": 0.166,   # average car, one person
    "air": 0.153,   # short-haul economy incl. radiative forcing
}

MODE_LABELS = {
    "rail": "🚆 Train",
    "bus": "🚌 Coach",
    "car": "🚗 Car (alone)",
    "air": "✈️ Flight",
}
BAND_EMOJI = {"green": "🟢", "amber": "🟠", "red": "🔴"}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def button(title: Text, intent: Text, entities: Dict[Text, Any]) -> Dict[Text, Text]:
    """Build a quick-reply button whose payload sets an entity directly."""
    return {"title": title, "payload": f"/{intent}{json.dumps(entities)}"}


def clean_city(value: Any) -> Optional[Text]:
    """Remove punctuation/extra spaces and capitalise: ' paris! ' -> 'Paris'."""
    if not value:
        return None
    text = re.sub(r"[^A-Za-zÀ-ÿ\s-]", "", str(value)).strip()
    return text.title() if len(text) >= 2 else None

def correct_city(city: Text) -> Text:
    """Fix small typos using the known city list, e.g. 'Berlinn' -> 'Berlin'."""
    matches = difflib.get_close_matches(city.lower(), CITY_COORDS.keys(), n=1, cutoff=0.8)
    return matches[0].title() if matches else city
# Local / other-language names -> English names used by the bot
CITY_ALIASES = {
    "wien": "vienna", "münchen": "munich", "munchen": "munich",
    "köln": "cologne", "koln": "cologne", "praha": "prague",
    "lisboa": "lisbon", "københavn": "copenhagen", "kobenhavn": "copenhagen",
    "kopenhagen": "copenhagen", "roma": "rome", "londres": "london",
}
SUPPORTED_ORIGINS = ["Berlin", "Hamburg", "Munich", "Cologne", "Paris", "London"]


def find_known_city(value: Any) -> Optional[tuple]:
    """Find a supported city anywhere in the user's text.
    Returns (CityName, was_corrected) or None.
    Works for 'Vienna', 'I'm travelling from Hamburg', 'wien' and typos like 'berlinn'."""
    text = str(value or "").lower()
    for alias, city in CITY_ALIASES.items():
        if re.search(rf"\b{alias}\b", text):
            return city.title(), False
    for city in CITY_COORDS:
        if re.search(rf"\b{city}\b", text):
            return city.title(), False
    for word in re.findall(r"[a-zà-ÿ]+", text):
        match = difflib.get_close_matches(word, CITY_COORDS.keys(), n=1, cutoff=0.8)
        if match:
            return match[0].title(), True
    return None


def previous_slot_value(tracker: Tracker, slot: Text) -> Any:
    """The value a slot had BEFORE the user's latest message."""
    events = tracker.events
    last_user = max((i for i, e in enumerate(events) if e.get("event") == "user"), default=len(events))
    for event in reversed(events[:last_user]):
        if event.get("event") == "slot" and event.get("name") == slot:
            return event.get("value")
    return None

def recently_failed(tracker: Tracker, marker: Text = "travel data") -> bool:
    """True if the bot's reply to the user's PREVIOUS message was a city error."""
    users_seen = 0
    for event in reversed(tracker.events):
        if event.get("event") == "user":
            users_seen += 1
            if users_seen > 1:
                break
        elif users_seen == 1 and event.get("event") == "bot" and marker in (event.get("text") or ""):
            return True
    return False


def haversine_km(a: tuple, b: tuple) -> float:
    """Straight-line distance between two (lat, lon) points in km."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(h))


def carbon_band(co2e_kg: float, distance_km: Optional[float]) -> Text:
    """Green / amber / red, based on grams of CO2e per passenger-km."""
    if distance_km:
        g_per_km = co2e_kg * 1000 / distance_km
        if g_per_km < 60:
            return "green"
        if g_per_km <= 140:
            return "amber"
        return "red"
    # No distance available: use absolute thresholds for a one-way trip
    if co2e_kg < 30:
        return "green"
    if co2e_kg <= 100:
        return "amber"
    return "red"

def route_distance_km(mode: Text, origin: Text, destination: Text) -> Optional[float]:
    """Distance used for the Climatiq request.
    Ground transport: straight line x 1.25 (roads/rails are not straight).
    Flights: straight line only, because the flight factor already includes an uplift."""
    a = CITY_COORDS.get(origin.lower())
    b = CITY_COORDS.get(destination.lower())
    if not a or not b:
        return None
    straight = haversine_km(a, b)
    return straight if mode == "air" else straight * 1.25


def climatiq_estimate(mode: Text, origin: Text, destination: Text) -> Optional[Dict]:
    """Ask Climatiq's Estimate endpoint for emissions. Returns None on any failure."""
    if not CLIMATIQ_KEY:
        return None
    factor = CLIMATIQ_FACTORS.get(mode)
    km = route_distance_km(mode, origin, destination)
    if not factor or not km:
        return None

    parameters: Dict[Text, Any] = {"distance": round(km, 1), "distance_unit": "km"}
    if factor["per_passenger"]:
        parameters["passengers"] = 1

    body = {
        "emission_factor": {
            "activity_id": factor["activity_id"],
            "data_version": CLIMATIQ_DATA_VERSION,
            "region": factor["region"],
            "year": factor["year"],
        },
        "parameters": parameters,
    }
    try:
        response = requests.post(
            CLIMATIQ_URL,
            json=body,
            headers={"Authorization": f"Bearer {CLIMATIQ_KEY}"},
            timeout=API_TIMEOUT,
        )
        if response.status_code != 200:
            logger.warning("Climatiq %s error %s: %s", mode, response.status_code, response.text[:300])
            return None
        data = response.json()
        if data.get("co2e") is None:
            logger.warning("Climatiq response had no co2e: %s", data)
            return None
        ef = data.get("emission_factor", {})
        return {
            "co2e_kg": float(data["co2e"]),
            "distance_km": km,
            "source": f"Climatiq API ({ef.get('source', '?')} {ef.get('year', '')})".strip(),
        }
    except (requests.RequestException, ValueError) as error:
        logger.warning("Climatiq call failed for %s: %s", mode, error)
        return None


def fallback_estimate(mode: Text, origin: Text, destination: Text) -> Optional[Dict]:
    """Offline estimate: straight-line distance x route factor x emission factor."""
    a = CITY_COORDS.get(origin.lower())
    b = CITY_COORDS.get(destination.lower())
    if not a or not b:
        return None
    detour = 1.09 if mode == "air" else 1.25
    km = haversine_km(a, b) * detour
    return {
        "co2e_kg": km * FALLBACK_FACTORS[mode],
        "distance_km": km,
        "source": "offline estimate (UK Government factors)",
    }


def estimate_emissions(mode: Text, origin: Text, destination: Text) -> Optional[Dict]:
    """Try Climatiq first; if it fails for any reason, use the offline estimate."""
    result = climatiq_estimate(mode, origin, destination)
    if result is None:
        result = fallback_estimate(mode, origin, destination)
    return result









# ---------------------------------------------------------------------------
# Quick-reply buttons (generated in Python, not in domain.yml)
# ---------------------------------------------------------------------------
class ActionAskDestination(Action):
    def name(self) -> Text:
        return "action_ask_destination"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
        buttons = [button(city, "inform", {"city": city}) for city in FEATURED_DESTINATIONS]
        dispatcher.utter_message(
            text="Where would you like to go? Pick one or type any city.",
            buttons=buttons,
        )
        return []


class ActionAskBudget(Action):
    def name(self) -> Text:
        return "action_ask_budget"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
        buttons = [
            button("Under €300", "inform", {"budget": "300"}),
            button("€300–800", "inform", {"budget": "800"}),
            button("€800+", "inform", {"budget": "1500"}),
        ]
        dispatcher.utter_message(
            text="What's your total budget in euros? Pick one or type an amount.",
            buttons=buttons,
        )
        return []


class ActionAskSustainabilityLevel(Action):
    def name(self) -> Text:
        return "action_ask_sustainability_level"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
        buttons = [
            button("🌱 High", "inform", {"sustainability_level": "high"}),
            button("⚖️ Medium", "inform", {"sustainability_level": "medium"}),
            button("💶 Low (price first)", "inform", {"sustainability_level": "low"}),
        ]
        # The explanation acts as a "tooltip" for the sustainability metric
        dispatcher.utter_message(
            text=("How important is low-carbon travel to you? 🌱 High = I'll prioritise "
                  "the lowest emissions, even if it costs more or takes longer."),
            buttons=buttons,
        )
        return []


# ---------------------------------------------------------------------------
# Form validation (runs automatically after each answer)
# ---------------------------------------------------------------------------
class ValidateTripForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_trip_form"

    def _validate_city(self, slot: Text, slot_value: Any, dispatcher: CollectingDispatcher,
                       tracker: Tracker, suggestions: List[Text]) -> Dict[Text, Any]:
        """Shared checks for destination and origin."""
        if slot_value is None:
            return {slot: None}
        # The user was answering a DIFFERENT question, but the NLU also tagged a city
        # for this slot (e.g. 'Vienna' typed as the origin). Keep the earlier answer.
        asked = tracker.get_slot("requested_slot")
        if asked not in (None, slot):
            return {slot: previous_slot_value(tracker, slot)}
        found = find_known_city(slot_value)
        if not found:
            # Two-stage error recovery inside the form:
            # 1st failure -> explain + constrained options; 2nd in a row -> also offer a human
            buttons = [button(c, "inform", {"city": c}) for c in suggestions]
            if recently_failed(tracker):
                dispatcher.utter_message(
                    text=("I still don't have travel data for that city. Pick one below, "
                          "or a human advisor can help with other destinations."),
                    buttons=buttons + [{"title": "👤 Talk to a human", "payload": "/request_human"}],
                )
            else:
                dispatcher.utter_message(
                    text=(f"Sorry, I don't have travel data for \"{str(slot_value)[:40]}\" yet. "
                          "I currently cover European city trips. Pick a city below, "
                          "or a human advisor can help with other routes."),
                    buttons=buttons + [{"title": "👤 Talk to a human", "payload": "/request_human"}],
                )
            return {slot: None}
        city, corrected = found
        if corrected:
            dispatcher.utter_message(text=f"I'll assume you meant {city} 🙂")
        return {slot: city}

    def validate_destination(self, slot_value: Any, dispatcher: CollectingDispatcher,
                             tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self._validate_city("destination", slot_value, dispatcher, tracker, FEATURED_DESTINATIONS)

    def validate_origin(self, slot_value: Any, dispatcher: CollectingDispatcher,
                        tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        result = self._validate_city("origin", slot_value, dispatcher, tracker, SUPPORTED_ORIGINS)
        origin = result.get("origin")
        if tracker.get_slot("requested_slot") == "origin":
            destination = previous_slot_value(tracker, "destination") or ""
        else:
            destination = tracker.get_slot("destination") or ""
        if origin and origin.lower() == destination.lower():
            dispatcher.utter_message(text="That's the same as your destination 🙂 Where are you travelling from?")
            return {"origin": None}
        return result

    def validate_budget(self, slot_value: Any, dispatcher: CollectingDispatcher,
                        tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        if slot_value is None:
            return {"budget": None}
        match = re.search(r"\d+", str(slot_value).replace(",", "").replace(".", ""))
        if not match:
            dispatcher.utter_message(text="Please give your budget as a number, e.g. 500.")
            return {"budget": None}
        amount = int(match.group())
        if amount < 50:
            dispatcher.utter_message(text="That budget looks very low for a trip. Could you check the amount?")
            return {"budget": None}
        return {"budget": str(amount)}






# ---------------------------------------------------------------------------
# Carbon calculation
# ---------------------------------------------------------------------------
class ActionCalculateCarbon(Action):
    def name(self) -> Text:
        return "action_calculate_carbon"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
        origin = tracker.get_slot("origin")
        destination = tracker.get_slot("destination")
        if not origin or not destination:
            dispatcher.utter_message(text="I need both a start and a destination to calculate emissions.")
            return []

        modes = ["rail", "bus", "car", "air"]
        # Call all modes at the same time (in parallel) to keep the reply fast
        with ThreadPoolExecutor(max_workers=len(modes)) as pool:
            results = list(pool.map(lambda m: estimate_emissions(m, origin, destination), modes))

        options = []
        for mode, result in zip(modes, results):
            if result is None:
                continue
            options.append({
                "mode": mode,
                "label": MODE_LABELS[mode],
                "co2e_kg": round(result["co2e_kg"], 1),
                "distance_km": round(result["distance_km"]) if result.get("distance_km") else None,
                "band": carbon_band(result["co2e_kg"], result.get("distance_km")),
                "source": result["source"],
            })

        if not options:
            # Error handling: nothing worked, so offer a human instead of failing silently
            dispatcher.utter_message(
                text=(f"Sorry, I couldn't calculate emissions for {origin} → {destination} right now. "
                      "Would you like a human travel advisor to help?"),
                buttons=[{"title": "👤 Talk to a human", "payload": "/request_human"}],
            )
            return [SlotSet("carbon_options", None)]

        options.sort(key=lambda o: o["co2e_kg"])
        lines = [f"{BAND_EMOJI[o['band']]} {o['label']}: ~{o['co2e_kg']:.0f} kg CO₂e" for o in options]
        dispatcher.utter_message(
            text=f"Estimated emissions, {origin} → {destination} (one way, per person):\n" + "\n".join(lines)
        )

        # Alert message for high-emission options (brief requirement)
        best, worst = options[0], options[-1]
        if worst["band"] == "red" and best["co2e_kg"] > 0:
            ratio = worst["co2e_kg"] / best["co2e_kg"]
            dispatcher.utter_message(
                text=f"⚠️ {worst['label']} emits about {ratio:.0f}× more CO₂ than {best['label']} on this route."
            )

        # Transparency: always say where the numbers come from (anti-greenwashing)
        sources = ", ".join(sorted({o["source"] for o in options}))
        dispatcher.utter_message(text=f"ℹ️ These are estimates. Source: {sources}.")

        # Structured data for the web frontend's colour-coded cards (Step 6)
        dispatcher.utter_message(json_message={"type": "carbon_cards", "options": options})

        return [SlotSet("carbon_options", options)]
# ---------------------------------------------------------------------------
# Step 4: transport ranking, real hotels (OpenStreetMap), verified certifications
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CERTIFICATIONS_FILE = os.path.join(DATA_DIR, "verified_certifications.json")
OSM_CACHE_DIR = os.path.join(DATA_DIR, "osm_cache")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 8      # seconds; only the first search per city waits, then it's cached
CACHE_MAX_AGE_DAYS = 7
SEARCH_RADIUS_M = 1200    # about a 15-minute walk from the main station

# Main railway station per city: arriving by train and walking to the hotel
# is the low-carbon choice, so we search for hotels around the station.
MAIN_STATIONS = {
    "amsterdam": ("Amsterdam Centraal", 52.3791, 4.9003),
    "barcelona": ("Barcelona Sants", 41.3790, 2.1400),
    "berlin": ("Berlin Hauptbahnhof", 52.5251, 13.3694),
    "cologne": ("Köln Hauptbahnhof", 50.9430, 6.9589),
    "copenhagen": ("København H", 55.6727, 12.5646),
    "hamburg": ("Hamburg Hauptbahnhof", 53.5530, 10.0069),
    "lisbon": ("Lisboa Santa Apolónia", 38.7139, -9.1225),
    "london": ("London St Pancras", 51.5308, -0.1260),
    "munich": ("München Hauptbahnhof", 48.1402, 11.5600),
    "paris": ("Paris Gare de Lyon", 48.8443, 2.3744),
    "prague": ("Praha hlavní nádraží", 50.0831, 14.4353),
    "rome": ("Roma Termini", 41.9010, 12.5011),
    "vienna": ("Wien Hauptbahnhof", 48.1851, 16.3779),
}

# Rough fare estimates (EUR per km) and speeds - clearly labelled as estimates to the user
TRANSPORT_PRICE_PER_KM = {"rail": 0.12, "bus": 0.07, "car": 0.18, "air": 0.10}
TRANSPORT_FIXED_EUR = {"rail": 0, "bus": 0, "car": 0, "air": 40}  # airport fees etc.
TRANSPORT_SPEED_KMH = {"rail": 100, "bus": 70, "car": 90, "air": 600}
AIRPORT_TIME_H = 2.5  # check-in, security, transfers

# Transport: how much carbon vs price matters, depending on the user's choice
WEIGHTS = {
    "high": {"carbon": 0.7, "price": 0.3},
    "medium": {"carbon": 0.5, "price": 0.5},
    "low": {"carbon": 0.25, "price": 0.75},
}
# Hotels: how much a verified certification vs walking distance matters
HOTEL_WEIGHTS = {
    "high": {"certified": 0.7, "proximity": 0.3},
    "medium": {"certified": 0.5, "proximity": 0.5},
    "low": {"certified": 0.3, "proximity": 0.7},
}

TRIP_SLOTS = ["destination", "origin", "travel_dates", "budget", "sustainability_level",
              "carbon_options", "hotel_options", "recommended_transport"]


def estimate_nights(travel_dates: Any) -> int:
    """'12-15 November' -> 3, 'next weekend' -> 2, anything else -> 3 (stated to the user)."""
    text = str(travel_dates or "").lower()
    days = [int(n) for n in re.findall(r"\b\d{1,2}\b", text)]
    if len(days) >= 2 and 0 < days[1] - days[0] <= 30:
        return days[1] - days[0]
    if "weekend" in text:
        return 2
    return 3


def normalise(values: List[float]) -> List[float]:
    """Scale values to 0-1 (0 = best/lowest, 1 = worst/highest)."""
    low, high = min(values), max(values)
    if high == low:
        return [0.0 for _ in values]
    return [(v - low) / (high - low) for v in values]


def rank_by_score(items: List[Dict], carbon_key: Text, price_key: Text, level: Text) -> List[Dict]:
    """Weighted score 0-100: lower carbon and lower price give a higher score.
    The weights depend on the user's sustainability level."""
    if not items:
        return []
    w = WEIGHTS.get(level, WEIGHTS["medium"])
    carbon_norm = normalise([i[carbon_key] for i in items])
    price_norm = normalise([i[price_key] for i in items])
    for item, c, p in zip(items, carbon_norm, price_norm):
        item["score"] = round(100 * (1 - (w["carbon"] * c + w["price"] * p)))
    return sorted(items, key=lambda i: i["score"], reverse=True)


def read_json_file(path: Text) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def fetch_osm_hotels(city: Text) -> Optional[List[Dict]]:
    """Real hotels near the city's main station, from OpenStreetMap (free, no key).
    Results are cached for 7 days: faster replies and fair use of a free service.
    Returns None if OpenStreetMap is unreachable and there is no cached copy."""
    station = MAIN_STATIONS.get(city.lower())
    if not station:
        return None
    _, lat, lon = station
    os.makedirs(OSM_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(OSM_CACHE_DIR, f"{city.lower()}.json")

    if os.path.exists(cache_file):
        age_days = (time.time() - os.path.getmtime(cache_file)) / 86400
        cached = read_json_file(cache_file)
        if cached is not None and age_days < CACHE_MAX_AGE_DAYS:
            return cached

    query = f"""
    [out:json][timeout:10];
    (
      node["tourism"="hotel"](around:{SEARCH_RADIUS_M},{lat},{lon});
      way["tourism"="hotel"](around:{SEARCH_RADIUS_M},{lat},{lon});
    );
    out center 60;
    """
    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers={"User-Agent": "EcoTravelAdvisor/1.0 (MSc student project)"},
            timeout=OVERPASS_TIMEOUT,
        )
        response.raise_for_status()
        elements = response.json().get("elements", [])
    except (requests.RequestException, ValueError) as error:
        logger.warning("OpenStreetMap request failed for %s: %s", city, error)
        return read_json_file(cache_file)  # an old copy is better than nothing

    hotels, seen = [], set()
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        h_lat = el.get("lat", el.get("center", {}).get("lat"))
        h_lon = el.get("lon", el.get("center", {}).get("lon"))
        if not name or h_lat is None or h_lon is None or name.lower() in seen:
            continue
        seen.add(name.lower())
        hotels.append({
            "name": name,
            "distance_m": round(haversine_km((lat, lon), (h_lat, h_lon)) * 1000),
            "stars": tags.get("stars"),
            "website": tags.get("website") or tags.get("contact:website"),
            "wheelchair": tags.get("wheelchair"),
            "osm_id": f"{el.get('type')}/{el.get('id')}",
        })
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(hotels, f, ensure_ascii=False, indent=1)
    except OSError as error:
        logger.warning("Could not write OSM cache: %s", error)
    return hotels


def load_verified_certifications(city: Text) -> List[Dict]:
    """Certifications the developer checked manually on official registries."""
    data = read_json_file(CERTIFICATIONS_FILE) or {}
    return [c for c in data.get("certifications", []) if c.get("city", "").lower() == city.lower()]


def simplify_name(name: Text) -> Text:
    """'Hôtel Danube-Garden' -> 'danube garden' (for matching names)."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\bhotel\b", " ", text)
    return " ".join(text.split())


def match_certification(hotel_name: Text, certifications: List[Dict]) -> Optional[Dict]:
    target = simplify_name(hotel_name)
    for cert in certifications:
        if difflib.SequenceMatcher(None, target, simplify_name(cert["hotel_name"])).ratio() >= 0.85:
            return cert
    return None


def rank_hotels(hotels: List[Dict], level: Text) -> List[Dict]:
    """Score 0-100 from verified certification and walking distance to the station."""
    w = HOTEL_WEIGHTS.get(level, HOTEL_WEIGHTS["medium"])
    for h in hotels:
        certified = 1.0 if h.get("certification") else 0.0
        if h.get("distance_m") is not None:
            proximity = max(0.0, 1 - h["distance_m"] / SEARCH_RADIUS_M)
        else:
            proximity = 0.5  # verified hotel, but distance unknown
        h["score"] = round(100 * (w["certified"] * certified + w["proximity"] * proximity))
    return sorted(hotels, key=lambda h: h["score"], reverse=True)


def hotel_line(position: int, h: Dict) -> Text:
    parts = []
    if h.get("distance_m") is not None:
        parts.append(f"{h['distance_m']} m from the station")
    cert = h.get("certification")
    if cert:
        parts.append(f"✅ {cert['certification']} (verified {cert['checked_on']})")
    else:
        parts.append("certification not verified")
    if h.get("wheelchair") == "yes":
        parts.append("♿ wheelchair accessible")
    if h.get("stars"):
        parts.append(f"{h['stars']}★")
    return f"{position}. 🏨 {h['name']}: " + " · ".join(parts) + f" (score {h['score']}/100)"


class ActionRecommendOptions(Action):
    def name(self) -> Text:
        return "action_recommend_options"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
        destination = tracker.get_slot("destination") or ""
        level = tracker.get_slot("sustainability_level") or "medium"
        budget = int(tracker.get_slot("budget") or 0)
        nights = estimate_nights(tracker.get_slot("travel_dates"))
        carbon_options = tracker.get_slot("carbon_options") or []
        events: List[Dict[Text, Any]] = []
        human_button = [{"title": "👤 Talk to a human", "payload": "/request_human"}]

        # 1) Rank transport (return trip): carbon + rough fare, weighted by preference
        transport = []
        for option in carbon_options:
            mode = option["mode"]
            km = option.get("distance_km") or 0
            price = 2 * (km * TRANSPORT_PRICE_PER_KM[mode] + TRANSPORT_FIXED_EUR[mode])
            hours = km / TRANSPORT_SPEED_KMH[mode] + (AIRPORT_TIME_H if mode == "air" else 0)
            transport.append({
                **option,
                "round_trip_co2e_kg": round(2 * option["co2e_kg"], 1),
                "round_trip_price_eur": round(price),
                "hours_one_way": round(hours, 1),
            })
        transport = rank_by_score(transport, "round_trip_co2e_kg", "round_trip_price_eur", level)
        best = transport[0] if transport else None

        if best:
            dispatcher.utter_message(text=(
                f"✅ Best transport for you ({level} sustainability): {best['label']}: "
                f"~{best['round_trip_co2e_kg']:.0f} kg CO₂e return, ~{best['hours_one_way']} h each way, "
                f"rough fare estimate ~€{best['round_trip_price_eur']} return (score {best['score']}/100)."
            ))
            events.append(SlotSet("recommended_transport", best))

            # 2) Budget left for accommodation (no invented hotel prices)
            if budget:
                per_night = (budget - best["round_trip_price_eur"]) / nights
                if per_night < 40:
                    dispatcher.utter_message(
                        text=(f"⚠️ After transport, your budget leaves under €40 per night for {nights} nights. "
                              "A human advisor could help find options."),
                        buttons=human_button,
                    )
                else:
                    dispatcher.utter_message(text=(
                        f"💶 After transport, you have about €{per_night:.0f} per night for {nights} nights. "
                        "I don't show hotel prices, so please check the hotel's own website."
                    ))

        # 3) Real hotels near the main station + manually verified certifications
        station = MAIN_STATIONS.get(destination.lower())
        if not station:
            dispatcher.utter_message(
                text=f"I don't have station data for {destination} yet, so I can't search hotels there. A human advisor can help.",
                buttons=human_button,
            )
            return events

        osm_hotels = fetch_osm_hotels(destination)
        certifications = load_verified_certifications(destination)
        if osm_hotels is None and not certifications:
            dispatcher.utter_message(
                text="Sorry, the hotel search (OpenStreetMap) isn't responding right now. Try again in a minute, or ask a human advisor.",
                buttons=human_button,
            )
            return events

        merged, matched = [], set()
        for h in osm_hotels or []:
            cert = match_certification(h["name"], certifications)
            if cert:
                matched.add(cert["hotel_name"])
            merged.append({**h, "certification": cert})
        for cert in certifications:  # verified hotels further from the station
            if cert["hotel_name"] not in matched:
                merged.append({"name": cert["hotel_name"], "distance_m": None, "stars": None,
                               "website": None, "wheelchair": None, "certification": cert})

        if not merged:
            dispatcher.utter_message(
                text=f"I couldn't find hotels near {station[0]}. A human advisor can help.",
                buttons=human_button,
            )
            return events

        ranked = rank_hotels(merged, level)[:5]
        lines = [hotel_line(i, h) for i, h in enumerate(ranked, start=1)]
        dispatcher.utter_message(
            text=f"Hotels within walking distance of {station[0]}:\n" + "\n".join(lines)
        )
        dispatcher.utter_message(text=(
            "ℹ️ Hotel data © OpenStreetMap contributors. Certifications are only shown when checked "
            "on an official registry; 'not verified' means unknown, not uncertified."
        ))

        # Structured data for the web carousel (Step 6)
        dispatcher.utter_message(json_message={
            "type": "hotel_carousel", "station": station[0], "nights": nights, "hotels": ranked,
        })
        dispatcher.utter_message(text="What would you like to do next?", buttons=NEXT_STEP_BUTTONS)
        events.append(SlotSet("hotel_options", ranked))
        return events


class ActionResetTrip(Action):
    """Clear the previous trip when a new one starts, but keep anything the
    user just said (e.g. 'I want to go to Lisbon' keeps destination = Lisbon)."""

    def name(self) -> Text:
        return "action_reset_trip"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
        just_said = {e.get("entity") for e in tracker.latest_message.get("entities", [])}
        return [SlotSet(slot, None) for slot in TRIP_SLOTS if slot not in just_said]

# ---------------------------------------------------------------------------
# Step 5: human handover with full context, 2-stage fallback, carbon offsets
# ---------------------------------------------------------------------------
HANDOVER_DIR = os.path.join(os.path.dirname(__file__), "..", "handover_queue")
HANDOVER_RETENTION_DAYS = 30  # GDPR storage limitation
OFFSETS_FILE = os.path.join(DATA_DIR, "carbon_offsets.json")

NEXT_STEP_BUTTONS = [
    {"title": "🌍 Carbon offset options", "payload": "/ask_offsets"},
    {"title": "👤 Talk to a human", "payload": "/request_human"},
    {"title": "🔄 Plan another trip", "payload": "/plan_trip"},
]


def build_transcript(tracker: Tracker) -> List[Dict[Text, Text]]:
    """Readable chat history for the human advisor."""
    transcript = []
    for event in tracker.events:
        if event.get("event") == "user":
            text = event.get("text") or ""
            if text.startswith("/"):
                text = f"[clicked button: {text}]"
            transcript.append({"from": "user", "text": text})
        elif event.get("event") == "bot" and event.get("text"):
            transcript.append({"from": "bot", "text": event["text"]})
    return transcript[-60:]  # the last 60 messages are plenty


def purge_old_handovers() -> None:
    """GDPR: delete handover files older than the retention period."""
    if not os.path.isdir(HANDOVER_DIR):
        return
    cutoff = time.time() - HANDOVER_RETENTION_DAYS * 86400
    for name in os.listdir(HANDOVER_DIR):
        path = os.path.join(HANDOVER_DIR, name)
        if name.endswith(".json") and os.path.getmtime(path) < cutoff:
            try:
                os.remove(path)
            except OSError:
                pass


def perform_handover(dispatcher: CollectingDispatcher, tracker: Tracker,
                     reason: Text) -> List[Dict[Text, Any]]:
    """Stop any form, package the full conversation context for a human advisor,
    and tell the frontend to show the handover indicator. Shared by
    action_handover and the fallback's second stage."""
    ticket = f"ECO-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    trip = {s: tracker.get_slot(s) for s in
            ["destination", "origin", "travel_dates", "budget", "sustainability_level"]}
    package = {
        "ticket_id": ticket,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason,
        "conversation_id": tracker.sender_id,
        "trip": trip,
        "recommended_transport": tracker.get_slot("recommended_transport"),
        "carbon_options": tracker.get_slot("carbon_options"),
        "hotel_shortlist": [h.get("name") for h in (tracker.get_slot("hotel_options") or [])],
        "transcript": build_transcript(tracker),
        "privacy_note": f"No contact details are requested. Deleted automatically after {HANDOVER_RETENTION_DAYS} days.",
    }

    saved = True
    try:
        os.makedirs(HANDOVER_DIR, exist_ok=True)
        purge_old_handovers()
        with open(os.path.join(HANDOVER_DIR, f"{ticket}.json"), "w", encoding="utf-8") as f:
            json.dump(package, f, ensure_ascii=False, indent=2)
    except OSError as error:
        logger.error("Could not save handover %s: %s", ticket, error)
        saved = False

    if saved:
        known = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in trip.items() if v)
        dispatcher.utter_message(text=(
            f"👤 I've passed our conversation to a human travel advisor (ticket {ticket}). "
            "They can see your trip details and our chat, so you won't need to repeat yourself."
            + (f"\nShared trip details: {known}." if known else "")
            + "\n🔒 I haven't asked for any contact details."
        ))
    else:
        dispatcher.utter_message(text="Sorry, I couldn't reach the advisor team right now. Please try again in a few minutes.")

    # Signal for the frontend's handover indicator (Step 6)
    dispatcher.utter_message(json_message={"type": "handover", "active": saved,
                                           "ticket_id": ticket if saved else None})
    return [ActiveLoop(None), SlotSet("requested_slot", None),
            SlotSet("handover_ticket", ticket if saved else None)]


class ActionHandover(Action):
    """User asked for a human (button or typed request)."""

    def name(self) -> Text:
        return "action_handover"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
        return perform_handover(dispatcher, tracker, "User asked for a human advisor")


def consecutive_fallbacks(tracker: Tracker) -> int:
    """How many of the user's latest messages in a row were not understood."""
    count = 0
    for event in reversed(tracker.events):
        if event.get("event") == "user":
            intent = ((event.get("parse_data") or {}).get("intent") or {}).get("name")
            if intent in ("nlu_fallback", "out_of_scope"):
                count += 1
            else:
                break
    return max(count, 1)


class ActionDefaultFallback(Action):
    """Two-stage clarification:
    1st misunderstanding -> re-prompt with constrained options (buttons);
    2nd in a row        -> hand over to a human with full context.
    Inside the trip form, the human option is offered as a button instead."""

    def name(self) -> Text:
        return "action_default_fallback"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
        human_button = [{"title": "👤 Talk to a human", "payload": "/request_human"}]

        if (tracker.active_loop or {}).get("name"):
            # Inside the trip form: the form will repeat its question with its own buttons
            dispatcher.utter_message(
                text="Sorry, I didn't quite get that. Please choose an option or type your answer again.",
                buttons=human_button,
            )
            return []

        if consecutive_fallbacks(tracker) >= 2:
            dispatcher.utter_message(text="I'm still not sure I understand, so let me bring in a human advisor.")
            return perform_handover(dispatcher, tracker,
                                    "Bot could not understand the user after clarification")

        dispatcher.utter_message(
            text="Sorry, I didn't quite get that. What would you like to do?",
            buttons=[{"title": "🧭 Plan a trip", "payload": "/plan_trip"}] + NEXT_STEP_BUTTONS[:2],
        )
        return []



class ActionOffsetInfo(Action):
    """Recognised offset programmes - presented honestly (reduce first, no 'carbon neutral' claims)."""

    def name(self) -> Text:
        return "action_offset_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
        best = tracker.get_slot("recommended_transport")
        if best:
            dispatcher.utter_message(text=(
                f"Your recommended trip ({best['label']}, return) emits about "
                f"{best['round_trip_co2e_kg']:.0f} kg CO₂e. Choosing lower-carbon transport is the most "
                "effective step. If you'd also like to compensate the rest, these are recognised programmes:"
            ))
        else:
            dispatcher.utter_message(text=(
                "Reducing emissions comes first, for example the train or coach instead of flying. "
                "If you'd also like to compensate, these are recognised programmes:"
            ))

        programmes = (read_json_file(OFFSETS_FILE) or {}).get("programmes", [])
        if not programmes:
            dispatcher.utter_message(
                text="Sorry, I can't load the offset programmes right now. A human advisor can help.",
                buttons=[{"title": "👤 Talk to a human", "payload": "/request_human"}],
            )
            return []

        lines = [f"• {p['name']} ({p['type']}; {p['standard']}): {p['url']}" for p in programmes]
        dispatcher.utter_message(text="\n".join(lines))
        dispatcher.utter_message(text=(
            "⚠️ Offsetting doesn't cancel emissions, so I never call a trip 'carbon neutral'. "
            "Prices and projects vary, so please check each provider's website."
        ))
        dispatcher.utter_message(json_message={"type": "offsets", "programmes": programmes})
        return []