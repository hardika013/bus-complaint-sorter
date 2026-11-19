import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()
API_KEY = __import__("os").getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Gemini API key not found. Check your .env file.")

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.5-flash-lite"

CATEGORIES = {
    "delay",
    "crowding",
    "driver",
    "vehicle",
    "route_request",
    "other"
}


def load_csv(filename):
    with open(filename, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def extract_route(complaint):
    """Python independently extracts the route number."""
    match = re.search(r"\broute\s*(\d+)\b", complaint, re.IGNORECASE)

    if match:
        return match.group(1)

    return None


def classify_with_gemini(complaint):
    """Classify complaints using Gemini with retry handling."""

    prompt = f"""
You are helping a college transport officer organise bus complaints.

Classify using exactly one category:
delay, crowding, driver, vehicle, route_request, other.

Urgency must be true ONLY when there is an actual safety risk.

Examples:
- Dangerous driving: urgent.
- Damaged tyres: urgent.
- Faulty doors: urgent.
- Fire or burning smell: urgent.
- Ordinary delays: NOT urgent.
- Ordinary overcrowding: NOT automatically urgent.
- Requests for new stops: NOT urgent.

Write a short anonymous summary.
Do not include names or personal identifiers.
Refer to the complainant as "student" if necessary.

Return ONLY valid JSON:
{{
    "category": "one category from the list",
    "urgency": false,
    "summary": "short anonymous summary"
}}

Complaint:
{complaint}
"""

    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json"
                }
            )

            if not response.text:
                raise ValueError("Gemini returned an empty response.")

            data = json.loads(response.text)

            if (
                data.get("category") not in CATEGORIES
                or type(data.get("urgency")) is not bool
                or not isinstance(data.get("summary"), str)
                or not data.get("summary").strip()
            ):
                raise ValueError("Invalid AI response format.")

            return data

        except (ValueError, TypeError, json.JSONDecodeError) as error:
            print(
                f"Invalid AI response. "
                f"Attempt {attempt + 1}/4: {error}"
            )

        except Exception as error:
            error_message = str(error).lower()

            temporary_error = any(
                code in error_message
                for code in [
                    "503", "429", "500", "502", "504",
                    "unavailable", "deadline_exceeded"
                ]
            )

            if temporary_error and attempt < 3:
                wait_time = 3 * (2 ** attempt)

                print(
                    f"Temporary Gemini error. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)
                continue

            print(f"Gemini API error: {error}")
            return None

    return None



def save_csv(filename, rows, fields):
    with open(filename, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("\n===================================")
    print("       BUS COMPLAINT SORTER")
    print("===================================\n")

    complaints = load_csv("complaints.csv")
    routes = load_csv("routes.csv")

    # Python creates the official route lookup.
    route_lookup = {
        str(route["route"]).strip(): route["area"]
        for route in routes
    }

    route_counts = Counter({route: 0 for route in route_lookup})

    results = []
    manual_review = []

    print(f"Loaded {len(complaints)} complaints.")
    print(f"Loaded {len(routes)} bus routes.\n")

    for index, complaint in enumerate(complaints, start=1):
        complaint_id = complaint["id"]
        description = complaint["complaint"]

        print(f"Processing {index}/{len(complaints)}: {complaint_id}")

        # Route extraction and validation are done by Python.
        route_number = extract_route(description)

        if route_number is None:
            route_status = "not_stated"
            route_area = ""
        elif route_number in route_lookup:
            route_status = "valid"
            route_area = route_lookup[route_number]
            route_counts[route_number] += 1
        else:
            route_status = "invalid"
            route_area = ""

        # Gemini provides a classification suggestion.
        ai_result = classify_with_gemini(description)

        if ai_result is None:
            category = "manual_review"
            urgency = "unknown"
            summary = "AI response invalid or unavailable."
            review_reason = "AI classification requires manual review."
        else:
            category = ai_result["category"]
            urgency = ai_result["urgency"]
            summary = ai_result["summary"]
            review_reason = ""

        # Human review is required for uncertain route information
        # or failed AI classification.
        if route_status != "valid":
            review_reason += " Route requires human verification."

        needs_review = (
            ai_result is None or route_status != "valid"
        )

        result = {
            "id": complaint_id,
            "date": complaint["date"],
            "complaint": description,
            "route": route_number or "",
            "route_area": route_area,
            "route_validation": route_status,
            "category": category,
            "urgency": urgency,
            "summary": summary,
            "human_review_required": needs_review,
            "review_reason": review_reason.strip(),
            "final_decision": ""
        }

        results.append(result)

        if needs_review:
            manual_review.append(result)

    # Save classified complaints.
    fields = [
        "id", "date", "complaint", "route", "route_area",
        "route_validation", "category", "urgency", "summary",
        "human_review_required", "review_reason", "final_decision"
    ]

    save_csv("classified_complaints.csv", results, fields)

    # Python counts complaints for each route.
    route_report = [
        {
            "route": route,
            "area": area,
            "complaint_count": route_counts[route]
        }
        for route, area in route_lookup.items()
    ]

    save_csv(
        "route_counts.csv",
        route_report,
        ["route", "area", "complaint_count"]
    )

    save_csv("manual_review.csv", manual_review, fields)

    # Create a simple summary report.
    category_counts = Counter(row["category"] for row in results)
    urgent_count = sum(row["urgency"] is True for row in results)

    with open("summary_report.txt", "w", encoding="utf-8") as file:
        file.write("BUS COMPLAINT SORTER - SUMMARY REPORT\n")
        file.write("=====================================\n\n")
        file.write(f"Total complaints: {len(results)}\n")
        file.write(f"Urgent safety complaints: {urgent_count}\n")
        file.write(f"Complaints requiring human review: {len(manual_review)}\n\n")

        file.write("COMPLAINT CATEGORIES\n")
        file.write("--------------------\n")

        for category, count in sorted(category_counts.items()):
            file.write(f"{category}: {count}\n")

        file.write("\nIMPORTANT:\n")
        file.write(
            "AI classifications are suggestions only. "
            "The transport officer must review and make all final decisions.\n"
        )

    print("\n===================================")
    print("          PROCESS COMPLETE")
    print("===================================")
    print("Created:")
    print("1. classified_complaints.csv")
    print("2. route_counts.csv")
    print("3. manual_review.csv")
    print("4. summary_report.txt")
    print("\nHuman review is required before final decisions.")


if __name__ == "__main__":
    main()
