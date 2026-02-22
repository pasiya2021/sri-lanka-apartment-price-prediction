"""
Generate a realistic sample dataset of Sri Lanka apartment listings
for testing the ML pipeline end-to-end.
"""
import csv
import random
import datetime

random.seed(42)

CSV_COLUMNS = [
    "ad_id", "title", "property_type", "listing_type", "location",
    "district", "detailed_address", "price_lkr", "price_type",
    "bedrooms", "bathrooms", "land_size", "land_size_unit",
    "property_size_sqft", "description", "posted_date", "posted_by",
    "url", "scraped_at",
]

# Realistic Sri Lankan data
DISTRICTS = [
    "Colombo", "Gampaha", "Kandy", "Galle", "Kalutara",
    "Negombo", "Kurunegala", "Matara", "Batticaloa", "Jaffna",
]

LOCATIONS_BY_DISTRICT = {
    "Colombo": ["Colombo 01", "Colombo 02", "Colombo 03", "Colombo 04",
                "Colombo 05", "Colombo 06", "Colombo 07", "Colombo 08",
                "Colombo 09", "Colombo 10", "Colombo 11", "Rajagiriya",
                "Nugegoda", "Dehiwala", "Mount Lavinia", "Wellawatte",
                "Bambalapitiya", "Havelock Town", "Narahenpita"],
    "Gampaha": ["Ja-Ela", "Kaduwela", "Wattala", "Maharagama",
                "Kelaniya", "Peliyagoda", "Battaramulla"],
    "Kandy": ["Kandy City", "Peradeniya", "Katugastota", "Digana"],
    "Galle": ["Galle City", "Unawatuna", "Hikkaduwa"],
    "Kalutara": ["Panadura", "Wadduwa", "Kalutara City"],
    "Negombo": ["Negombo City", "Kochchikade"],
    "Kurunegala": ["Kurunegala City"],
    "Matara": ["Matara City"],
    "Batticaloa": ["Batticaloa City"],
    "Jaffna": ["Jaffna City", "Nallur"],
}

APARTMENT_NAMES = [
    "Altair", "Havelock City", "Cinnamon Life", "Capitol TwinPeaks",
    "Monarch", "Emperor", "Platinum One Suites", "Elements",
    "On320", "Krrish Square", "ITC One", "The Grand",
    "Luxury Residencies", "Blue Ocean", "Skyline Towers",
    "Marina Square", "Clearpoint", "96 Residencies",
]

PROPERTY_TYPES = ["Apartment", "Apartment"]
LISTING_TYPES = ["Sale", "Sale", "Sale", "Rent"]

DESCRIPTIONS = [
    "Modern apartment with stunning views. Well-maintained and ready to move in.",
    "Spacious luxury apartment in prime location. Close to schools and shopping.",
    "Newly built apartment with all modern amenities. Swimming pool and gym.",
    "Fully furnished apartment with parking. 24/7 security.",
    "Semi-furnished apartment with balcony. Great for families.",
    "Penthouse apartment with rooftop terrace. Panoramic city views.",
    "Compact studio apartment. Ideal for young professionals.",
    "Duplex apartment with high ceilings. Premium finishes throughout.",
    "Sea-facing apartment with beach access. Restaurant and spa on-site.",
    "Budget-friendly apartment near public transport. Well-connected area.",
]

def generate_price(district, bedrooms, size_sqft, listing_type, has_name):
    """Generate a realistic price based on features."""
    # Base price per sqft varies by district
    base_per_sqft = {
        "Colombo": 45000, "Gampaha": 22000, "Kandy": 20000,
        "Galle": 25000, "Kalutara": 18000, "Negombo": 20000,
        "Kurunegala": 15000, "Matara": 16000, "Batticaloa": 14000,
        "Jaffna": 15000,
    }

    base = base_per_sqft.get(district, 20000)

    # Branded apartments cost more
    if has_name:
        base *= random.uniform(1.2, 1.8)

    # More bedrooms → premium
    base *= (1 + (bedrooms - 2) * 0.08)

    price = base * size_sqft

    # Add noise
    price *= random.uniform(0.7, 1.4)

    if listing_type == "Rent":
        price = price / 200  # monthly rent ~ price/200

    return round(price, -3)  # round to nearest 1000


def generate_dataset(n=300):
    rows = []
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for i in range(1, n + 1):
        district = random.choices(
            DISTRICTS,
            weights=[40, 15, 8, 7, 5, 5, 5, 5, 5, 5],  # Colombo-heavy
            k=1
        )[0]

        location = random.choice(LOCATIONS_BY_DISTRICT[district])
        listing_type = random.choice(LISTING_TYPES)
        bedrooms = random.choice([1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5])
        bathrooms = max(1, bedrooms - random.choice([0, 0, 1]))
        size_sqft = random.randint(400, 5000)

        # Correlate size with bedrooms somewhat
        size_sqft = max(400, int(bedrooms * random.randint(250, 600)
                                 + random.randint(-200, 400)))

        use_name = random.random() < 0.35
        apt_name = random.choice(APARTMENT_NAMES) if use_name else ""

        if apt_name:
            title = f"{apt_name} – {bedrooms} Bedroom Apartment in {location}"
        else:
            title = f"{bedrooms} Bedroom Apartment for {listing_type} in {location}"

        price = generate_price(district, bedrooms, size_sqft, listing_type, use_name)

        floor = random.choice([None, None] + list(range(1, 40)))
        furnished = random.choice(["Fully Furnished", "Semi Furnished", "Unfurnished", ""])
        parking = random.choice(["Yes", "No", ""])
        gym_pool = random.choice(["Swimming Pool", "Gym", "Gym and Pool", ""])

        desc_parts = [random.choice(DESCRIPTIONS)]
        if furnished and furnished != "Unfurnished":
            desc_parts.append(f"{furnished}.")
        if parking == "Yes":
            desc_parts.append("Parking available.")
        if gym_pool:
            desc_parts.append(f"Amenities: {gym_pool}.")
        if floor:
            desc_parts.append(f"Located on floor {floor}.")

        description = " ".join(desc_parts)

        posted_date = (datetime.datetime.now()
                       - datetime.timedelta(days=random.randint(1, 180))
                       ).strftime("%Y-%m-%d")

        row = {
            "ad_id": f"AD{100000 + i}",
            "title": title,
            "property_type": "Apartment",
            "listing_type": listing_type,
            "location": location,
            "district": district,
            "detailed_address": f"{location}, {district}",
            "price_lkr": str(int(price)),
            "price_type": "Total Price" if listing_type == "Sale" else "Per Month",
            "bedrooms": str(bedrooms),
            "bathrooms": str(bathrooms),
            "land_size": "",
            "land_size_unit": "",
            "property_size_sqft": str(size_sqft),
            "description": description,
            "posted_date": posted_date,
            "posted_by": random.choice(["Owner", "Agent", "Developer"]),
            "url": f"https://properties.lk/apartment/{100000 + i}",
            "scraped_at": now,
        }
        rows.append(row)

    # Write CSV
    with open("apartment_data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} sample apartment records -> apartment_data.csv")


if __name__ == "__main__":
    generate_dataset(300)
