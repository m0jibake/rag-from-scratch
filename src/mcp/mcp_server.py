from email import header
from typing import Any
# import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"


async def get_coordinates(city: str) -> dict:
    """Get latitude and longitude for a city name."""
    import requests
    
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
    
    try:
        response = requests.get(url, timeout=30, verify=False)
        data = response.json()
        
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            return {
                "city": result["name"],
                "country": result.get("country", "Unknown"),
                "latitude": result["latitude"],
                "longitude": result["longitude"]
            }
        return {"error": f"City '{city}' not found"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

@mcp.tool()
async def get_forecast(city: str) -> dict:
    import requests


    coordinates_result = await get_coordinates(city)


    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={coordinates_result.get('latitude')}&longitude={coordinates_result.get('longitude')}"
        "&timezone=Europe/Berlin"
        "&forecast_days=1"
        "&hourly=temperature_2m,relative_humidity_2m,precipitation,rain,snowfall,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m"
        "&daily=temperature_2m_max,temperature_2m_min,sunrise,sunset,precipitation_sum,rain_sum,snowfall_sum,wind_speed_10m_max"
    )

    try:
        import certifi
        import sys
        import os
        # Debug: print environment info
        print(f"REQUESTS_CA_BUNDLE: {os.environ.get('REQUESTS_CA_BUNDLE', 'not set')}", file=sys.stderr)
        print(f"SSL_CERT_FILE: {os.environ.get('SSL_CERT_FILE', 'not set')}", file=sys.stderr)
        print(f"certifi location: {certifi.where()}", file=sys.stderr)
        
        response = requests.get(url, timeout=30, verify=False)
        return response.text
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    return response.text


def main():
    # Initialize and run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
