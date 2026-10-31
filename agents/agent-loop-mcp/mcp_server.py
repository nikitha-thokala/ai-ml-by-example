"""
MCP tool server for the sample agent: trip, weather and unit-conversion tools over sample data.
The agent starts this file itself over stdio; you don't run it by hand.

  Tools: get_trip, get_weather, convert_temperature
"""
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer(name="trip-tools", instructions="Read-only tools for planning a trip. No tool changes anything.")
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False)

# --- sample data (in a real app: a database and a weather API) ---
TRIPS = {
    "T100": {"trip_id": "T100", "traveler": "Sam", "cities": ["Lisbon", "Madrid", "Barcelona"]},
    "T200": {"trip_id": "T200", "traveler": "Priya", "cities": ["Oslo", "Stockholm"]},
}
WEATHER = {
    "Lisbon": {"temp_c": 24, "condition": "sunny"},
    "Madrid": {"temp_c": 31, "condition": "clear"},
    "Barcelona": {"temp_c": 27, "condition": "humid"},
    "Oslo": {"temp_c": 12, "condition": "rain"},
    "Stockholm": {"temp_c": 14, "condition": "cloudy"},
}


@server.tool(annotations=READ_ONLY)
def get_trip(trip_id: str) -> dict:
    """Get a trip's traveler and the list of cities it visits.

    Args:
        trip_id: The trip ID, e.g. "T100".
    """
    return TRIPS.get(trip_id.strip().upper(), {"error": f"trip {trip_id} not found"})


@server.tool(annotations=READ_ONLY)
def get_weather(city: str) -> dict:
    """Get today's weather for one city, in Celsius.

    Args:
        city: The city name, e.g. "Lisbon".
    """
    weather = WEATHER.get(city.strip().title())
    return {"city": city, **weather} if weather else {"error": f"no weather data for {city}"}


@server.tool(annotations=READ_ONLY)
def convert_temperature(celsius: float) -> dict:
    """Convert a temperature from Celsius to Fahrenheit.

    Args:
        celsius: The temperature in Celsius.
    """
    return {"celsius": celsius, "fahrenheit": round(celsius * 9 / 5 + 32, 1)}


if __name__ == "__main__":
    server.run("stdio")
