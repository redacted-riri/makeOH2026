import requests
def get_compass_direction(degrees):
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = round(degrees / 45) % 8
    return directions[index]
def cbusWeather():
    # Added wind_direction_10m to the URL!
    url = "https://api.open-meteo.com/v1/forecast?latitude=39.9625&longitude=-83.0032&current=temperature_2m,wind_speed_10m,wind_gusts_10m,wind_direction_10m&wind_speed_unit=mph&temperature_unit=fahrenheit"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        current = data['current']
        
        temp = current['temperature_2m']
        wind = current['wind_speed_10m']
        gusts = current['wind_gusts_10m']
        wind_deg = current['wind_direction_10m']
        wind_dir = get_compass_direction(wind_deg)
        
        return {"Temp": temp, "Wind": wind, "Gusts": gusts, "Direction": wind_dir}
        
    except Exception as e:
        print(f"⚠️ Weather API Error: {e}")
        return None
# test
if __name__ == "__main__":
    cbusWeather()