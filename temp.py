import board
import busio
import adafruit_si7021

def init_sensor():
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        sensor = adafruit_si7021.SI7021(i2c)
        return sensor
    except Exception as e:
        print("Failed to initialize Si7021:", e)
        return None


def get_temp(sensor):
    try:
        return sensor.temperature
    except Exception as e:
        print("Temperature read error:", e)
        return None


def get_humidity(sensor):
    try:
        return sensor.relative_humidity
    except Exception as e:
        print("Humidity read error:", e)
        return None