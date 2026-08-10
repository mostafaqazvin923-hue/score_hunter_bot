import requests

url = "https://pro-api.coinmarketcap.com/public-api/v1/simple/price"

params = {
    "ids": "52,1027,5426,21794",
    "convert": "USD"
}

response = requests.get(url, params=params, timeout=20)

print("STATUS:", response.status_code)
print("DATA:", response.text)
