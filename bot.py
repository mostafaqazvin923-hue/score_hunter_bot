import requests

url = "https://api.coingecko.com/api/v3/simple/price"

params = {
    "ids": "ethereum,solana,ripple,aptos",
    "vs_currencies": "usd"
}

response = requests.get(url, params=params, timeout=20)

print("STATUS:", response.status_code)
print("DATA:", response.text)
