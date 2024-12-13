import requests
import pandas as pd
from pathlib import Path
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Configure retry logic
session = requests.Session()
retry = Retry(
    total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Directory setup
data_dir = Path.cwd() / "data"
data_dir.mkdir(exist_ok=True)

api_url= "https://datasets-server.huggingface.co/rows?dataset=Q-b1t%2FIMDB-Dataset-of-50K-Movie-Reviews-Backup&config=default&split=train&offset=0&length=100"




params = {
    "dataset": "Q-b1t/IMDB-Dataset-of-50K-Movie-Reviews-Backup",
    "config": "default",
    "split": "train",
    "offset": 0,
    "length": 100,
}

all_rows = []

while True:
    try:
        response = session.get(api_url, params=params)
        if response.status_code == 200:
            data = response.json()
            current_rows = data.get("rows", [])
            if not current_rows:
                break
            all_rows.extend([row["row"] for row in current_rows])
            params["offset"] += params["length"]
        else:
            print(f"Failed to fetch data. HTTP Status: {response.status_code}")
            break
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        break

# Save data
df = pd.DataFrame(all_rows)
output_path = data_dir / "IMDB-Movie-Reviews.csv"
df.to_csv(output_path, index=False)

print(f"Data saved to {output_path}")
print(f"Dataframe shape: {df.shape}")