import requests
from tqdm import tqdm
import pandas as pd
import sys

BASE = 'https://maps.googleapis.com/maps/api/place/nearbysearch'
STR_REQ = BASE + '/json?location={},{}&radius={}&type={}&key={}'


def main(api_key):
    data = pd.read_csv('data/block_data.csv')
    latitudes = data['Latitude']
    longitudes = data['Longitude']
    radius = 500
    places = ['restaurants', 'bar', 'park']
    latitudes = data['Latitude']
    longitudes = data['Longitude']
    radius = 500  # meters
    k = {}
    counter = 0
    with tqdm(total=len(places) * len(latitudes)) as pbar:
        for place in places:
            k[place] = []
            for lat, long in zip(latitudes, longitudes):
                req = STR_REQ.format(lat, long, radius, place, api_key)
                res = requests.get(req)
                k[place].append(len(res.json()['results']))
                counter += 1
                pbar.update(counter)

    for place in places:
        data[place] = k[place]
    data.to_csv('data/places.csv')


if __name__ == '__main__':
    main(sys.argv[1:])
