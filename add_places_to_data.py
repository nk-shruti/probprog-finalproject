import requests
from tqdm import tqdm
import pandas as pd

API_KEY = 'AIzaSyDxzBQzQ5w18tBK2UQJU43X8plPKEyfpig'

def main():
    data = pd.read_csv('data/final_data.csv')
    latitudes = data['Latitude']
    longitudes = data['Longitude']
    radius = 500
    places = ['restaurants', 'bar', 'park']
    latitudes  = data['Latitude']
    longitudes = data['Longitude']
    radius = 500 #meters
    k = {}
    counter = 0
    with tqdm(total=len(places)*len(latitudes)) as pbar:
        for place in places:
            k[place] = []
            for lat, long in zip(latitudes, longitudes):
                res = requests.get('https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={},{}&radius={}&type={}&key=AIzaSyDxzBQzQ5w18tBK2UQJU43X8plPKEyfpig'.format(lat, long, radius, place))
                k[place].append(len(res.json()['results']))
                counter+=1
                pbar.update(counter)
    
    for place in places:
        data[place] = k[place]
    data.to_csv('data/places.csv')
        
if __name__=='__main__':
    main()