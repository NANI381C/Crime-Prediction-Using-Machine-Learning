# src/hotspot_cluster.py
from sklearn.cluster import KMeans
import folium
from folium.plugins import HeatMap, MarkerCluster
from geopy.geocoders import Nominatim
import time

# Location coordinates for Indian states and major cities
LOCATION_COORDS = {
    'Delhi': (28.7041, 77.1025),
    'Mumbai': (19.0760, 72.8777),
    'Bangalore': (12.9716, 77.5946),
    'Chennai': (13.0827, 80.2707),
    'Hyderabad': (17.3850, 78.4867),
    'Kolkata': (22.5726, 88.3639),
    'Pune': (18.5204, 73.8567),
    'Ahmedabad': (23.0225, 72.5714),
    'Jaipur': (26.9124, 75.7873),
    'Surat': (21.1702, 72.8311),
    'Lucknow': (26.8467, 80.9462),
    'Kanpur': (26.4499, 80.3319),
    'Nagpur': (21.1458, 79.0882),
    'Indore': (22.7196, 75.8577),
    'Thane': (19.2183, 72.9781),
    'Bhopal': (23.2599, 77.4126),
    'Visakhapatnam': (17.6868, 83.2185),
    'Pimpri-Chinchwad': (18.6279, 73.8007),
    'Patna': (25.5941, 85.1376),
    'Vadodara': (22.3072, 73.1812),
    'Karnataka': (15.3173, 75.7139),
    'Maharashtra': (19.7515, 75.7139),
    'West Bengal': (22.9868, 87.8550),
    'Tamil Nadu': (11.1271, 78.6569),
    'Uttar Pradesh': (26.8467, 80.9462),
    'Madhya Pradesh': (23.2599, 77.4126),
    'Rajasthan': (27.0238, 74.2179),
    'Gujarat': (22.2587, 71.1924),
    'Bihar': (25.0961, 85.3131),
    'Punjab': (31.1471, 75.3412),
    'Haryana': (29.0588, 76.0856),
    'Kerala': (10.8505, 76.2711),
    'Odisha': (20.9517, 85.0985),
    'Jharkhand': (23.6102, 85.2799),
    'Chhattisgarh': (21.2787, 81.8661),
    'Assam': (26.2006, 92.9376),
    'Uttarakhand': (30.0668, 79.0193),
    'Himachal Pradesh': (31.1048, 77.1734),
    'Jammu & Kashmir': (33.7782, 76.5762),
    'Goa': (15.2993, 74.1240),
    'Arunachal Pradesh': (28.2170, 94.7278),
    'Nagaland': (26.1584, 94.5624),
    'Manipur': (24.6637, 93.9063),
    'Mizoram': (23.1645, 92.9376),
    'Tripura': (23.9408, 91.9882),
    'Sikkim': (27.5330, 88.5122),
    'Meghalaya': (25.4670, 91.3662),
    'Telangana': (18.1124, 79.0193)
}


def _extract_location_names(df_monthly, encoders):
    if 'City_enc' in df_monthly.columns:
        df_mean = df_monthly.groupby('City_enc')['Crime_Count'].mean().reset_index()
        le = encoders.get('City', None)
        if le is not None:
            df_mean['City'] = le.inverse_transform(df_mean['City_enc'].astype(int))
        else:
            df_mean['City'] = df_mean['City_enc'].astype(str)
    elif 'City' in df_monthly.columns:
        df_mean = df_monthly.groupby('City')['Crime_Count'].mean().reset_index()
    else:
        raise KeyError('No City or City_enc column found in the dataset.')
    return df_mean


def _resolve_coordinates(location_name):
    lat, lon = None, None
    try:
        loc = Nominatim(user_agent="crime_hotspot_app", timeout=10).geocode(f"{location_name}, India")
        if loc:
            lat, lon = loc.latitude, loc.longitude
    except Exception:
        pass
    if lat is None and location_name in LOCATION_COORDS:
        lat, lon = LOCATION_COORDS[location_name]
    return lat, lon


def generate_hotspot_map(df_monthly, encoders, n_clusters=5):
    df_mean = _extract_location_names(df_monthly, encoders)
    lats, lons = [], []
    for city in df_mean['City']:
        lat, lon = _resolve_coordinates(city)
        lats.append(lat)
        lons.append(lon)
        time.sleep(0.1)

    df_mean['lat'] = lats
    df_mean['lon'] = lons
    df_mean = df_mean.dropna(subset=['lat', 'lon'])

    if df_mean.empty:
        return None

    kmeans = KMeans(n_clusters=min(n_clusters, len(df_mean)), random_state=42)
    df_mean['cluster'] = kmeans.fit_predict(df_mean[['Crime_Count']])

    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles='CartoDB dark_matter')
    heat_data = df_mean[['lat', 'lon']].values.tolist()
    HeatMap(heat_data, radius=25, blur=15, min_opacity=0.4).add_to(m)

    marker_cluster = MarkerCluster(name='Location Clusters').add_to(m)
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'cadetblue']

    for _, r in df_mean.iterrows():
        popup_html = f"<b>{r['City']}</b><br>Avg Crimes: {r['Crime_Count']:.1f}"
        folium.CircleMarker(
            location=[r['lat'], r['lon']],
            radius=6 + (r['Crime_Count'] / (df_mean['Crime_Count'].max() + 1)) * 6,
            color=colors[int(r['cluster']) % len(colors)],
            fill=True,
            fill_color=colors[int(r['cluster']) % len(colors)],
            fill_opacity=0.7,
            popup=popup_html
        ).add_to(marker_cluster)
    return m


def calculate_risk(crime_count):
    if crime_count < 50:
        return "Low"
    elif crime_count < 150:
        return "Medium"
    else:
        return "High"


def generate_risk_map(df_monthly, encoders):
    df_mean = _extract_location_names(df_monthly, encoders)
    lats, lons = [], []
    for city in df_mean['City']:
        lat, lon = _resolve_coordinates(city)
        lats.append(lat)
        lons.append(lon)
        time.sleep(0.1)

    df_mean['lat'] = lats
    df_mean['lon'] = lons
    df_mean['Risk_Level'] = df_mean['Crime_Count'].apply(calculate_risk)
    df_mean = df_mean.dropna(subset=['lat', 'lon'])

    if df_mean.empty:
        return None

    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles='CartoDB dark_matter')
    risk_colors = {"High": "red", "Medium": "orange", "Low": "green"}

    for _, r in df_mean.iterrows():
        color = risk_colors.get(r['Risk_Level'], 'gray')
        popup_html = f"<b>{r['City']}</b><br>Risk: {r['Risk_Level']}<br>Avg Crimes: {r['Crime_Count']:.1f}"
        folium.CircleMarker(
            location=[r['lat'], r['lon']],
            radius=10,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=popup_html
        ).add_to(m)

    return m
