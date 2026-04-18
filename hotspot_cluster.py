# src/hotspot_cluster.py
from sklearn.cluster import KMeans
import folium
from geopy.geocoders import Nominatim
import time

# Fallback coordinates for major Indian cities
FALLBACK_COORDS = {
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
    'Vadodara': (22.3072, 73.1812)
}

def generate_hotspot_map(df_monthly, encoders, n_clusters=5):
    """
    df_monthly: aggregated monthly df (City_enc, Date, Crime_Count, ...)
    encoders: dict returned from data_prep (to decode city names)
    """
    df_mean = df_monthly.groupby('City_enc')['Crime_Count'].mean().reset_index()
    # decode names
    le = encoders.get('City', None)
    if le is not None:
        df_mean['City'] = le.inverse_transform(df_mean['City_enc'].astype(int))
    else:
        df_mean['City'] = df_mean['City_enc'].astype(str)

    # geocode each city with fallback
    geolocator = Nominatim(user_agent="crime_hotspot_app", timeout=10)
    lats, lons = [], []
    for city in df_mean['City']:
        lat, lon = None, None
        try:
            loc = geolocator.geocode(f"{city}, India")
            if loc:
                lat, lon = loc.latitude, loc.longitude
        except Exception:
            pass
        
        # Use fallback if geocoding failed
        if lat is None and city in FALLBACK_COORDS:
            lat, lon = FALLBACK_COORDS[city]
        
        lats.append(lat)
        lons.append(lon)
        time.sleep(0.1)  # Reduced delay

    df_mean['lat'] = lats
    df_mean['lon'] = lons
    df_mean = df_mean.dropna(subset=['lat','lon'])

    if df_mean.empty:
        return None

    kmeans = KMeans(n_clusters=min(n_clusters, len(df_mean)), random_state=42)
    df_mean['cluster'] = kmeans.fit_predict(df_mean[['Crime_Count']])

    m = folium.Map(location=[20.5937,78.9629], zoom_start=5, tiles='CartoDB dark_matter')
    colors = ['red','orange','yellow','green','blue','purple','cadetblue']

    for _, r in df_mean.iterrows():
        folium.CircleMarker(
            location=[r['lat'], r['lon']],
            radius=6 + (r['Crime_Count'] / (df_mean['Crime_Count'].max()+1))*6,
            color=colors[int(r['cluster'])%len(colors)],
            fill=True,
            popup=f"{r['City']}: Avg {r['Crime_Count']:.1f}"
        ).add_to(m)
    return m
