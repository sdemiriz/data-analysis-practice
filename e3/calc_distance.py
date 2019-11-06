from pykalman import KalmanFilter as kalman
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import sys

def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis= 1, trkseg= trkseg, doc= doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent= ' ')

def get_data(file):
    # Parse input file
    parsedXML = ET.parse(file)

    # Set up columns for latitude and longitude
    locations = pd.DataFrame(columns= ['lat', 'lon'])

    # Extract latitude and longitude
    for trkpt in parsedXML.iter('{http://www.topografix.com/GPX/1/0}trkpt'):
        observations = pd.DataFrame({"lat": [trkpt.get('lat')], 
                                     "lon": [trkpt.get('lon')]})
        observations['lat'] = observations['lat'].astype(float)
        observations['lon'] = observations['lon'].astype(float)

        # Add values to columns
        locations = locations.append(observations)

    return locations

# Convert degrees to radians
def degToRad(degrees):
    return (degrees * (math.pi/180))

# Adapted from https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
def haversine_calc(points):
    # points[0] - latitude 1
    # points[1] - longitude 1
    # points[2] - latitude 2
    # points[3] - longitude 2

    # Earth's radius constant in meters
    earthRadius = 6371000

    # Convert points to radians
    diff_lat = degToRad(points[2] - points[0])
    diff_lon = degToRad(points[3] - points[1])

    # Apply calculation
    a = math.sin(diff_lat/2)**2 + math.cos(degToRad(points[0])) * math.cos(degToRad(points[2])) * math.sin(diff_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return (earthRadius * c)

def distance(data):
    # Copy and shift data
    data_copy = data.copy()
    data_copy = data_copy.shift(periods= -1)
    data_copy = data_copy.add_prefix('copy_')

    # Join both dataframes and apply calculations
    joined_data = pd.concat([data, data_copy], axis= 1)
    joined_data['distance'] = joined_data.apply(haversine_calc, axis= 1)

    # Calculate sum of distances
    return joined_data['distance'].sum()

def kalman_smooth(data):
    # Kalman filtering, select latitude and longitude
    kalman_data = data[['lat', 'lon']]
    
    # Set up Kalman model parameters
    initial_state = kalman_data.iloc[0]
    observation_covariance = np.diag([0.0002, 0.0002]) ** 2 # 20/100000 = 0.0002
    transition_covariance = np.diag([0.0001, 0.0001]) ** 2  # 10/100000 = 0.0001
    kalman_model = kalman(initial_state_mean = initial_state, 
                    initial_state_covariance = observation_covariance,
                    transition_covariance = transition_covariance, 
                    observation_covariance = observation_covariance)
    
    # Apply smoothing
    kalman_smoothed, _ = kalman_model.smooth(kalman_data)
    return pd.DataFrame(data = kalman_smoothed, columns = ['lat', 'lon'])

def main():
    # Formatted output
    points = get_data(sys.argv[1])
    print('Unfiltered distance: %0.2f' % (distance(points),))
    
    smoothed_points = kalman_smooth(points)
    print('Filtered distance: %0.2f' % (distance(smoothed_points),))
    output_gpx(smoothed_points, 'out.gpx')

if __name__ == '__main__':
    main()