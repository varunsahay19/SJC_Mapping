#Defining the function to plot the Folium map for Tap Test data in particular

import folium
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datetime import timedelta
import pandas as pd
from folium.plugins import MarkerCluster
import h5py
import os
from datetime import timezone

def load_sjc_folder(folder_path, analysis_opt):
    """Load DAS data from all .h5 SJC files in a folder, with time and channel filtering.
    
    Automatically handles two different data orientations:
    - Shape (channels, time_points): ~50K channels × ~12K time points (transposed format)
    - Shape (time_points, channels): ~12K time points × ~50K channels (standard format)
    
    Always returns data as (time_points, channels) for consistency.
    """

    all_raw = []
    all_tstamps = []

    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".h5"):
            continue

        fpath = os.path.join(folder_path, fname)
        with h5py.File(fpath, "r") as f:
            # Extract all timestamps
            s_tdata = np.array(
                f["Acquisition"]["Raw[0]"]["RawDataTime"], dtype=np.dtype("<i8")
            )
            s_tdata_npdt64 = s_tdata.astype("datetime64[us]")

            # Time slice selection
            if analysis_opt["file_time_slice"]:
                # Handle both timezone-aware and naive datetime objects
                beg_time = analysis_opt["beg_time"]
                end_time = analysis_opt["end_time"]
                
                # Convert to naive UTC if timezone-aware
                if beg_time.tzinfo is not None:
                    beg_time_utc = beg_time.astimezone(timezone.utc).replace(tzinfo=None)
                    end_time_utc = end_time.astimezone(timezone.utc).replace(tzinfo=None)
                else:
                    beg_time_utc = beg_time
                    end_time_utc = end_time
                
                # Convert to numpy datetime64 for comparison
                beg_time_np = np.datetime64(beg_time_utc)
                end_time_np = np.datetime64(end_time_utc)
                
                time_slice_idx = (s_tdata_npdt64 >= beg_time_np) & (s_tdata_npdt64 < end_time_np)
            else:
                time_slice_idx = np.arange(len(s_tdata_npdt64))

            if not np.any(time_slice_idx):
                continue

            # Channel slice selection
            schan = min(analysis_opt["mmap_chan"])
            echan = max(analysis_opt["mmap_chan"]) + 1  # non-inclusive
            chan_idx = np.arange(schan, echan)

            # Get raw data shape to determine orientation
            raw_data_shape = f["Acquisition"]["Raw[0]"]["RawData"].shape
            num_time_samples = len(s_tdata_npdt64)
            
            # Determine data orientation
            # If first dimension matches number of time samples, it's (time, channels)
            # Otherwise, it's (channels, time) and needs transposing
            if raw_data_shape[0] == num_time_samples:
                # Standard format: (time_points, channels)
                raw_data = np.array(
                    f["Acquisition"]["Raw[0]"]["RawData"][time_slice_idx, schan:echan],
                    dtype=np.float32
                )
            else:
                # Transposed format: (channels, time_points) - need to transpose
                raw_data = np.array(
                    f["Acquisition"]["Raw[0]"]["RawData"][schan:echan, time_slice_idx],
                    dtype=np.float32
                )
                # Transpose to (time_points, channels)
                raw_data = raw_data.T

            all_raw.append(raw_data)
            all_tstamps.append(s_tdata_npdt64[time_slice_idx])

    if not all_raw:
        raise ValueError("No valid data found with the given filters.")

    raw_all = np.concatenate(all_raw, axis=0)
    t_all = np.concatenate(all_tstamps, axis=0)

    return raw_all, t_all


def plot_tap_test_map(df, lat_col='Lat', lon_col='Long', label_col='Localized Channels',
                    map_filename='folium_map.jpg', zoom_start=15):
    df = df.dropna(subset=[lat_col, lon_col])
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')

    map_center = [df[lat_col].mean(), df[lon_col].mean()]
    m = folium.Map(location=map_center, zoom_start=zoom_start)

    for _, row in df.iterrows():
        label = row[label_col] if pd.notna(row[label_col]) else "No Channel Info"
        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=label,
            tooltip=label
        ).add_to(m)

    m.save(map_filename)
    print(f"Map saved to {map_filename}")
    return m

def parse_channel_column(tap_test_df):
    def clean_channel(ch):
        if isinstance(ch, str):
            # Handle range with '-' or with '/' or both
            ch = ch.replace(' ', '')  # remove any spaces
            if '-' in ch:
                try:
                    parts = ch.split('-')
                    return int((int(parts[0]) + int(parts[1])) / 2)
                except:
                    return np.nan
            elif '/' in ch:
                try:
                    parts = ch.split('/')
                    return int((int(parts[0]) + int(parts[1])) / 2)
                except:
                    return np.nan
        try:
            return int(ch)
        except:
            return np.nan

    tap_test_df['Channel'] = tap_test_df['Localized Channels'].apply(clean_channel)
    tap_test_df['Lat'] = pd.to_numeric(tap_test_df['Lat'], errors='coerce')
    tap_test_df['Long'] = pd.to_numeric(tap_test_df['Long'], errors='coerce')
    return tap_test_df

# --- Function to load vehicle GPS data and convert to UTC ---
def load_vehicle_gps(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # clean header if needed

    # Convert MATLAB datenum to datetime
    df['datetime_local'] = pd.to_datetime(df.iloc[:, 0] - 719529, unit='D', origin='unix')
    df['datetime_utc'] = df['datetime_local'] + timedelta(hours=7)  # convert from PDT to UTC
    df = df.rename(columns={df.columns[1]: "Lat", df.columns[2]: "Long"})
    return df[['datetime_utc', 'Lat', 'Long']]



def plot_vehicle_gps_folium(gps_df, filter_point_interval=100):
    # Get center of map (mean location)
    lat_center = gps_df['Lat'].mean()
    lon_center = gps_df['Long'].mean()

    # Create Folium map centered on average location
    m = folium.Map(location=[lat_center, lon_center], zoom_start=14)

    # Add marker cluster
    marker_cluster = MarkerCluster().add_to(m)

    # Plot every 10th point
    # filter_point_interval = 100  # Change this to plot more or fewer points

    # Plot every 10th point
    for _, row in gps_df.iloc[::filter_point_interval].iterrows():
        folium.Marker(
            location=[row['Lat'], row['Long']],
            popup=row['datetime_utc'].strftime('%Y-%m-%d %H:%M:%S'),
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)

    return m


def get_vehicle_gps_at_time_bounds(vehicle_df, beg_time, end_time):
    """
    Find the vehicle GPS points closest in time to beg_time and end_time.
    """
    vehicle_df['time_diff_start'] = (vehicle_df['datetime_utc'] - beg_time).abs()
    vehicle_df['time_diff_end'] = (vehicle_df['datetime_utc'] - end_time).abs()

    closest_start = vehicle_df.loc[vehicle_df['time_diff_start'].idxmin()]
    closest_end = vehicle_df.loc[vehicle_df['time_diff_end'].idxmin()]

    return pd.DataFrame([closest_start, closest_end])[['datetime_utc', 'Lat', 'Long']]

def match_vehicle_points_to_closest_taps(vehicle_bounds_df, tap_test_df):
    match_results = []

    for _, v in vehicle_bounds_df.iterrows():
        vehicle_loc = (v['Lat'], v['Long'])

        min_dist = float('inf')
        closest_tap = None

        for _, tap in tap_test_df.iterrows():
            tap_loc = (tap['Lat'], tap['Long'])
            if pd.isna(tap_loc[0]) or pd.isna(tap_loc[1]):
                continue  # skip invalid coordinates
            distance = geodesic(vehicle_loc, tap_loc).meters

            if distance < min_dist:
                min_dist = distance
                closest_tap = {
                    'Vehicle Lat': vehicle_loc[0],
                    'Vehicle Long': vehicle_loc[1],
                    'Vehicle Time (UTC)': v['datetime_utc'],
                    'Tap Lat': tap_loc[0],
                    'Tap Long': tap_loc[1],
                    'Tap Location': tap['Location'],
                    'Channel': tap['Channel'],
                    'Distance (m)': distance
                }

        if closest_tap:
            match_results.append(closest_tap)
        
        matched_taps_df = pd.DataFrame(match_results)
    
    return matched_taps_df

def find_closest_vehicle_to_each_tap(vehicle_df, matched_taps_df):
    """
    For each tap point in matched_taps_df, find the closest vehicle GPS point.
    Returns a list of dicts with 'CH' (channel) and 'Time' (UTC).
    """
    vehicle_tap_crossings = []

    for _, tap in matched_taps_df.iterrows():
        tap_loc = (tap['Tap Lat'], tap['Tap Long'])

        if pd.isna(tap_loc[0]) or pd.isna(tap_loc[1]) or pd.isna(tap['Channel']):
            continue  # skip bad data

        min_dist = float('inf')
        closest_time = None

        for _, v in vehicle_df.iterrows():
            veh_loc = (v['Lat'], v['Long'])
            if pd.isna(veh_loc[0]) or pd.isna(veh_loc[1]):
                continue

            dist = geodesic(tap_loc, veh_loc).meters
            if dist < min_dist:
                min_dist = dist
                closest_time = v['datetime_utc']

        vehicle_tap_crossings.append({
            "CH": int(round(tap['Channel'])),
            "Time": closest_time
        })

    return vehicle_tap_crossings

def get_vehicle_gps_between(vehicle_df, beg_time, end_time, time_col="datetime_utc", inclusive=True):
    """
    Return vehicle GPS rows between beg_time and end_time (both timezone-aware).
    Assumes vehicle_df[time_col] exists. Keeps original index and returns a copy.

    Parameters:
    - vehicle_df: pd.DataFrame with a datetime column (preferably timezone-aware).
    - beg_time, end_time: datetime, np.datetime64 or str. These will be converted to UTC timestamps.
    - time_col: name of the datetime column in vehicle_df (default 'datetime_utc').
    - inclusive: if True include end_time, otherwise use < end_time.

    Returns:
    - pd.DataFrame subset of vehicle_df between the requested times (UTC).
    """
    if time_col not in vehicle_df.columns:
        raise KeyError(f"Time column '{time_col}' not found in vehicle_df")

    # Normalize vehicle times to UTC-aware timestamps
    vehicle_times = pd.to_datetime(vehicle_df[time_col])
    if vehicle_times.dt.tz is None:
        vehicle_times = vehicle_times.dt.tz_localize('UTC')
    else:
        vehicle_times = vehicle_times.dt.tz_convert('UTC')

    # Helper to convert beg/end to UTC-aware Timestamps
    def _to_utc_ts(t):
        ts = pd.to_datetime(t)
        # pd.Timestamp may carry tzinfo attribute via tz_localize/convert or be tz-aware already
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.tz_localize('UTC')
        else:
            ts = ts.tz_convert('UTC')
        return ts

    beg_ts = _to_utc_ts(beg_time)
    end_ts = _to_utc_ts(end_time)

    if inclusive:
        mask = (vehicle_times >= beg_ts) & (vehicle_times <= end_ts)
    else:
        mask = (vehicle_times >= beg_ts) & (vehicle_times < end_ts)

    out = vehicle_df.loc[mask].copy()
    out[time_col] = vehicle_times.loc[mask].values
    return out


def build_vehicle_tap_crossings(matched_taps_df):
    vehicle_tap_crossings = []

    for _, row in matched_taps_df.iterrows():
        try:
            ch = int(round(row["Channel"]))  # ensure it's an integer
            vehicle_tap_crossings.append({
                "CH": ch,
                "Time": row["Vehicle Time (UTC)"]
            })
        except Exception as e:
            print(f"Skipping malformed channel value: {row['Channel']} ({e})")
            continue

    return vehicle_tap_crossings

def get_two_closest_rows_naive(y_times, vehicle_times, vehicle_df):
    # Remove timezone from vehicle_times (DatetimeIndex)
    
    vehicle_times_np = vehicle_times.tz_convert(None).to_numpy(dtype='datetime64[us]')

    results = []
    for t in y_times:
        t_np = np.datetime64(t.tz_convert(None))  # also naive datetime64

        diffs = np.abs(vehicle_times_np - t_np)
        idxs = np.argpartition(diffs, 2)[:2]
        idxs = idxs[np.argsort(diffs[idxs])]
        results.append(vehicle_df.iloc[idxs].reset_index(drop=True))

    return results

def get_the_gps_for_channels(y_times,x_lowess, two_closest_vehicle_rows):
    gps_data = []
    for i, chan in enumerate(x_lowess):
        # Get the two closest vehicle rows
        # print(chan)
        closest_rows = two_closest_vehicle_rows[i]
        # print(closest_rows)
        # Get the timestamps of the closest vehicle rows
        time_low_idx= np.argmin(closest_rows['datetime_utc'].values)
        time_high_idx = 1 - time_low_idx
        # print(time_low_idx)
        # time_high = max(closest_rows['datetime_utc'].values[0],closest_rows['datetime_utc'].values[1])
        # print(time_low, time_high)
        # Get the GPS coordinates and time
        gps_lat = closest_rows['Lat'].values
        gps_lon = closest_rows['Long'].values
        gps_lat_start = gps_lat[time_low_idx]
        gps_lon_start = gps_lon[time_low_idx]
        gps_lat_end = gps_lat[time_high_idx]
        gps_lon_end = gps_lon[time_high_idx]

        # Interpolate GPS coordinates between the two closest times
        # Assume x_lowess[i] corresponds to a time between closest_rows['datetime_utc'][time_low_idx] and [time_high_idx]
        time_low = closest_rows['datetime_utc'].values[time_low_idx]
        time_high = closest_rows['datetime_utc'].values[time_high_idx]
        # Fractional position between the two times
        total_time_diff = (time_high - time_low).astype('timedelta64[us]').astype(float)
        if total_time_diff == 0:
            frac = 0.0
        else:
            target_time = y_times[i].to_datetime64()
            frac = ((target_time - time_low).astype('timedelta64[us]').astype(float)) / total_time_diff
            frac = np.clip(frac, 0, 1)
        # Linear interpolation
        interp_lat = gps_lat_start + frac * (gps_lat_end - gps_lat_start)
        interp_lon = gps_lon_start + frac * (gps_lon_end - gps_lon_start)
        gps_data.append({
            'Channel': chan,
            'Lat': interp_lat,
            'Lon': interp_lon,
            'Time': target_time
        })
    return pd.DataFrame(gps_data)