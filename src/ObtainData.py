import cdsapi
import os, re
import requests
import yaml

# ==========================================================
# config setup
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

RAWDATA_DIR = config['rawdata_dir']
LANDCOVER_DIR = config['rawdata_subdirs']['landcover_dir']
ERA5_DIR = config['rawdata_subdirs']['era5_dir']
LANDCOVER_TXT_FILES_DIR = config['landcover_txt_files_dir']

YEARS = config['years']
AREA_BOUNDS = config['area_bounds']

# make data folders
os.makedirs(RAWDATA_DIR, exist_ok=True)
os.makedirs(f"{RAWDATA_DIR}/{ERA5_DIR}", exist_ok=True)
os.makedirs(f"{RAWDATA_DIR}/{LANDCOVER_DIR}", exist_ok=True)

print(f"Years to download: {YEARS}")

# ==========================================================
# download ERA5 data
print("Downloading ERA5 data...")

# if all files already exist, skip download
if all(os.path.exists(f"{RAWDATA_DIR}/{ERA5_DIR}/t2m_{year}.nc") for year in YEARS):
    print("All ERA5 files already exist, skipping download.")
    download_era5 = False
else:
    download_era5 = True

if download_era5:
    for year in YEARS:
        print(f"Downloading {year} data...")
        if os.path.exists(f"{RAWDATA_DIR}/{ERA5_DIR}/t2m_{year}.nc"):
            print(f"  File for year {year} already exists, skipping download.")
            continue
        
        dataset = "reanalysis-era5-single-levels"
        request = {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature"],
            "year": [str(year)],
            "month": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12"
            ],
            "day": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12",
                "13", "14", "15",
                "16", "17", "18",
                "19", "20", "21",
                "22", "23", "24",
                "25", "26", "27",
                "28", "29", "30",
                "31"
            ],
            "time": [
                "00:00", "01:00", "02:00",
                "03:00", "04:00", "05:00",
                "06:00", "07:00", "08:00",
                "09:00", "10:00", "11:00",
                "12:00", "13:00", "14:00",
                "15:00", "16:00", "17:00",
                "18:00", "19:00", "20:00",
                "21:00", "22:00", "23:00"
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": AREA_BOUNDS
        }

        client = cdsapi.Client()
        client.retrieve(dataset, request).download(f"{RAWDATA_DIR}/{ERA5_DIR}/t2m_{year}.nc")



# ==========================================================
# download landcover data
print("Downloading Landcover data...")

def parse_tile_coordinates(filename):
    """
    Parses '50N_090W' into (50, -90).
    Returns (lat_top, lon_left).
    """
    # Regex to find pattern like 50N_090W
    match = re.search(r'(\d{2})([NS])_(\d{3})([EW])', filename)
    if not match:
        return None, None
    
    lat_num, lat_dir, lon_num, lon_dir = match.groups()
    
    lat = int(lat_num)
    if lat_dir == 'S': lat = -lat
    
    lon = int(lon_num)
    if lon_dir == 'W': lon = -lon
    
    return lat, lon

def is_tile_in_bbox(tile_top, tile_left, u_north, u_west, u_south, u_east):
    """
    Checks if a 10x10 degree tile overlaps with the user bbox.
    Hansen tiles are 10x10, coordinate is Top-Left corner.
    """
    tile_bottom = tile_top - 10
    tile_right = tile_left + 10
    
    # Check for non-overlap
    if tile_bottom >= u_north: return False # Tile is strictly above box
    if tile_top <= u_south:    return False # Tile is strictly below box
    if tile_left >= u_east:    return False # Tile is strictly right of box
    if tile_right <= u_west:   return False # Tile is strictly left of box
    
    return True

def download_landcover_year(year, bounds):
    download_dir = f"{RAWDATA_DIR}/{LANDCOVER_DIR}/landcover_{year}"
    os.makedirs(download_dir, exist_ok=True)

    # url list files
    with open(f"{LANDCOVER_TXT_FILES_DIR}/landcover_files_{year}.txt", "r") as f:
        url_list_raw = f.read()

    urls = [line.strip() for line in url_list_raw.strip().split('\n') if line.strip()]
    download_queue = []

    #print(f"Filtering {len(urls)} URLs against BBox: N{bounds[0]}, W{bounds[1]}, S{bounds[2]}, E{bounds[3]}...")

    for url in urls:
        filename = url.split('/')[-1]
        lat_top, lon_left = parse_tile_coordinates(filename)
        
        if lat_top is not None:
            if is_tile_in_bbox(lat_top, lon_left, bounds[0], bounds[1], bounds[2], bounds[3]):
                download_queue.append(url)

    #print(f"Found {len(download_queue)} matching tiles.")

    # download files
    for i, url in enumerate(download_queue):
        filename = url.split('/')[-1]
        filepath = os.path.join(download_dir, filename)
        
        if os.path.exists(filepath):
            #print(f"[{i+1}/{len(download_queue)}] Skipping {filename} (already exists)")
            continue
            
        print(f"[{i+1}/{len(download_queue)}] Downloading {filename}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    #print(f"Download complete for year {year}.")


# if all landcover files already exist, skip download
if all(
    os.path.exists(f"{RAWDATA_DIR}/{LANDCOVER_DIR}/landcover_{year}") and
    len(os.listdir(f"{RAWDATA_DIR}/{LANDCOVER_DIR}/landcover_{year}")) > 0
    for year in YEARS):
    print("All Landcover files already exist, skipping download.")
    download_landcover = False
else:
    download_landcover = True

# download landcover data for each year
if download_landcover:
    for year in YEARS:
        print(f"Starting download for year {year}...")
        download_landcover_year(year, AREA_BOUNDS)
print("Data download complete.")

