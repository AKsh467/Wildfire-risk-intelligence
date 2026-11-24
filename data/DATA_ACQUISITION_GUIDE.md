# Data Acquisition Guide: NASA FIRMS & NASA POWER

This guide provides official instructions to acquire the real-world satellite and meteorological datasets for the **Wildfire Risk Intelligence and Hotspot Analytics** project covering India (2020–2025).

---

## 1. NASA FIRMS VIIRS S-NPP Active Fire Data (Archive)

The NASA Fire Information for Resource Management System (FIRMS) provides free global active fire hotspot detections derived from the Suomi NPP VIIRS 375 m sensor.

### Step-by-Step Acquisition:
1. **Visit the NASA FIRMS Archive Download Portal:**  
   [https://firms.modaps.eosdis.nasa.gov/download/](https://firms.modaps.eosdis.nasa.gov/download/)
2. **Log in / Sign up with NASA Earthdata:**  
   Registration is free and required for NASA EOSDIS downloads.
3. **Select Request Type:**  
   Choose **"Country / Region Archives"** or **"Custom Area"**.
4. **Select Sensor / Satellite:**  
   Choose **VIIRS 375m (S-NPP)** (Product identifier: `SV-N2` or `VNP14IMGTDL`).
5. **Set Country / Spatial Bounds:**  
   - Country: **India**
   - Or Custom Bounding Box:
     - North: $38.0^\circ\text{N}$
     - South: $6.0^\circ\text{N}$
     - West: $68.0^\circ\text{E}$
     - East: $98.0^\circ\text{E}$
6. **Set Date Range:**  
   - Start Date: `2020-01-01`
   - End Date: `2025-12-31`
7. **Output Format:**  
   Select **CSV**.
8. **Place the Downloaded Files:**  
   Extract and place the CSV file(s) into:  
   `data/raw/firms/` (e.g., `data/raw/firms/fire_archive_SV-N2_India_2020_2025.csv`).

### Schema Verification:
Expected columns include:
`latitude`, `longitude`, `bright_ti4`, `scan`, `track`, `acq_date`, `acq_time`, `satellite`, `instrument`, `confidence`, `version`, `bright_ti5`, `frp`, `daynight`.

To test that your downloaded file is valid and correctly placed, run:
```bash
python -m src.ingestion.firms_loader
```

---

## 2. NASA POWER Meteorological & Climatology Data

The NASA Prediction of Worldwide Energy Resources (POWER) project provides solar, surface, and meteorological data from satellite observations and GEOS assimilations at a $0.5^\circ \times 0.5^\circ$ spatial resolution.

### Method A: Automated Ingestion (Recommended)
Use our built-in API client (`src/ingestion/power_api_client.py`). It queries the NASA POWER REST API and automatically caches the responses locally:

```python
from src.ingestion.power_api_client import NasaPowerClient

client = NasaPowerClient()

# Example: Fetch daily weather for Central India (22.5°N, 78.5°E) for 2020-2025
df_weather = client.fetch_point_dataframe(
    lat=22.5,
    lon=78.5,
    start_date="20200101",
    end_date="20251231"
)
print(df_weather.head())
```

Responses are automatically cached in:  
`data/raw/power/`  
Subsequent calls read from disk instantly without sending redundant network requests.

### Method B: Manual Bulk Download via POWER Data Access Viewer
If you prefer a single CSV download:
1. Visit the **NASA POWER Data Access Viewer**:  
   [https://power.larc.nasa.gov/data-access-viewer/](https://power.larc.nasa.gov/data-access-viewer/)
2. Choose **Daily** temporal average.
3. Select User Community: **Agroclimatology (AG)**.
4. Select Parameters:
   - Temperature at 2 Meters (`T2M`)
   - Maximum Temperature at 2 Meters (`T2M_MAX`)
   - Minimum Temperature at 2 Meters (`T2M_MIN`)
   - Relative Humidity at 2 Meters (`RH2M`)
   - Corrected Total Precipitation (`PRECTOTCORR`)
   - Wind Speed at 10 Meters (`WS10M`)
   - Surface Pressure (`PS`)
5. Select Dates: `2020-01-01` to `2025-12-31`.
6. Download as CSV and save to:  
   `data/raw/power/`
