from dateutil import parser
import pandas as pd
import datetime

def parse_date_flexible(date_input) -> str:
    """
    Parse a date input in various formats (string or Timestamp) and return a string in 'YYYY-MM-DD' format.
    
    This function can handle:
      - Strings such as '2024-08-08-07', '2024-08-08T07:02:59+00:00', '2024/08/08 07', etc.
      - pd.Timestamp or datetime.datetime objects.
      
    Args:
        date_input: A date provided as a string, pd.Timestamp, or datetime.datetime.
    
    Returns:
        str: The date formatted as 'YYYY-MM-DD'.
    
    Raises:
        ValueError: If the date cannot be parsed.
    """
    # If it's already a datetime object (including pd.Timestamp), just format it.
    if isinstance(date_input, (datetime.datetime, pd.Timestamp)):
        return date_input.strftime("%Y-%m-%d")
    
    try:
        dt = parser.parse(date_input)
        return dt.strftime("%Y-%m-%d")
    except Exception as e:
        raise ValueError(f"Unable to parse date input '{date_input}': {e}")

def build_where_date_clause(start_date, end_date, date_column='date'):

    start_date = '1900-01-01' if start_date is None else parse_date_flexible(start_date)
    end_date = '3900-01-01' if end_date is None else  parse_date_flexible(end_date)

    # Convert to datetime objects and set the start time at midnight.
    start_date_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    # For the end date, add one day (this handles month/year boundaries automatically).
    end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(days=1)

    # Format the datetime objects to strings with time (e.g., "YYYY-MM-DD HH:MM:SS")
    start_date_formatted = start_date_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_date_formatted = end_date_dt.strftime("%Y-%m-%d %H:%M:%S")

    # Build the where clause so that we use a half-open interval [start_date, end_date)
    where_clause = f"{date_column} >= '{start_date_formatted}' AND {date_column} < '{end_date_formatted}'"

    return where_clause


def _gpkg_images_none(gpkg_images) -> bool:
    """Return True if no images are loaded."""
    return gpkg_images is None

def _gpkg_images_empty(gpkg_images) -> bool:
    """Return True if the images GeoDataFrame is empty."""
    return gpkg_images.empty if gpkg_images is not None else True
