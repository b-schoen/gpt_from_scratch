import os
import requests
import hashlib
import pathlib


def _generate_cached_filename_for_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def download_file_from_url(
    url: str,
    cache_dir: pathlib.Path = pathlib.Path("download_cache"),
) -> pathlib.Path:
    """Download file, using local cache to avoid repeated downloads."""

    # create cachedir if not exists
    cache_dir.mkdir(exist_ok=True)

    # Generate a unique filename based on the URL
    filename = _generate_cached_filename_for_url(url)
    filepath = cache_dir / filename

    # Check if the file is already cached
    if filepath.exists():
        print(f"File found in cache: {filepath}")
        return filepath

    # If not cached, download the file
    print(f"Downloading file from {url}")
    response = requests.get(url)

    # Raise an exception for HTTP errors
    response.raise_for_status()

    # Save the file to cache
    with open(filepath, "wb") as f:
        f.write(response.content)

    print(f"File downloaded and cached: {filepath}")
    return filepath
