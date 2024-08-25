import os
import requests
import hashlib
import pathlib
from typing import Iterable, Type
import dataclasses
import json
import pickle


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


def head(filepath: pathlib.Path, n: int) -> Iterable[str]:
    """Equivalent to `head` command."""

    with filepath.open() as f:

        for _ in range(n):

            line = f.readline()

            # break if done
            if not line:
                break

            yield line.rstrip("\n")


def write_to_json_file[T](data: T, file_path: str) -> None:
    with open(file_path, "wt") as f:
        json.dump(data, f, indent=2)


def read_from_json_file[T](file_path: str) -> T:
    with open(file_path, "rt") as f:
        return json.load(f)


def write_to_pickle_file[T](data: T, file_path: str) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def read_from_pickle_file[T](file_path: str) -> T:
    with open(file_path, "rb") as f:
        return pickle.load(f)


def serialize_dataclass_to_pickle_file[T](instance: T, file_path: str) -> None:
    write_to_pickle_file(data=instance, file_path=file_path)


def deserialize_dataclass_from_pickle_file[T](cls: Type[T], file_path: str) -> T:
    return read_from_pickle_file(file_path)


def serialize_dataclass_to_json_file[T](instance: T, file_path: str) -> None:

    assert dataclasses.is_dataclass(instance)

    write_to_json_file(data=dataclasses.asdict(instance), file_path=file_path)


def deserialize_dataclass_from_json_file[T](cls: Type[T], file_path: str) -> T:

    instance_as_json = read_from_json_file(file_path)

    return cls(**instance_as_json)
