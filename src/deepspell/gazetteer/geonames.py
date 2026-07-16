"""GeoNames importer — open data (CC-BY 4.0) for COUNTRY / STATE / CITY.

Expects the standard GeoNames dump files in a directory:

* ``countryInfo.txt``        -> COUNTRY tokens (ISO code as abbreviation)
* ``admin1CodesASCII.txt``   -> STATE tokens (admin1 code suffix as abbreviation)
* ``cities500.txt`` (or any cities dump) -> CITY tokens, parented to their
  admin1 region when known, otherwise directly to the country.

GeoNames does not contain road names; combine with the legacy TSV corpus or
a future OSM importer for ROAD coverage.

``fetch_geonames`` downloads the three files (https://download.geonames.org).
"""

from __future__ import annotations

import logging
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path

from . import Gazetteer
from .staging import StagedRow, load_staged

logger = logging.getLogger(__name__)

GEONAMES_BASE_URL = "https://download.geonames.org/export/dump"
DEFAULT_CITIES_FILE = "cities500"

_COUNTRY_BASE = 1_000_000
_ADMIN1_BASE = 2_000_000
_CITY_BASE = 100_000_000  # + geonameid (unique across the dump)


def fetch_geonames(target_dir: str | Path, cities_file: str = DEFAULT_CITIES_FILE) -> Path:
    """Download countryInfo, admin1 codes and a cities dump into *target_dir*."""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("countryInfo.txt", "admin1CodesASCII.txt"):
        path = target / name
        if not path.exists():
            logger.info("downloading %s ...", name)
            urllib.request.urlretrieve(f"{GEONAMES_BASE_URL}/{name}", path)
    txt = target / f"{cities_file}.txt"
    if not txt.exists():
        zip_path = target / f"{cities_file}.zip"
        logger.info("downloading %s.zip ...", cities_file)
        urllib.request.urlretrieve(f"{GEONAMES_BASE_URL}/{cities_file}.zip", zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extract(f"{cities_file}.txt", target)
        zip_path.unlink()
    return target


def _read_tsv(path: Path) -> Iterator[list[str]]:
    with open(path, encoding="utf-8") as tsv_file:
        for line in tsv_file:
            if not line.strip() or line.startswith("#"):
                continue
            yield line.rstrip("\n").split("\t")


def _rows(directory: Path, cities_file: str, source: str) -> Iterator[StagedRow]:
    # -- countries: ISO(0), ISO3(1), ..., name(4), ..., geonameid(16)
    country_temp_id: dict[str, int] = {}
    for parts in _read_tsv(directory / "countryInfo.txt"):
        if len(parts) < 5 or not parts[4].strip():
            continue
        iso = parts[0].strip()
        temp_id = _COUNTRY_BASE + len(country_temp_id)
        country_temp_id[iso] = temp_id
        yield temp_id, "COUNTRY", parts[4].strip(), iso, None, source

    # -- admin1 regions: code(US.CA), name, ascii name, geonameid
    admin1_temp_id: dict[str, int] = {}
    for parts in _read_tsv(directory / "admin1CodesASCII.txt"):
        if len(parts) < 2 or "." not in parts[0]:
            continue
        code = parts[0].strip()
        country_iso, _, region_code = code.partition(".")
        parent = country_temp_id.get(country_iso)
        temp_id = _ADMIN1_BASE + len(admin1_temp_id)
        admin1_temp_id[code] = temp_id
        yield temp_id, "STATE", parts[1].strip(), region_code or None, parent, source

    # -- cities: geonameid(0), name(1), ascii(2), ..., country code(8), ..., admin1(10)
    cities_path = directory / f"{cities_file}.txt"
    for parts in _read_tsv(cities_path):
        if len(parts) < 11 or not parts[1].strip():
            continue
        country_iso = parts[8].strip()
        admin1_code = f"{country_iso}.{parts[10].strip()}" if parts[10].strip() else ""
        parent = admin1_temp_id.get(admin1_code) or country_temp_id.get(country_iso)
        yield _CITY_BASE + int(parts[0]), "CITY", parts[1].strip(), None, parent, source


def import_geonames(
    gaz: Gazetteer,
    directory: str | Path,
    cities_file: str = DEFAULT_CITIES_FILE,
    source: str = "geonames",
) -> int:
    """Import GeoNames dumps from *directory*. Returns the number of imported rows."""
    count = load_staged(gaz, _rows(Path(directory), cities_file, source))
    gaz.set_meta(f"import:{source}", str(directory))
    return count
