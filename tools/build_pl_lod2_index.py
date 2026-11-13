"""Build server/data/pl_lod2_sheets.json.gz: where GUGiK's LoD2 3D building models of Poland are (pl_lod2.py).

GUGiK publishes the models (CityGML 2.0, LoD2, 2015-2018) as one zip per powiat (county), named by its TERYT code:
https://opendata.geoportal.gov.pl/InneDane/Budynki3D/LOD2/<TERYT>_gml.zip, each holding one file per 1:5,000 map
sheet of the national division (Modele_3D/<TERYT>_<sheet>.gml, e.g. 1465_N-34-139-A-c-1-1.gml), only that powiat's
buildings in it. Not every powiat has them. This tool asks the host for every possible powiat code (HEAD, 4 at a
time), reads the central directory of each zip that exists (HTTP Range requests: a few KB, not the zip) and writes,
per sheet, the entries of the powiats that cover it: where the entry starts in its zip, its compressed and full
sizes, CRC-32 and compression method. The server then needs one Range request per sheet and no directory reads,
and knows offline where there are models. A zip that has changed since (its size or the entry's CRC differ) is
found at runtime and left out (the Overture buildings stay).

  python tools/build_pl_lod2_index.py [--out server/data/pl_lod2_sheets.json.gz]

Licence of the data: CC BY 4.0 (GUGiK); the index is facts about the files. Run by hand when GUGiK publishes more.
"""
import argparse
import concurrent.futures as cf
import datetime
import gzip
import http.client
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile

BASE = "https://opendata.geoportal.gov.pl/InneDane/Budynki3D/LOD2/"
UA = "BeamNG.Earth data build (https://github.com/TecnaGamer/BeamNG.Earth)"
NAME = re.compile(r"(?:^|/)(\d{4})_([A-Z]-\d{1,2}-\d{1,3}-[A-D]-[a-d]-[1-4]-[1-4])\.gml$", re.I)


def _open(req, timeout):
    """urlopen with retries (the host now and then drops a connection without an answer)."""
    for attempt in range(5):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError:
            raise
        except (OSError, http.client.HTTPException):
            if attempt == 4:
                raise
            time.sleep(2.0 * 2 ** attempt)


class RangeReader(io.RawIOBase):
    """A seekable file over HTTP Range requests (enough for zipfile to read a central directory)."""

    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or not len(b):
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        req = urllib.request.Request(self.url, headers={"User-Agent": UA, "Range": f"bytes={self.pos}-{end}"})
        with _open(req, 120) as r:
            d = r.read()
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def head(code):
    url = f"{BASE}{code}_gml.zip"
    try:
        with _open(urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA}), 60) as r:
            return code, int(r.headers["Content-Length"]), r.headers.get("Last-Modified")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return code, None, None
        raise


def entries(code, size):
    z = zipfile.ZipFile(io.BufferedReader(RangeReader(f"{BASE}{code}_gml.zip", size), buffer_size=1 << 16))
    out = []
    for i in z.infolist():
        m = NAME.search(i.filename)
        if i.is_dir() or not m:
            continue
        if m.group(1) != code:
            print(f"  {code}: {i.filename} names powiat {m.group(1)}", file=sys.stderr)
        out.append((m.group(2), [code, i.header_offset, i.compress_size, i.file_size, i.CRC, i.compress_type]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "server", "data",
                                                  "pl_lod2_sheets.json.gz"))
    a = ap.parse_args()
    # TERYT powiat codes: voivodeships 02..32 (even), land powiats 01..35, city powiats 61..65
    codes = [f"{w:02d}{p:02d}" for w in range(2, 33, 2) for p in list(range(1, 36)) + list(range(61, 66))]
    with cf.ThreadPoolExecutor(4) as ex:
        found = [(c, n, lm) for c, n, lm in ex.map(head, codes) if n]
    print(f"{len(found)} powiats with LoD2 of {len(codes)} codes asked", file=sys.stderr)
    zips, sheets = {}, {}
    with cf.ThreadPoolExecutor(4) as ex:
        for (code, size, lm), ents in zip(found, ex.map(lambda f: entries(f[0], f[1]), found)):
            zips[code] = {"size": size, "modified": lm, "entries": len(ents)}
            for sheet, e in ents:
                sheets.setdefault(sheet, []).append(e)
            print(f"  {code}: {len(ents)} sheets, {size / 1e6:.1f} MB", file=sys.stderr)
    doc = {"format": 1, "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
           "base": BASE, "licence": "CC BY 4.0", "attribution": "GUGiK (Główny Urząd Geodezji i Kartografii)",
           "fields": ["teryt", "offset", "compressed", "size", "crc32", "method"],
           "zips": dict(sorted(zips.items())), "sheets": {k: sorted(v) for k, v in sorted(sheets.items())}}
    raw = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode()
    with open(a.out, "wb") as f:
        f.write(gzip.compress(raw, 9, mtime=0))
    print(f"{len(sheets)} sheets, {sum(len(v) for v in sheets.values())} entries -> {a.out} "
          f"({os.path.getsize(a.out) / 1e3:.0f} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
