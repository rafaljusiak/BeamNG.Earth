# Poland's LoD2 building model index

`pl_lod2_sheets.json.gz` lists where GUGiK's CityGML LoD2 3D building models of Poland are: for each 1:5,000 map sheet
that has models, the powiats (TERYT codes) whose zip holds a file for it, and that file's place in the zip (offset,
compressed and full size, CRC-32, compression method), as published at
`https://opendata.geoportal.gov.pl/InneDane/Budynki3D/LOD2/<TERYT>_gml.zip` (read 2026-11-13: 236 powiats, 38,913
sheets). It holds no model data: the server downloads a sheet's file when a player first drives near it
(`server/pl_lod2.py`, on by default, `BNE_PL_LOD2=off` to turn off).

The models: Główny Urząd Geodezji i Kartografii (GUGiK), "Modele 3D budynków",
https://www.geoportal.gov.pl/pl/dane/inne-dane/modele-3d-budynkow/ , licensed CC BY 4.0. Attribution: "GUGiK 3D
building models (CC BY 4.0)".

Rebuild: `python tools/build_pl_lod2_index.py` (HEAD requests for every possible powiat code, then each zip's central
directory by HTTP Range requests).

`pl_lod1_sheets.json.gz` is the same for GUGiK's LoD1 2024 models (one block of measured height per building, from the
2023 laser scan), `https://opendata.geoportal.gov.pl/InneDane/Budynki3D/LOD1/2024/<voivodeship>/<TERYT>.zip`: all 380
powiats, 58,458 sheets (read 2026-11-13; `python tools/build_pl_lod2_index.py --lod1`). Same licence and attribution.
