# Real_Terrain

Fire cases on real elevation data at the scale of an event, with the wind driven through the domain by mass inflow and pressure outflow rather than recycled. Each case builds its own inputs from a public elevation source with a generator script kept next to the deck.

| Subdirectory | Description | Character |
|---|---|---|
| `Marshall_Fire` | The 30 December 2021 Marshall Fire windstorm, Boulder County: 25.6 km of SRTM1 terrain at 100 m, a compressible atmosphere with a 10 m first cell, 35 m/s westerlies, three ignitions on terrain of different curvature, spotting, 1024 x 1024 fire cells. | demonstration |
| `Palisades_Fire` | The NLR Palisades stress test, 7 January 2025: 24 km of SRTM1 terrain, a 300 m atmosphere over a 30 m fire grid with its own finer elevation model, a southwesterly at 38 mph, chaparral to the shoreline, two-way coupling, 800 x 800 fire cells. | demonstration |
