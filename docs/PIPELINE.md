# The pipeline, as flow charts

Plain-text Mermaid: GitHub, VS Code and Obsidian draw it. To edit, change the text; to add a step, add a node and an
arrow. Every box names the code that does it; arrows carry the data handed on.

**Colours (who owns each step):**

| Colour | Owner |
|---|---|
| blue | ArgusSim |
| green | MESA_ARGUS (ours) |
| amber | hdps |
| red | stand-ins, where hdps or Argus has no piece yet |
| grey | outputs |

Charts:
1. Overview
2. One exposure: from the sky to a raw frame
3. hdps: the Task chain on each frame
4. The reference night
5. Outputs, truth and real/bogus training

## 1. Overview

```mermaid
flowchart TD
  classDef argus fill:#dbeafe,stroke:#1e40af,color:#0b1d4a
  classDef ours fill:#dcfce7,stroke:#166534,color:#052e16
  classDef hdps fill:#fde68a,stroke:#92400e,color:#3b1d02
  classDef toy fill:#fecaca,stroke:#991b1b,color:#450a0a
  classDef out fill:#e5e7eb,stroke:#374151,color:#111827

  LIB["star library<br/><small>make_stars.py · stars/lib*, libnm* (newton)</small>"]:::ours
  OBS["observation file<br/><small>schema.py · runconfig.py: every parameter, limits, why</small>"]:::ours
  SCENE["scene: field stars, placed + injected stars, dust<br/><small>observe.py · gaia.py · dust.py</small>"]:::ours
  SURVEY["survey: which telescope sees the field when, band, tracking angle<br/><small>ArgusSim coverage_exact · mesa_argus/survey.py</small>"]:::argus
  REF["reference night<br/><small>observe.py make_reference → chart 4</small>"]:::ours
  EXP["each exposure → raw frame<br/><small>night_movie.py · sensor.py · camera.py → chart 2</small>"]:::ours
  HDPS["hdps worker on every frame<br/><small>hdps_worker.py · GPUWorker → chart 3</small>"]:::hdps
  MEAS["measurements from hdps products<br/><small>night_movie.py measure_frame, alert_table</small>"]:::ours
  OUT["catalogues · photometry · alerts.csv · Avro packets<br/>light curves · movies"]:::out
  TRAIN["real/bogus training<br/><small>rb_training_set.py · train_rb.py → chart 5</small>"]:::ours

  LIB --> SCENE
  OBS --> SCENE
  OBS --> SURVEY
  SCENE --> REF
  SCENE --> EXP
  SURVEY --> EXP
  REF -->|"reference coadds per tile"| HDPS
  EXP -->|"raw FITS per exposure"| HDPS
  HDPS -->|"solved FITS · tiles · alert_catalog · .avro"| MEAS
  MEAS --> OUT
  OUT -->|"labelled alerts"| TRAIN
  TRAIN -.->|"model_bundle"| HDPS
```

## 2. One exposure: from the sky to a raw frame

```mermaid
flowchart LR
  classDef argus fill:#dbeafe,stroke:#1e40af,color:#0b1d4a
  classDef ours fill:#dcfce7,stroke:#166534,color:#052e16
  classDef toy fill:#fecaca,stroke:#991b1b,color:#450a0a

  SED["each star's SED now<br/><small>MESA RSP · PHOEBE · sncosmo · templates</small>"]:::ours
  PHOT["electrons in the band at this airmass<br/><small>ArgusSim ABPhot · SystemThroughput</small>"]:::argus
  SKY["sky + dark, seeing<br/><small>ArgusSim sky model · seasonal seeing</small>"]:::argus
  GEOM["star → sensor pixel<br/><small>sensor.Projector: CIRS → array frame → focal plane</small>"]:::ours
  LAYOUT["telescope layout and axes<br/><small>ArgusSim Layout</small>"]:::argus
  WIN["readout window, N × N minipix<br/><small>sensor.window · field.window</small>"]:::ours
  PSF["PSF at each star's field angle<br/><small>ArgusSim ZemaxPSF.with_seeing</small>"]:::argus
  RENDER["image: PSFs + sky, Poisson, full well, read noise<br/><small>sensor.render</small>"]:::ours
  CAM["raw frame: uint16, AEGIS header keys<br/><small>camera.py · toy gain, no bias</small>"]:::toy

  SED --> PHOT --> RENDER
  SKY --> RENDER
  LAYOUT --> GEOM --> WIN --> RENDER
  PSF --> RENDER
  RENDER -->|"electrons"| CAM
```

## 3. hdps: the Task chain on each frame

All of it is hdps's own code, run by its `GPUWorker` on the CPU through the `hdps_full_cpu` branch. The chain is
`[[hdps.tasks]]` in the observation file.

```mermaid
flowchart TD
  classDef hdps fill:#fde68a,stroke:#92400e,color:#3b1d02
  classDef toy fill:#fecaca,stroke:#991b1b,color:#450a0a
  classDef out fill:#e5e7eb,stroke:#374151,color:#111827

  MAN["ratchet manifests, one source per camera<br/><small>group_images_by_ratchet_camera · ManifestSource</small>"]:::hdps
  VALKEY["valkey cache: WCS per camera and ratchet<br/><small>fakeredis</small>"]:::toy
  T1["DataCleanup → ImageReduction"]:::hdps
  T2["SourceDetection → SourceFilter"]:::hdps
  T3["ReferenceRetrieval (ATLAS-Refcat2)"]:::hdps
  T4["LinearWCSSolver → SourceMatching → TPVWCSSolver<br/><small>+ zero point coadd_magzp</small>"]:::hdps
  T5["LightweightFITSOutput"]:::hdps
  T6["Reprojection onto hdps tiles"]:::hdps
  T7["CoaddTap: ratchet coadd per tile"]:::hdps
  T8["SFFTSubtraction vs reference coadd<br/><small>mask_no_coverage_in_fit · mask_bright_stars</small>"]:::hdps
  T9["DiffDetection → TransientOutput"]:::hdps
  T10["AlertVetting: detect, features, real/bogus score"]:::hdps
  RB["real/bogus model<br/><small>AllReal (score 1) or a trained RF bundle</small>"]:::toy
  KAFKA["Kafka stream<br/><small>Avro files: alert_files.py</small>"]:::toy
  T11["TileImageOutput"]:::hdps
  P1["solved FITS: SOURCES, WCS, magzp"]:::out
  P2["ratchet coadds"]:::out
  P3["transient_catalog"]:::out
  P4["alert_catalog + alert_stamps"]:::out
  P5["science + difference tiles"]:::out

  MAN --> T1 --> T2 --> T3 --> T4 --> T5 --> T6 --> T7 --> T8 --> T9 --> T10 --> T11
  VALKEY <-.-> T4
  RB -.-> T10
  T10 -.->|"packets"| KAFKA
  T5 --> P1
  T7 -->|"finalize at ratchet change"| P2
  T9 --> P3
  T10 --> P4
  T11 --> P5
```

## 4. The reference night

```mermaid
flowchart LR
  classDef ours fill:#dcfce7,stroke:#166534,color:#052e16
  classDef hdps fill:#fde68a,stroke:#92400e,color:#3b1d02
  classDef out fill:#e5e7eb,stroke:#374151,color:#111827

  RN["the night before: same field and stars,<br/>transients and injection left out<br/><small>observe.py make_reference</small>"]:::ours
  RX["exposures → raw frames<br/><small>chart 2</small>"]:::ours
  RH["hdps worker up to CoaddTapTask<br/><small>hdps_worker.run(stop_after)</small>"]:::hdps
  RC["ratchet coadds per (camera, ratchet, tile)<br/><small>reference/hdps/&lt;band&gt;/coadds</small>"]:::out
  PICK["one coadd per tile: the deepest<br/><small>CoaddReferenceProvider</small>"]:::hdps

  RN --> RX --> RH --> RC --> PICK -->|"SFFT reference_dir"| SCI["science night, chart 3"]
```

## 5. Outputs, truth and real/bogus training

```mermaid
flowchart TD
  classDef ours fill:#dcfce7,stroke:#166534,color:#052e16
  classDef hdps fill:#fde68a,stroke:#92400e,color:#3b1d02
  classDef out fill:#e5e7eb,stroke:#374151,color:#111827

  SOLVED["solved FITS"]:::out
  ALERTS["alert_catalog · stamps · .avro"]:::out
  TILES["science + difference tiles"]:::out
  CAT["per-frame catalogues: mag = coadd_magzp − 2.5 log10 flux<br/><small>measure_frame</small>"]:::ours
  TRUTH["truth labels: simulated star, class, true flux change<br/><small>pipeline/truth.py</small>"]:::ours
  LC["catalogue.csv + light curves<br/><small>pipeline/lightcurves.py</small>"]:::ours
  AT["&lt;name&gt;_night_&lt;band&gt;.alerts.csv<br/><small>alert_table + label_alerts</small>"]:::out
  MOV["field movie + pipeline movie<br/><small>night_movie.py · pipeline_movie.py</small>"]:::out
  INJ["many injected observations<br/><small>make_training_data.py</small>"]:::ours
  SET["training_set.parquet<br/><small>rb_training_set.py</small>"]:::ours
  RF["random forest → hdps bundle<br/><small>train_rb.py</small>"]:::ours
  USE["AlertVettingTask model_bundle, score_cut fa10"]:::hdps

  SOLVED --> CAT --> LC
  TRUTH --> CAT
  ALERTS --> AT
  TRUTH --> AT
  TILES --> MOV
  CAT --> MOV
  INJ --> AT
  AT --> SET --> RF --> USE
```

## Editing

- **A new step:** add a node with the right colour class, and arrows in and out. Write the code in `<small>`, and
  the data handed on as the arrow's label.
- **A stand-in replaced by the real thing** (a trained model, real camera electronics): change its class from `toy`
  to `hdps` or `ours`.
- **Keep the overview small.** Detail goes in the stage charts.
