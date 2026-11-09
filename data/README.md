# Data

Raw data is **not** committed (see `.gitignore`).

## Synthetic demo data (no download)

```bash
iotids synth --out data/synthetic
```

CIC-IoT2023-style columns and labels, three benign device profiles, seven attack families and natural drift at `ts = 0.6`.
Use it for demos, tests and CI only.

## CIC-IoT2023 (main dataset)

- Source: Canadian Institute for Cybersecurity, University of New Brunswick: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- Download the **CSV** folder (≈ 169 files `part-00000-...csv`, 46 features + `label`), then put the files here:

```
data/CICIoT2023/part-00000-363d1ba3-....csv
data/CICIoT2023/part-00001-....csv
...
```

- 34 labels are mapped to 8 families automatically (`src/iotids/data.py`):
  `Benign, DDoS, DoS, Mirai, Recon, Spoofing, BruteForce, Web`.
- Config: `configs/default.yaml`. Start with `max_files: 20`, `sample_frac: 0.25`.

## Other datasets (generic CSV)

Any flow CSV with a label column works. Set it up in a config:

```yaml
data:
  path: data/TON_IoT
  label_col: type            # name of the label column
  benign_values: [normal]    # labels that mean benign
  family_map: {scanning: Recon, ddos: DDoS, injection: Web, xss: Web, password: BruteForce}
  drop_cols: [src_ip, dst_ip, src_port, dst_port, ts]   # identifiers -> drop
  time_col: ts               # if present -> natural drift in the stream experiment
```

Suggested: **TON_IoT** (network), **N-BaIoT** (per-device benign/Mirai/Gafgyt), **Edge-IIoTset**.

Cite each dataset you use in your README or report.
