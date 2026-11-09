"""Builds notebooks/01_quickstart.ipynb (run: python scripts/make_notebook.py)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Self-Supervised, Drift-Aware IoT Intrusion Detection — Quickstart\n\n"
       "This notebook walks through the full pipeline on **synthetic CIC-IoT2023-like data** so it runs in a few "
       "minutes on a laptop. Switch to `configs/default.yaml` once you have downloaded the real dataset.\n\n"
       "**Protocol:** models see *benign traffic only*. Attack families `Mirai, Web, BruteForce, Spoofing` are "
       "held out completely → the test measures detection of **unknown (zero-day) attacks**."),
    code("import os, sys, json\nfrom pathlib import Path\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "os.chdir(ROOT); sys.path.insert(0, str(ROOT / 'src'))\n"
         "import pandas as pd, numpy as np, matplotlib.pyplot as plt\n"
         "from IPython.display import Image, display\n"
         "from iotids.config import load_config\nfrom iotids.data import load_dataset, make_splits\n"
         "from iotids.pipeline import run_batch, run_stream\nfrom iotids.synthetic import write_csvs\n"
         "pd.set_option('display.precision', 4)"),
    md("## 1. Data"),
    code("if not Path('data/synthetic').exists():\n    write_csvs('data/synthetic')\n"
         "cfg = load_config('configs/synthetic.yaml')\ndf = load_dataset(cfg)\n"
         "print(df.shape)\ndf['family'].value_counts().plot.barh(title='Rows per family', figsize=(6, 3));"),
    code("feats = ['Rate', 'Tot size', 'syn_flag_number', 'IAT']\n"
         "fig, axes = plt.subplots(1, 4, figsize=(15, 3))\n"
         "for ax, f in zip(axes, feats):\n"
         "    for fam, g in df.groupby('family'):\n"
         "        ax.hist(np.log1p(g[f].replace(np.inf, np.nan).dropna()), bins=60, histtype='step', density=True, label=fam)\n"
         "    ax.set_title(f'log1p({f})')\naxes[0].legend(fontsize=7); plt.tight_layout()"),
    md("## 2. Zero-day split"),
    code("sp = make_splits(df, cfg)\npd.DataFrame(sp.summary()).fillna(0).astype(int)"),
    md("## 3. Train all detectors & evaluate on unknown attack families"),
    code("run = run_batch(cfg)"),
    code("m = pd.read_csv(run / 'metrics.csv')\n"
         "m[['model','eval','roc_auc','pr_auc','tpr@fpr=1%','precision','recall','f1','fpr']]"),
    code("for f in ['roc_zero_day.png', 'per_family_detection.png', 'per_family_auc.png', 'training_curves.png']:\n"
         "    display(Image(run / 'figures' / f))"),
    md("## 4. Concept drift: static vs self-adapting detector\n\n"
       "Benign traffic shifts at `ts = 0.6` (think: firmware update). The static model's false-positive rate explodes; "
       "the adaptive model detects the drift with ADWIN (label-free), fine-tunes on likely-benign recent traffic "
       "plus a replay buffer, and re-calibrates its threshold."),
    code("srun = run_stream(cfg, 'ae')\ndisplay(pd.read_csv(srun / 'stream_summary.csv'))\n"
         "display(Image(srun / 'figures' / 'stream_static_vs_adaptive.png'))"),
    md("## 5. Next steps\n- Download CIC-IoT2023 (see `data/README.md`) and run `iotids all --config configs/default.yaml`\n"
       "- Try other `unknown_families` choices (leave-one-family-out)\n- `streamlit run app/streamlit_app.py` for the dashboard"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "notebooks/01_quickstart.ipynb")
print("ok")
