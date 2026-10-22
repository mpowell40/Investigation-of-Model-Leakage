"""Shared settings, file naming and manifest handling for Experiment 3 (tool use)."""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "exp3_config.json"
PLANS_DIR = ROOT / "plans"

# Same columns as the team's shared manifest sheet, in the same order.
MANIFEST_COLS = [
    "file", "experiment", "class", "prompt_id", "speaker", "rep", "date_time",
    "network", "client_version", "search_setting", "search_observed",
    "reply_words", "notes",
]

DEFAULT_CONFIG = {
    "speaker": "",           # 1, 2 or 3 -> S1/S2/S3, assigned once, never changed
    "network": "",           # home Wi-Fi name
    "client_version": "",    # Chrome version
    "pilot_protocol": "",    # TCP or UDP: what the pilot's voice flow used
    "capture_dir": "captures",
    "manifest": "manifest_exp3.csv",
    "dumpcap": "",
    "interface": "",
    "interface_label": "",
}

MIN_BYTES = 50 * 1024    # nightly check: smaller files are probably broken
MIN_SECONDS = 10.0       # nightly check: shorter files are probably broken

FILE_RE = re.compile(
    r"^(?P<exp>tool|base)_(?:(?P<cls>search|static|off)_)?"
    r"s(?P<spk>\d+)_p(?P<pid>[TB]\d\d)_r(?P<rep>\d\d)\.pcapng$"
)


def load_config(require=True):
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    elif require:
        raise SystemExit("No exp3_config.json yet. Run:  python capture.py setup")
    return cfg


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def capture_dir(cfg):
    d = _resolve(cfg["capture_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_path(cfg):
    return _resolve(cfg["manifest"])


def make_name(cls, speaker, prompt_id, rep):
    if cls == "base":
        return f"base_s{speaker}_p{prompt_id}_r{int(rep):02d}.pcapng"
    return f"tool_{cls}_s{speaker}_p{prompt_id}_r{int(rep):02d}.pcapng"


def parse_name(name):
    """Return the fields encoded in a capture file name, or None if it breaks the pattern."""
    m = FILE_RE.match(name)
    if not m:
        return None
    exp, cls, pid = m["exp"], m["cls"], m["pid"]
    if exp == "base":
        if cls is not None or not ("B01" <= pid <= "B04"):
            return None
        cls = "base"
    elif cls is None or not ("T01" <= pid <= "T08"):
        return None
    return {"experiment": exp, "class": cls, "speaker": m["spk"],
            "prompt_id": pid, "rep": int(m["rep"])}


def read_manifest(path):
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def append_manifest(path, row):
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in MANIFEST_COLS})
