"""Experiment 2 (language ID) capture helper: one rep at a time, questions spoken by TTS.

    python lang_capture.py setup            one-time: speaker ID, Wi-Fi, interface, TTS voice
    python lang_capture.py tts              make (and optionally play) the 20 audio clips
    python lang_capture.py run --speaker 1 --rep 1    record one rep: 10 questions in each language
    python lang_capture.py status           progress for every speaker and rep

Each rep is one block per language (English, Spanish, Hindi), each shuffled
separately. The block order rotates every rep so time of day doesn't line up
with language. Before each block it tells you which Claude
voice language to set. The same edge-tts multilingual voice speaks both
languages, so within a speaker only the language changes. Each speaker slot
(S1/S2/S3) is a different TTS voice, set in exp2_config.json under "voices".

Run speaker 1, rep 1 as the test. If it checks out, do the rest the same way.
"""
import argparse
import asyncio
import csv
import datetime
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Reuse Experiment 3's capture engine and pcap checks rather than copying them.
sys.path.insert(0, str(ROOT.parent / "exp3_tool_use"))
import capture as exp3   # noqa: E402
import first_check       # noqa: E402
import pcaplib           # noqa: E402

from questions import LANG_NAME, LANGS, QUESTIONS, VOICE_SETTING, question_text  # noqa: E402

CONFIG_PATH = ROOT / "exp2_config.json"
PLANS_DIR = ROOT / "plans"
TTS_DIR = ROOT / "tts"
REPS = 5
SILENCE_BEFORE = 3       # seconds of silence after voice mode starts listening

MANIFEST_COLS = exp3.common.MANIFEST_COLS
PLAN_COLS = ["seq", "file", "lang", "prompt_id", "rep", "prompt", "status"]

DEFAULT_CONFIG = {
    "network": "",
    "client_version": "",
    "pilot_protocol": "",
    # Speaker slot -> edge-tts voice. Must be *Multilingual* voices, fixed once recording starts.
    "voices": {"1": "en-US-AndrewMultilingualNeural",
               "2": "en-US-AvaMultilingualNeural",
               "3": "en-US-BrianMultilingualNeural"},
    "tts_rate": "+0%",
    "play_volume": "1",
    "capture_dir": "captures",
    "manifest": "manifest_exp2.csv",
    "dumpcap": "",
    "interface": "",
    "interface_label": "",
}

ask, ask_choice, banner = exp3.ask, exp3.ask_choice, exp3.banner


# ---------- config and files ----------

def load_config(require=True):
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    elif require:
        raise SystemExit("No exp2_config.json yet. Run:  python lang_capture.py setup")
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


def make_name(lang, speaker, prompt_id, rep):
    return f"lang_{lang}_s{speaker}_p{prompt_id}_r{int(rep):02d}.pcapng"


# ---------- TTS ----------

def tts_path(voice, lang, prompt_id):
    return TTS_DIR / voice / f"{lang}_{prompt_id}.mp3"


def make_tts(cfg, voice):
    """Generate any missing clips. The same file is replayed for every rep."""
    try:
        import edge_tts
    except ImportError:
        raise SystemExit(f"edge-tts isn't installed for {sys.executable}. Run:  pip install edge-tts")

    async def gen(text, path):
        await edge_tts.Communicate(text, voice, rate=cfg["tts_rate"]).save(str(path))

    for pid in QUESTIONS:
        for lang in LANGS:
            path = tts_path(voice, lang, pid)
            if path.exists() and path.stat().st_size > 0:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  generating {path.name}: {question_text(lang, pid)}")
            # The free edge-tts service drops requests now and then; retry before giving up.
            for attempt in range(1, 6):
                try:
                    asyncio.run(gen(question_text(lang, pid), path))
                    break
                except Exception as e:
                    path.unlink(missing_ok=True)
                    if attempt == 5:
                        raise SystemExit(f"TTS failed for {path.name}: {e}")
                    time.sleep(2 * attempt)


def play(cfg, path):
    subprocess.run(["afplay", "-v", str(cfg["play_volume"]), str(path)], check=True)


# ---------- plan ----------

def plan_path(rep, speaker):
    return PLANS_DIR / f"rep{rep}_s{speaker}.csv"


def build_plan(rep, speaker, rng):
    k = (rep - 1) % len(LANGS)
    order = LANGS[k:] + LANGS[:k]
    items = []
    for lang in order:
        block = list(QUESTIONS)
        rng.shuffle(block)
        for pid in block:
            items.append({"file": make_name(lang, speaker, pid, rep), "lang": lang,
                          "prompt_id": pid, "rep": rep,
                          "prompt": question_text(lang, pid), "status": "pending"})
    for i, it in enumerate(items, 1):
        it["seq"] = i
    return items


def read_plan(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_plan(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PLAN_COLS)
        w.writeheader()
        for it in items:
            w.writerow({c: it[c] for c in PLAN_COLS})


def ensure_plan(rep, speaker):
    path = plan_path(rep, speaker)
    if not path.exists():
        write_plan(path, build_plan(rep, speaker, random.Random()))
    return path, read_plan(path)


def append_manifest(cfg, row):
    path = _resolve(cfg["manifest"])
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in MANIFEST_COLS})


# ---------- setup / tts ----------

def cmd_setup(_args):
    cfg = load_config(require=False)
    banner("Experiment 2 setup (do once, then keep fixed)")
    cfg["network"] = ask("Home Wi-Fi name or location", cfg["network"])
    cfg["client_version"] = ask("Chrome version (open chrome://version)", cfg["client_version"])
    cfg["pilot_protocol"] = ask("Protocol of the pilot's voice flow (TCP / UDP / blank if unsure)",
                                cfg["pilot_protocol"]).upper()
    print("\nThe TTS voice must speak both languages, so use a *Multilingual* one.")
    print("List them with:  edge-tts --list-voices | grep Multilingual")
    for spk in ("1", "2", "3"):
        cfg["voices"][spk] = ask(f"TTS voice for speaker S{spk}", cfg["voices"][spk])
    cfg["tts_rate"] = ask("TTS speaking rate (e.g. +0%, -10%)", cfg["tts_rate"])
    cfg["play_volume"] = ask("Playback volume for afplay (0-1+)", cfg["play_volume"])

    cfg["dumpcap"] = cfg["dumpcap"] or exp3.find_dumpcap()
    cfg["dumpcap"] = ask("Path to dumpcap (comes with Wireshark)", cfg["dumpcap"])
    ifaces = exp3.list_interfaces(cfg["dumpcap"]) if cfg["dumpcap"] else []
    if ifaces:
        print("\nInterfaces:")
        for i, (name, label) in enumerate(ifaces, 1):
            print(f"  {i:>2}. {name}  {label}")
        pick = ask("Number of your Wi-Fi adapter (usually en0 on a Mac)")
        if pick.isdigit() and 1 <= int(pick) <= len(ifaces):
            cfg["interface"], cfg["interface_label"] = ifaces[int(pick) - 1]
    else:
        print("dumpcap listed no interfaces. On macOS, install Wireshark's ChmodBPF package so "
              "dumpcap can capture without sudo.")
    save_config(cfg)
    print(f"\nSaved {CONFIG_PATH.name}. Next:  python lang_capture.py tts")


def cmd_tts(args):
    cfg = load_config()
    voice = cfg["voices"][str(args.speaker)]
    make_tts(cfg, voice)
    print(f"All {len(QUESTIONS) * len(LANGS)} clips for S{args.speaker} are in {TTS_DIR / voice}")
    if args.play:
        for pid in QUESTIONS:
            for lang in LANGS:
                print(f"  {lang} {pid}: {question_text(lang, pid)}")
                play(cfg, tts_path(voice, lang, pid))
                time.sleep(0.5)


# ---------- one capture ----------

REDO_RULES = """  Redo if: the clip was cut off or Claude cut in / Claude didn't hear the
  question / anyone else spoke / voice mode errored, disconnected or reconnected."""


def record_one(item, idx, total, cfg, speaker, sitting):
    path = capture_dir(cfg) / item["file"]
    banner(f"[{idx}/{total}]  {item['file']}   (voice language: {VOICE_SETTING[item['lang']]})")
    print(f'\n  TTS will say:  "{item["prompt"]}"\n')
    if path.exists():
        if ask_choice("  This file already exists. Overwrite it? [y/n]: ", ("y", "n")) == "n":
            return "skip"

    print("  1. Open a NEW chat in Claude (never reuse one).")
    key = ask_choice("  2. Press Enter to start the capture  (s = skip, q = quit): ", ("", "s", "q"))
    if key == "s":
        return "skip"
    if key == "q":
        return "quit"

    cap = exp3.Capture(cfg["dumpcap"], cfg["interface"], path)
    started = datetime.datetime.now()
    try:
        cap.start()
    except RuntimeError as e:
        print(f"\n  {e}")
        return "quit"
    try:
        print("     Capturing.")
        input("  3. Click the voice button. Press Enter once the listening indicator shows: ")
        print(f"     {SILENCE_BEFORE} s of silence...")
        time.sleep(SILENCE_BEFORE)
        print("     Playing the question.")
        play(cfg, tts_path(cfg["voices"][speaker], item["lang"], item["prompt_id"]))
        print("  4. Stay quiet while Claude replies. When it finishes, wait 3 s, end voice mode.")
        input("  Press Enter once you've ended voice mode to stop the capture: ")
    finally:
        cap.stop()

    if not path.exists():
        print("  No file was written. Check the interface in setup, then redo.")
        return "redo"
    size = path.stat().st_size
    try:
        _, dur = pcaplib.capture_stats(path)
    except Exception:
        dur = 0.0
    print(f"\n  Saved {path.name}: {size / 1024:.0f} KB, {dur:.0f} s")
    if size < exp3.common.MIN_BYTES or dur < exp3.common.MIN_SECONDS:
        print("  WARNING: under 50 KB or 10 s. That usually means a broken capture.")
    print(REDO_RULES)
    if ask_choice("  Keep it? [Enter = keep / r = redo (deletes the file)]: ", ("", "r")) == "r":
        path.unlink(missing_ok=True)
        return "redo"

    lang_name = LANG_NAME[item["lang"]]
    # Wrong-language replies are kept and noted, per the protocol.
    right = ask_choice(f"  Did Claude reply in {lang_name}? [Enter = yes / n = no]: ", ("", "n"))
    while True:
        words = ask("  Rough word count of Claude's reply")
        if words.isdigit():
            break
        print("  Please enter a number.")
    notes = ask("  Notes (Enter for none)")

    append_manifest(cfg, {
        "file": item["file"],
        "experiment": "lang",
        "class": item["lang"],
        "prompt_id": item["prompt_id"],
        "speaker": f"S{speaker}",
        "rep": item["rep"],
        "date_time": started.strftime("%Y-%m-%d %H:%M:%S"),
        "network": cfg["network"],
        "client_version": cfg["client_version"],
        "reply_words": words,
        "notes": "; ".join(x for x in (
            f"sitting {sitting}",
            f"voice language setting: {VOICE_SETTING[item['lang']]}",
            f"tts: {cfg['voices'][speaker]} {cfg['tts_rate']}",
            "" if right == "" else f"Claude replied in the wrong language (not {lang_name})",
            notes) if x),
    })
    return "done"


# ---------- run a rep ----------

def cmd_run(args):
    cfg = load_config()
    if not (cfg["dumpcap"] and cfg["interface"]):
        raise SystemExit("No dumpcap/interface in setup. Re-run:  python lang_capture.py setup")
    speaker = str(args.speaker)
    make_tts(cfg, cfg["voices"][speaker])
    path, items = ensure_plan(args.rep, speaker)
    pending = [it for it in items if it["status"] != "done"]
    banner(f"Experiment 2, rep {args.rep}, speaker S{speaker} ({cfg['voices'][speaker]}): "
           f"{len(items) - len(pending)}/{len(items)} done")
    if not pending:
        print("This rep is complete.")
        return

    cfg["client_version"] = ask("Chrome version today (chrome://version)", cfg["client_version"])
    save_config(cfg)
    print("""
Before you start:
  - Laptop plugged in, on your usual home Wi-Fi.
  - Mac output = speakers at a fixed volume; Chrome's mic = MacBook Pro Microphone.
    (The question plays through the speakers into the mic.)
  - Quiet room. Phone silent and away. Don't speak during captures.
  - Other tabs closed; sync/chat apps closed; updates paused.
  - Claude web app in Chrome, your own account, same Claude voice as always.""")
    input("Press Enter when all of that is true: ")

    prev_lang = None
    for idx, item in enumerate(items, 1):
        if item["status"] == "done":
            prev_lang = item["lang"]
            continue
        if item["lang"] != prev_lang:
            banner(f"SET Claude Settings > General > Voice > Language: {VOICE_SETTING[item['lang']]}")
            input("Press Enter once it's set: ")
            prev_lang = item["lang"]

        while True:
            result = record_one(item, idx, len(items), cfg, speaker, args.sitting)
            if result != "redo":
                break
            print("  Redo: start over with a new chat.")
        if result == "quit":
            break
        if result == "skip":
            print("  Skipped; it stays pending for later.")
            continue

        item["status"] = "done"
        write_plan(path, items)
        if sum(1 for it in items if it["status"] == "done") == 1:
            ok = first_check.print_report(capture_dir(cfg) / item["file"], cfg["pilot_protocol"])
            if ask_choice("  Does it look right? [Enter = yes / n = stop]: ", ("", "n")) == "n" or not ok:
                print("  Stopping here. Fix the setup before recording more.")
                break

    remaining = sum(1 for it in items if it["status"] != "done")
    if remaining:
        print(f"\n{remaining} captures still pending in rep {args.rep}. Run the same command to resume.")
    else:
        banner(f"Rep {args.rep} complete")
        print("Set Claude's voice language back to English if you're done for now.")


def cmd_status(_args):
    for spk in ("1", "2", "3"):
        for rep in range(1, REPS + 1):
            p = plan_path(rep, spk)
            if not p.exists():
                print(f"S{spk} rep {rep}: not started ({len(QUESTIONS) * len(LANGS)} captures)")
                continue
            items = read_plan(p)
            print(f"S{spk} rep {rep}: {sum(1 for it in items if it['status'] == 'done')}/{len(items)} done")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(fn=cmd_setup)
    t = sub.add_parser("tts")
    t.add_argument("--speaker", type=int, choices=[1, 2, 3], default=1)
    t.add_argument("--play", action="store_true", help="play every clip so you can check it")
    t.set_defaults(fn=cmd_tts)
    r = sub.add_parser("run")
    r.add_argument("--speaker", type=int, choices=[1, 2, 3], required=True)
    r.add_argument("--rep", type=int, choices=range(1, REPS + 1), required=True)
    r.add_argument("--sitting", type=int, default=1, help="sitting number, written to the notes")
    r.set_defaults(fn=cmd_run)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    args = ap.parse_args()
    try:
        args.fn(args)
    except KeyboardInterrupt:
        print("\nStopped. Finished captures are saved; run the same command to resume.")
        sys.exit(130)


if __name__ == "__main__":
    main()
