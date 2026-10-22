"""Experiment 3 capture helper: walks you through a sitting one capture at a time.

    python capture.py setup                 one-time: speaker ID, Wi-Fi, interface
    python capture.py run --sitting 1       record a sitting (resumes where you left off)
    python capture.py run --sitting 1 --manual   use the Wireshark window yourself
    python capture.py status                progress across all three sittings

For each capture it shows the exact words to say, starts and stops the capture
(dumpcap, Wireshark's own capture engine, interface = your Wi-Fi, filter
"port 443", saved as .pcapng under the correct file name), then asks for
search_observed / reply_words / notes and writes the manifest row right away.
"""
import argparse
import datetime
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import common
import first_check
import pcaplib
import plan_sitting

CAPTURE_FILTER = "port 443"


# ---------- small console helpers ----------

def ask(prompt, default=""):
    shown = f"{prompt} [{default}]: " if default else f"{prompt}: "
    value = input(shown).strip()
    return value or default


def ask_choice(prompt, choices):
    while True:
        value = input(prompt).strip().lower()
        if value in choices:
            return value
        print(f"  Please type one of: {', '.join(c or 'Enter' for c in choices)}")


def banner(text):
    line = "=" * 72
    print(f"\n{line}\n{text}\n{line}")


# ---------- dumpcap ----------

def find_dumpcap():
    found = shutil.which("dumpcap")
    if found:
        return found
    for c in (r"C:\Program Files\Wireshark\dumpcap.exe",
              r"C:\Program Files (x86)\Wireshark\dumpcap.exe",
              "/Applications/Wireshark.app/Contents/MacOS/dumpcap",
              "/usr/bin/dumpcap"):
        if os.path.exists(c):
            return c
    return ""


def list_interfaces(dumpcap):
    out = subprocess.run([dumpcap, "-D"], capture_output=True, text=True, timeout=30)
    result = []
    for line in out.stdout.splitlines():
        if ". " not in line:
            continue
        rest = line.split(". ", 1)[1].strip()
        name, label = rest, ""
        if rest.endswith(")") and " (" in rest:
            name, label = rest[:-1].rsplit(" (", 1)
        result.append((name, label))
    return result


class Capture:
    def __init__(self, dumpcap, interface, path):
        self.args = [dumpcap, "-q", "-i", interface, "-f", CAPTURE_FILTER, "-w", str(path)]
        self.log_path = Path(path).with_suffix(".dumpcap.log")
        self.proc = None

    def start(self):
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.args, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=self.log,
                                     creationflags=flags)
        time.sleep(2)                       # let it open the interface
        if self.proc.poll() is not None:
            self.log.close()
            msg = self.log_path.read_text(errors="replace").strip()
            self.log_path.unlink(missing_ok=True)
            raise RuntimeError(f"dumpcap stopped right away:\n{msg}")

    def stop(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        # A clean stop lets dumpcap finish writing the pcapng.
        self.proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            print("  (dumpcap had to be force-stopped; the nightly check will flag the file if it's damaged)")
        self.log.close()
        self.log_path.unlink(missing_ok=True)


# ---------- setup ----------

def cmd_setup(_args):
    cfg = common.load_config(require=False)
    banner("Experiment 3 setup (do once, then keep fixed)")
    print("Speaker IDs are assigned once by the team and never change (S1, S2, S3).")
    spk = ask("Your speaker number (1, 2 or 3)", str(cfg["speaker"])).lstrip("sS")
    if spk not in ("1", "2", "3"):
        raise SystemExit("Speaker must be 1, 2 or 3.")
    cfg["speaker"] = spk
    cfg["network"] = ask("Home Wi-Fi name or location", cfg["network"])
    cfg["client_version"] = ask("Chrome version (open chrome://version)", cfg["client_version"])
    cfg["pilot_protocol"] = ask("Protocol of the pilot's voice flow (TCP / UDP / blank if unsure)",
                                cfg["pilot_protocol"]).upper()
    cfg["capture_dir"] = ask("Folder for capture files", cfg["capture_dir"])
    cfg["manifest"] = ask("Manifest CSV", cfg["manifest"])

    cfg["dumpcap"] = cfg["dumpcap"] or find_dumpcap()
    cfg["dumpcap"] = ask("Path to dumpcap (comes with Wireshark)", cfg["dumpcap"])
    if cfg["dumpcap"] and os.path.exists(cfg["dumpcap"]):
        ifaces = list_interfaces(cfg["dumpcap"])
        if ifaces:
            print("\nInterfaces:")
            for i, (name, label) in enumerate(ifaces, 1):
                print(f"  {i:>2}. {label or name}")
            pick = ask("Number of your Wi-Fi adapter")
            if pick.isdigit() and 1 <= int(pick) <= len(ifaces):
                cfg["interface"], cfg["interface_label"] = ifaces[int(pick) - 1]
        else:
            print("dumpcap listed no interfaces. Is Npcap installed (it comes with Wireshark)?")
    else:
        print("dumpcap not found. You can still record with:  python capture.py run --sitting N --manual")

    common.save_config(cfg)
    print(f"\nSaved {common.CONFIG_PATH.name}. Captures go to {common.capture_dir(cfg)}")


# ---------- one capture ----------

PROCEDURE = """
  3. Click the voice button; wait for the listening indicator.
  4. Stay silent 3 seconds (count it in your head).
  5. Say the prompt once, normal pace and volume.
  6. Don't speak, cough or click while Claude replies.
  7. When the reply finishes, stay silent 3 more seconds.
  8. End voice mode."""

REDO_RULES = """  Redo if: you stumbled or misread / Claude cut in / Claude's voice reached
  the mic or anyone else spoke / voice mode errored, disconnected or reconnected."""


def record_one(item, idx, total, cfg, manual, sitting):
    path = common.capture_dir(cfg) / item["file"]
    banner(f"[{idx}/{total}]  {item['file']}   (web search {item['search_setting'].upper()})")
    print(f'\n  SAY:  "{item["prompt"]}"\n')

    if path.exists():
        if ask_choice("  This file already exists. Overwrite it? [y/n]: ", ("y", "n")) == "n":
            return "skip"

    print("  1. Open a NEW chat in Claude (never reuse one).")
    key = ask_choice("  2. Press Enter to start the capture  (s = skip, q = quit): ", ("", "s", "q"))
    if key == "s":
        return "skip"
    if key == "q":
        return "quit"

    cap = None
    started = datetime.datetime.now()
    if manual:
        print(f"     Start the Wireshark capture now (interface: Wi-Fi, filter: {CAPTURE_FILTER}).")
    else:
        cap = Capture(cfg["dumpcap"], cfg["interface"], path)
        try:
            cap.start()
        except RuntimeError as e:
            print(f"\n  {e}")
            return "quit"
        print("     Capturing.")
    print(PROCEDURE)
    try:
        input("\n  Press Enter once you've ended voice mode to stop the capture: ")
    finally:
        if cap:
            cap.stop()

    if manual:
        print(f"  9. Stop the Wireshark capture and save it as:\n     {path}")
        while not path.exists():
            if ask_choice("  Press Enter once it's saved (r = redo this capture): ", ("", "r")) == "r":
                return "redo"

    if not path.exists():
        print("  No file was written. Check the interface in setup, then redo.")
        return "redo"

    size = path.stat().st_size
    try:
        _, dur = pcaplib.capture_stats(path)
    except Exception:
        dur = 0.0
    print(f"\n  Saved {path.name}: {size / 1024:.0f} KB, {dur:.0f} s")
    if size < common.MIN_BYTES or dur < common.MIN_SECONDS:
        print("  WARNING: under 50 KB or 10 s. That usually means a broken capture.")
    print(REDO_RULES)
    if ask_choice("  Keep it? [Enter = keep / r = redo (deletes the file)]: ", ("", "r")) == "r":
        path.unlink(missing_ok=True)
        return "redo"

    print("\n  10. Check the chat transcript:")
    hint = {"search": "expected Y", "static": "expected N", "off": "search is off, expected N",
            "base": "expected N"}[item["class"]]
    observed = ask_choice(f"  Did the search indicator appear? ({hint}; log what happened) [y/n]: ",
                          ("y", "n")).upper()
    while True:
        words = ask("  Rough word count of Claude's reply")
        if words.isdigit():
            break
        print("  Please enter a number.")
    notes = ask("  Notes (Enter for none)")

    common.append_manifest(common.manifest_path(cfg), {
        "file": item["file"],
        "experiment": "base" if item["class"] == "base" else "tool",
        "class": item["class"],
        "prompt_id": item["prompt_id"],
        "speaker": f"S{cfg['speaker']}",
        "rep": item["rep"],
        "date_time": started.strftime("%Y-%m-%d %H:%M:%S"),
        "network": cfg["network"],
        "client_version": cfg["client_version"],
        "search_setting": item["search_setting"],
        "search_observed": observed,
        "reply_words": words,
        "notes": "; ".join(x for x in (f"sitting {sitting}", notes) if x),
    })
    return "done"


def take_break():
    print("\n  15 captures done: take a 2-minute break so your voice stays consistent.")
    try:
        for left in range(120, 0, -1):
            print(f"\r  {left:>3} s  (Ctrl+C to skip)", end="", flush=True)
            time.sleep(1)
        print()
    except KeyboardInterrupt:
        print()
    input("  Press Enter when ready to continue: ")


# ---------- run a sitting ----------

def cmd_run(args):
    cfg = common.load_config()
    if not args.manual and not (cfg["dumpcap"] and cfg["interface"]):
        raise SystemExit("No dumpcap/interface in setup. Re-run setup, or add --manual.")
    path, items, created = plan_sitting.ensure_plan(args.sitting, cfg["speaker"])
    pending = [it for it in items if it["status"] != "done"]
    banner(f"Experiment 3, sitting {args.sitting}, speaker S{cfg['speaker']}: "
           f"{len(items) - len(pending)}/{len(items)} done")
    if not pending:
        print("This sitting is complete. Run:  python check_files.py")
        return
    if args.sitting == 3:
        print("Sitting 3 must be on a different day from sittings 1 and 2 (it's the held-out test set).")

    cfg["client_version"] = ask("Chrome version today (chrome://version)", cfg["client_version"])
    common.save_config(cfg)
    print("""
Before you start:
  - Laptop plugged in, on your usual home Wi-Fi.
  - Wired headphones in. Phone silent and away from the laptop.
  - Quiet room, windows closed, no TV or music.
  - Other tabs closed; Dropbox / Slack / Spotify / sync apps closed; updates paused.
  - Claude web app in Chrome, your own account, same voice as always.""")
    input("Press Enter when all of that is true: ")

    done_now = 0
    prev_setting = None
    for idx, item in enumerate(items, 1):
        if item["status"] == "done":
            prev_setting = item["search_setting"]
            continue
        if item["search_setting"] != prev_setting:
            banner(f"SET WEB SEARCH {item['search_setting'].upper()} in Claude's settings now")
            input("Press Enter once it's set: ")
            prev_setting = item["search_setting"]

        while True:
            result = record_one(item, idx, len(items), cfg, args.manual, args.sitting)
            if result != "redo":
                break
            print("  Redo: start over with a new chat.")
        if result == "quit":
            break
        if result == "skip":
            print("  Skipped; it stays pending for later.")
            continue

        item["status"] = "done"
        plan_sitting.write_plan(path, items)
        done_now += 1

        if sum(1 for it in items if it["status"] == "done") == 1:
            ok = first_check.print_report(common.capture_dir(cfg) / item["file"],
                                          cfg.get("pilot_protocol", ""))
            print("\n  Compare with Wireshark if you like: Statistics > Conversations and I/O Graphs.")
            if ask_choice("  Does it look right? [Enter = yes / n = stop and tell the team]: ",
                          ("", "n")) == "n" or not ok:
                print("  Stopping here. Tell the team before recording more.")
                break
        if done_now % 15 == 0 and idx < len(items):
            take_break()

    remaining = sum(1 for it in items if it["status"] != "done")
    if remaining:
        print(f"\n{remaining} captures still pending in sitting {args.sitting}. "
              f"Run the same command to resume.")
    else:
        banner(f"Sitting {args.sitting} complete")
        if any(it["search_setting"] == "off" for it in items):
            print("Turn web search back ON in Claude's settings.")
        print("Next: python check_files.py   then commit and push to GitHub.")


def cmd_status(_args):
    cfg = common.load_config()
    for s in (1, 2, 3):
        p = plan_sitting.plan_path(s, cfg["speaker"])
        if not p.exists():
            print(f"Sitting {s}: not started ({plan_sitting.EXPECTED[s]} captures)")
            continue
        items = plan_sitting.read_plan(p)
        done = sum(1 for it in items if it["status"] == "done")
        print(f"Sitting {s}: {done}/{len(items)} done")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(fn=cmd_setup)
    r = sub.add_parser("run")
    r.add_argument("--sitting", type=int, choices=[1, 2, 3], required=True)
    r.add_argument("--manual", action="store_true",
                   help="start/stop/save in the Wireshark window yourself")
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
