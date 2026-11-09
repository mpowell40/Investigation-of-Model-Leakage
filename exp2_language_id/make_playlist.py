"""Join all 10 questions of each language into one audio file, with a gap after each.

    python make_playlist.py                     english/spanish/hindi_all_s1.mp4, 25 s gaps
    python make_playlist.py --gap 30 --speaker 2

Output is AAC audio in an .mp4 file (plays on iPhone, Mac, anywhere), in playlists/.
Uses the same TTS clips as lang_capture.py (generated if missing). macOS only:
needs afconvert, which ships with the OS.
"""
import argparse
import subprocess
import tempfile
import wave
from pathlib import Path

import lang_capture as lc
from questions import LANG_NAME, LANGS, QUESTIONS

RATE = 24000             # edge-tts output is 24 kHz mono


def clip_frames(mp3, tmp):
    wav = Path(tmp) / (mp3.stem + ".wav")
    subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{RATE}", "-c", "1",
                    str(mp3), str(wav)], check=True)
    with wave.open(str(wav)) as w:
        return w.readframes(w.getnframes())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gap", type=float, default=25, help="seconds of silence after each question")
    ap.add_argument("--speaker", type=int, choices=[1, 2, 3], default=1)
    args = ap.parse_args()

    cfg = lc.load_config(require=False)
    voice = cfg["voices"][str(args.speaker)]
    lc.make_tts(cfg, voice)
    silence = b"\x00\x00" * int(RATE * args.gap)
    out_dir = lc.ROOT / "playlists"
    out_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        for lang in LANGS:
            wav = Path(tmp) / f"{lang}_all.wav"
            with wave.open(str(wav), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                for pid in QUESTIONS:
                    w.writeframes(clip_frames(lc.tts_path(voice, lang, pid), tmp))
                    w.writeframes(silence)
                seconds = w.getnframes() / RATE
            out = out_dir / f"{LANG_NAME[lang].lower()}_all_s{args.speaker}.mp4"
            subprocess.run(["afconvert", "-f", "mp4f", "-d", "aac", str(wav), str(out)], check=True)
            print(f"{out}  ({seconds:.0f} s)")


if __name__ == "__main__":
    main()
