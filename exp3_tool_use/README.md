# Experiment 3: Tool use (web search) data collection

Tooling for Experiment 3 of the *AI Voice Side-Channel: Data Collection Protocol*.
Goal: detect from packet timing and size alone whether Claude ran a web search
before answering. Our own traffic, our own accounts, scripted prompts, and
packet metadata only (size, direction, timing), never content.

You still do the recording yourself: you speak, Claude answers in voice mode.
These scripts handle everything around that:

| Script | What it does |
|---|---|
| `capture.py setup` | One-time setup: speaker ID, Wi-Fi name, Chrome version, capture interface |
| `capture.py run --sitting N` | Walks you through a sitting: shows the exact words, starts/stops the capture, names the file, writes the manifest row |
| `capture.py status` | Progress across the three sittings |
| `plan_sitting.py --sitting N` | Prints the shuffled order for a sitting (`run` makes this automatically) |
| `first_check.py FILE` | The 30-second first-file check (conversations + up/down I/O) |
| `check_files.py` | The nightly check (files vs manifest, names, broken files, progress) |

Python 3.9+ standard library only. Nothing to `pip install`.

## What gets recorded (112 captures, ~2 hours)

| Set | Prompts | Reps | Captures | File name |
|---|---|---|---|---|
| A: matched pairs, search **ON** | T01–T08 current-info + T01–T08 static | 4 | 64 | `tool_search_s<N>_pT<NN>_r<NN>.pcapng`, `tool_static_...` |
| B: control, search **OFF** | T01–T08 current-info | 4 | 32 | `tool_off_s<N>_pT<NN>_r<NN>.pcapng` |
| C: latency baseline | B01–B04 | 4 | 16 | `base_s<N>_pB<NN>_r<NN>.pcapng` |

Every T prompt ends with "Keep it to one or two sentences." Baseline prompts don't.
All prompts are in `prompts.py`, word for word from the protocol.

## One-time setup

1. **Wireshark** installed with **Npcap** (the Windows installer offers it). This
   gives you `dumpcap`, Wireshark's capture engine, which the helper drives
   with the same settings as the protocol: your Wi-Fi interface, capture filter
   `port 443`, `.pcapng` output.
2. **Claude**: web app in Chrome, your own account. Pick the team's voice and
   never change it. Web search **ON**.
3. **Hardware**: your laptop, plugged in, usual home Wi-Fi, wired headphones.
4. In this folder:
   ```
   python capture.py setup
   ```
   Enter your speaker number (the team-assigned S1/S2/S3, never changed), Wi-Fi
   name, Chrome version (from `chrome://version`), the pilot's protocol
   (TCP or UDP) if you know it, and pick your Wi-Fi adapter from the list.
   Settings are saved to `exp3_config.json`, which stays on your machine.

If you'd rather click Start/Stop in the Wireshark window yourself, add
`--manual` to `run`. The helper then tells you the exact file name to save as and
checks that it exists before moving on.

## Each sitting

| Sitting | What's in it | Captures |
|---|---|---|
| 1 | Baseline r01; Set A reps 1–2 | 36 |
| 2 | Baseline r02; Set A rep 3; Set B reps 1–2 | 36 |
| 3 (**different day**) | Baseline r03; Set A rep 4; Set B reps 3–4; baseline r04 | 40 |

Sitting 3 is the held-out test set: don't train or tune on it until models are frozen.

```
python capture.py run --sitting 1
```

The helper then:

1. Confirms today's Chrome version and runs the pre-sitting checklist
   (headphones, phone away, other tabs and sync apps closed, updates paused).
2. Tells you to **set web search ON or OFF** whenever the next capture needs a
   different setting (start of each sitting, before Set B, and back on for the
   final baseline in sitting 3).
3. For each capture, in the shuffled order:
   - shows the file name and the **exact words to say**
   - you open a **new chat**, press Enter, and the capture starts
   - you follow the on-screen procedure: voice button, 3 s silence, say the
     prompt once, stay silent during the reply, 3 s silence, end voice mode
   - you press Enter and the capture stops and is saved under the right name
   - it warns if the file is under 50 KB or 10 s
   - you choose **keep** or **redo** (redo deletes the file; start again in a new chat)
   - you log what the transcript shows: search indicator Y/N, rough reply
     word count, notes. The manifest row is written immediately.
4. After the **first capture of every sitting** it runs the first-file check
   and asks whether it looks right. If not, stop and tell the team.
5. Every 15 captures it gives you a 2-minute break timer.

You can quit any time with `q` (or Ctrl+C). Running the same command later
resumes at the next unfinished capture, in the same order. `s` skips a capture
for now; it stays pending until you come back to it.

When a sitting is done:
```
python check_files.py
git add -A && git commit -m "Exp 3 sitting N" && git push
```

## Rules that matter

- **Redo** (new chat) if you stumbled or misread, Claude cut in, Claude's voice
  reached the mic, someone else spoke, or voice mode errored or reconnected.
- **Label what happened, not what you expected.** If a current-info prompt
  didn't trigger a search, keep the file and log N. If a static prompt did
  search, keep it and log Y. Don't redo these.
- In Set B, Claude may say it can't check live info. That's expected; just log
  the word count.
- Hit a voice usage limit? Stop and finish the sitting another day; add a note.
- No port-443 traffic? Wrong interface; re-run `setup` and pick the active Wi-Fi adapter.

## Files

```
exp3_tool_use/
  capture.py  plan_sitting.py  first_check.py  check_files.py
  common.py   prompts.py       pcaplib.py
  plans/sitting<N>_s<speaker>.csv   run order + done/pending per capture (commit these)
  captures/*.pcapng                 the recordings
  manifest_exp3.csv                 one row per file, same columns as the team sheet
  exp3_config.json                  your machine's settings (git-ignored)
```

**Manifest:** the columns and order match the team's shared manifest
(`file, experiment, class, prompt_id, speaker, rep, date_time, network,
client_version, search_setting, search_observed, reply_words, notes`). Each
row's notes start with `sitting N`, so the held-out set is easy to filter.
It's a separate CSV per experiment so three people appending to one file over
git don't hit merge conflicts. Concatenating them later is trivial.

**Repo size:** 112 captures of roughly 1 minute each add up to a few hundred MB.
That's fine for GitHub file limits (100 MB each) but heavy for the repo. If the
team prefers, track `*.pcapng` with Git LFS (`git lfs track "*.pcapng"`) before
the first commit of captures.
