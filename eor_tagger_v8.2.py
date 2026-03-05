#!/usr/bin/env python3
"""
EOR Tagger — Metadata & Audio Analytics Tool
Edged Out Records — Powered by Koryuai
v8.2.0 — Open Source Edition
"""

import sys, os, json, threading, io, shutil, platform
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
from pathlib import Path
from collections import OrderedDict

# ── Core dependency check ────────────────────────────────────────────────────
MISSING = []
try:    import numpy as np
except: MISSING.append("numpy")
try:    import soundfile as sf
except: MISSING.append("soundfile")
try:
    from mutagen.wave import WAVE
    from mutagen.mp3  import MP3
    from mutagen.flac import FLAC as FLACFile, Picture as FLACPicture
    from mutagen.id3  import (
        TIT2, TPE1, TALB, TPE2, TPE3, TDRC, TRCK, TCON, TBPM, TKEY,
        TPUB, TCOP, TCOM, TSRC, TEXT, TXXX, COMM, WOAR, APIC, USLT,
    )
except: MISSING.append("mutagen")
try:    from PIL import Image, ImageTk
except: MISSING.append("Pillow")
try:    import sounddevice as sd
except: MISSING.append("sounddevice")

HAS_LANGDETECT = False
try:    from langdetect import detect; HAS_LANGDETECT = True
except: pass

if MISSING:
    r = tk.Tk(); r.withdraw()
    messagebox.showerror("EOR Tagger — Missing Dependencies",
        f"Please install:\n\npip install {' '.join(MISSING)}")
    sys.exit(1)

# ── Optional dependencies ─────────────────────────────────────────────────────
HAS_LIBROSA = False
try:    import librosa; HAS_LIBROSA = True
except: pass

HAS_VADER = False
try:    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer; HAS_VADER = True
except: pass

HAS_RAKE = False
try:    from rake_nltk import Rake; HAS_RAKE = True
except: pass

HAS_SPACY = False
try:    import spacy; HAS_SPACY = True
except: pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".aiff", ".aif", ".ogg"}

def get_desktop():
    d = Path.home() / "Desktop"
    return str(d) if d.exists() else str(Path.home())

METADATA_FIELDS = [
    ("copyright",       "Copyright",       "TCOP", "copyright"),
    ("publisher",       "Publisher",       "TPUB", "organization"),
    ("website",         "Website",         "WOAR", "contact"),
    ("lyricist",        "Lyricist",        None,   "lyricist"),
    ("composer",        "Composer",        "TCOM", "composer"),
    ("isrc",            "ISRC",            "TSRC", "isrc"),
    ("title",           "Title",           "TIT2", "title"),
    ("artist",          "Artist",          "TPE1", "artist"),
    ("album",           "Album",           "TALB", "album"),
    ("album_artist",    "Album Artist",    "TPE2", "albumartist"),
    ("performer",       "Performer",       "TPE3", "performer"),
    ("date",            "Date",            "TDRC", "date"),
    ("track_number",    "Track Number",    "TRCK", "tracknumber"),
    ("genre",           "Genre",           "TCON", "genre"),
    ("description",     "Description",     None,   "description"),
    ("comment",         "Comment",         "COMM", "comment"),
    ("bpm",             "BPM",             "TBPM", "bpm"),
    ("key",             "Key",             "TKEY", "initialkey"),
    ("instrumentation", "Instrumentation", None,   "instrumentation"),
    ("mood",            "Mood",            None,   "mood"),
    ("style",           "Style",           None,   "style"),
    ("character",       "Character",       None,   "character"),
    ("subgenre",        "Subgenre",        None,   "subgenre"),
    ("theme",           "Theme",           None,   "theme"),
    ("keywords",        "Keywords",        None,   "keywords"),
    ("usecase",         "UseCase",         None,   "usecase"),
    ("language",        "Language",        "TLAN", "language"),
]

ID3_CLASSES = {
    "TIT2": TIT2, "TPE1": TPE1, "TALB": TALB, "TPE2": TPE2, "TPE3": TPE3,
    "TDRC": TDRC, "TRCK": TRCK, "TCON": TCON, "TBPM": TBPM, "TKEY": TKEY,
    "TPUB": TPUB, "TCOP": TCOP, "TCOM": TCOM, "TSRC": TSRC, "TEXT": TEXT,
}

LIGHT = {
    "bg": "#f0f0f6", "bg2": "#e2e2ee", "bg3": "#d8d8e6",
    "input": "#ffffff", "fg": "#1a1a2e", "muted": "#6a6a8a",
    "accent": "#00a080", "accent_hover": "#00c098",
    "accent_light": "#d0f0ea",
    "border": "#c8c8da", "red": "#cc4444", "green": "#00a080",
    "select_bg": "#00a080", "select_fg": "#ffffff",
    "tree_bg": "#ffffff", "tree_fg": "#1a1a2e",
    "drag_bg": "#ffe8a0",
}
DARK = {
    "bg": "#1a1a2e", "bg2": "#252540", "bg3": "#2e2e4a",
    "input": "#2a2a45", "fg": "#d8d8e8", "muted": "#7a7a9a",
    "accent": "#00d4aa", "accent_hover": "#00f0c0",
    "accent_light": "#003830",
    "border": "#3a3a5a", "red": "#ff6b6b", "green": "#00d4aa",
    "select_bg": "#00d4aa", "select_fg": "#1a1a2e",
    "tree_bg": "#22223a", "tree_fg": "#d8d8e8",
    "drag_bg": "#665500",
}

# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class AudioAnalyzer:

    @staticmethod
    def analyze_features(filepath) -> dict:
        results = {"bpm": "", "key": "", "character": ""}
        if not HAS_LIBROSA: return results
        
        try:
            y, sr = librosa.load(str(filepath), sr=22050, mono=True, duration=60)
        except Exception as e:
            return results 
            
        # 1. BPM
        try:
            bpm = librosa.beat.beat_track(y=y, sr=sr)[0]
            if bpm > 0:
                results["bpm"] = str(int(round(float(bpm))))
        except: pass
        
        # 2. Key
        try:
            chroma_mean = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr), axis=1)
            maj_p = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
            min_p = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
            keys  = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            best_corr, best_key = -2.0, "C"
            for i in range(12):
                sh    = np.roll(chroma_mean, -i)
                maj_c = np.corrcoef(sh, maj_p)[0, 1]
                min_c = np.corrcoef(sh, min_p)[0, 1]
                if maj_c > best_corr: best_corr, best_key = maj_c, keys[i]
                if min_c > best_corr: best_corr, best_key = min_c, keys[i] + "m"
            results["key"] = best_key
        except: pass

        # 3. Character
        try:
            if hasattr(librosa.feature, 'rms'):
                rms_val = np.mean(librosa.feature.rms(y=y)[0])
            else:
                rms_val = np.mean(librosa.feature.rmse(y=y)[0])
                
            zcr_val = np.mean(librosa.feature.zero_crossing_rate(y)[0])
            
            if rms_val > 0.14 or (rms_val > 0.09 and zcr_val > 0.08):
                results["character"] = "High Energy / Aggressive"
            elif rms_val > 0.06:
                results["character"] = "Medium Energy / Bouncy"
            else:
                results["character"] = "Low Energy / Ambient"
        except: pass

        return results

# ═══════════════════════════════════════════════════════════════════════════════
# LYRICAL ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class LyricalAnalyzer:

    @staticmethod
    def analyze(text: str) -> dict:
        results = {"mood": "", "keywords": "", "theme": "", "language": ""}
        if not text or not text.strip(): return results
        
        if HAS_LANGDETECT:
            try: results["language"] = detect(text).upper()
            except: pass

        if HAS_VADER:
            try:
                compound = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
                if   compound >=  0.5: results["mood"] = "Uplifting"
                elif compound >=  0.2: results["mood"] = "Positive"
                elif compound >= -0.05: results["mood"] = "Chill"
                elif compound >= -0.2: results["mood"] = "Melancholic"
                else:                  results["mood"] = "Dark / Intense"
            except: pass
            
        if HAS_RAKE:
            try:
                r = Rake(min_length=1, max_length=3)
                r.extract_keywords_from_text(text)
                results["keywords"] = ", ".join(r.get_ranked_phrases()[:3])
            except: pass
            
        if HAS_SPACY:
            try:
                doc  = spacy.load("en_core_web_sm")(text[:3000])
                ents = list(set(e.text for e in doc.ents if e.label_ not in ("CARDINAL","ORDINAL")))
                if ents: results["theme"] = ", ".join(ents[:4])
            except: pass
            
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# METADATA HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class MetadataHandler:

    def __init__(self, filepath):
        self.path = Path(filepath)
        self.ext  = self.path.suffix.lower()

    def read_all(self) -> dict:
        if self.ext in (".wav", ".mp3"): return self._read_id3()
        if self.ext == ".flac":          return self._read_vorbis()
        return {}

    def write_all(self, data: dict, cover_path=None, lyrics=None):
        if self.ext in (".wav", ".mp3"): self._write_id3(data, cover_path, lyrics)
        elif self.ext == ".flac":        self._write_vorbis(data, cover_path, lyrics)

    def read_cover_art(self):
        try:
            if self.ext in (".wav", ".mp3"):
                a = WAVE(str(self.path)) if self.ext == ".wav" else MP3(str(self.path))
                if a.tags:
                    for f in a.tags.getall("APIC"): return f.data, f.mime
            elif self.ext == ".flac":
                a = FLACFile(str(self.path))
                if a.pictures: return a.pictures[0].data, a.pictures[0].mime
        except: pass
        return None, None

    def read_lyrics(self) -> str:
        try:
            if self.ext in (".wav", ".mp3"):
                a = WAVE(str(self.path)) if self.ext == ".wav" else MP3(str(self.path))
                if a.tags:
                    for f in a.tags.getall("USLT"): return f.text
            elif self.ext == ".flac":
                a = FLACFile(str(self.path))
                return a.get("LYRICS", [""])[0]
        except: pass
        return ""

    def write_wherefrom(self, value: str = "EOR Metadata Tagger"):
        try:
            if self.ext in (".wav", ".mp3"):
                a = WAVE(str(self.path)) if self.ext == ".wav" else MP3(str(self.path))
                if a.tags is None: a.add_tags()
                for f in [f for f in a.tags.getall("TXXX") if f.desc.upper() == "WHEREFROM"]:
                    a.tags.remove(f)
                a.tags.add(TXXX(encoding=3, desc="WHEREFROM", text=[value]))
                try: a.save(v2_version=3)
                except: a.save()
            elif self.ext == ".flac":
                a = FLACFile(str(self.path))
                a["WHEREFROM"] = value
                a.save()
        except: pass

    def copy_tags_from(self, source_path, cover_path=None):
        src       = MetadataHandler(source_path)
        data      = src.read_all()
        lyr       = src.read_lyrics()
        cov, mime = src.read_cover_art()
        self.write_all(data, cover_path, lyr)
        if not cover_path and cov:
            self._embed_cover_bytes(cov, mime)

    def _embed_cover_bytes(self, data, mime):
        try:
            if self.ext in (".wav", ".mp3"):
                a = WAVE(str(self.path)) if self.ext == ".wav" else MP3(str(self.path))
                if a.tags is None: a.add_tags()
                a.tags.delall("APIC")
                a.tags.add(APIC(encoding=3, mime=mime or "image/jpeg", type=3, desc="Cover", data=data))
                try: a.save(v2_version=3)
                except: a.save()
            elif self.ext == ".flac":
                a = FLACFile(str(self.path))
                a.clear_pictures()
                p = FLACPicture(); p.type=3; p.mime=mime or "image/jpeg"; p.data=data; p.depth=24
                a.add_picture(p); a.save()
        except: pass

    def _read_id3(self) -> dict:
        try:
            tags = (WAVE(str(self.path)) if self.ext == ".wav" else MP3(str(self.path))).tags
            if not tags: return {}
            data = {}
            for key, _, id3_tag, _ in METADATA_FIELDS:
                if id3_tag == "WOAR":
                    ff = tags.getall("WOAR"); data[key] = ff[0].url if ff else ""
                elif id3_tag == "COMM":
                    ff = tags.getall("COMM"); data[key] = str(ff[0]) if ff else ""
                elif id3_tag is None:
                    data[key] = next((str(f) for f in tags.getall("TXXX") if f.desc.upper() == key.upper()), "")
                else:
                    f = tags.get(id3_tag); data[key] = str(f) if f else ""
            return data
        except: return {}

    def _write_id3(self, data: dict, cover_path=None, lyrics=None):
        try:
            a = WAVE(str(self.path)) if self.ext == ".wav" else MP3(str(self.path))
            if a.tags is None: a.add_tags()
            tags = a.tags
            for key, _, id3_tag, _ in METADATA_FIELDS:
                val = data.get(key, "").strip()
                if id3_tag == "WOAR":
                    tags.delall("WOAR")
                    if val: tags.add(WOAR(url=val))
                elif id3_tag == "COMM":
                    tags.delall("COMM")
                    if val: tags.add(COMM(encoding=3, lang="eng", desc="", text=[val]))
                elif id3_tag is None:
                    for f in [f for f in tags.getall("TXXX") if f.desc.upper() == key.upper()]: tags.remove(f)
                    if val: tags.add(TXXX(encoding=3, desc=key.upper(), text=[val]))
                elif cls := ID3_CLASSES.get(id3_tag):
                    tags.delall(id3_tag)
                    if val: tags.add(cls(encoding=3, text=[val]))
            if lyrics is not None:
                tags.delall("USLT")
                if lyrics.strip(): tags.add(USLT(encoding=3, lang="eng", desc="", text=lyrics))
            if cover_path and os.path.isfile(cover_path):
                tags.delall("APIC")
                mime = "image/jpeg" if cover_path.lower().endswith((".jpg",".jpeg")) else "image/png"
                with open(cover_path, "rb") as f:
                    tags.add(APIC(encoding=3, mime=mime, type=3, desc="Cover", data=f.read()))
            
            try: a.save(v2_version=3)
            except: a.save()
        except: pass

    def _read_vorbis(self) -> dict:
        try:
            a = FLACFile(str(self.path))
            return {k: (v[0] if (v := a.get(vorb, [])) else "") for k, _, _, vorb in METADATA_FIELDS}
        except: return {}

    def _write_vorbis(self, data: dict, cover_path=None, lyrics=None):
        try:
            a = FLACFile(str(self.path))
            for key, _, _, vorb in METADATA_FIELDS:
                val = data.get(key, "").strip()
                if val: a[vorb] = val
                elif vorb in a: del a[vorb]
            if lyrics is not None:
                if lyrics.strip(): a["LYRICS"] = lyrics
                elif "LYRICS" in a: del a["LYRICS"]
            if cover_path and os.path.isfile(cover_path):
                a.clear_pictures()
                pic = FLACPicture()
                pic.type = 3
                pic.mime = "image/jpeg" if cover_path.lower().endswith((".jpg",".jpeg")) else "image/png"
                with open(cover_path, "rb") as f: pic.data = f.read()
                pic.depth = 24
                try:
                    img = Image.open(cover_path); pic.width, pic.height = img.size
                except: pass
                a.add_picture(pic)
            a.save()
        except: pass


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO PLAYER
# ═══════════════════════════════════════════════════════════════════════════════

class AudioPlayer:

    def __init__(self):
        self._stream      = None
        self._audio_orig  = None
        self._sr          = None
        self._pos         = 0
        self._playing     = False
        self._lock        = threading.Lock()

    def load(self, filepath):
        self.stop()
        try:
            audio, sr = sf.read(str(filepath), dtype="float32")
            with self._lock:
                self._audio_orig = audio
                self._sr = sr; self._pos = 0
        except Exception as e:
            print(f"Player failed to load audio: {e}")
            with self._lock:
                self._audio_orig = None

    @property
    def current_audio(self):
        return self._audio_orig

    @property
    def duration(self) -> float:
        return len(self._audio_orig) / self._sr if self._audio_orig is not None and self._sr else 0

    @property
    def playing(self) -> bool:
        return self._playing

    def play(self):
        if self._audio_orig is None or self._playing: return
        self._playing = True
        ch = self.current_audio.shape[1] if self.current_audio.ndim > 1 else 1

        def callback(outdata, frames, time_info, status):
            with self._lock:
                active = self.current_audio
                if active is None:
                    outdata.fill(0); return
                end   = self._pos + frames
                chunk = active[self._pos:end].copy()

                out_ch = outdata.shape[1]
                if len(chunk) == 0:
                    outdata.fill(0); self._playing = False; raise sd.CallbackStop
                if chunk.ndim == 1: chunk = chunk.reshape(-1, 1)
                if chunk.shape[1] < out_ch: chunk = np.tile(chunk, (1, out_ch))
                elif chunk.shape[1] > out_ch: chunk = chunk[:, :out_ch]
                if len(chunk) < frames:
                    outdata[:len(chunk)] = chunk; outdata[len(chunk):] = 0
                    self._playing = False; raise sd.CallbackStop
                outdata[:] = chunk
                self._pos = end

        try:
            self._stream = sd.OutputStream(
                samplerate=self._sr, channels=ch,
                callback=callback,
                finished_callback=lambda: setattr(self, "_playing", False),
            )
            self._stream.start()
        except Exception as e:
            self._playing = False; raise e

    def pause(self):
        self._playing = False
        if self._stream:
            try: self._stream.stop(); self._stream.close()
            except: pass
            self._stream = None

    def stop(self):
        self.pause()
        with self._lock: self._pos = 0

    def seek_ratio(self, ratio: float):
        if self._audio_orig is not None:
            with self._lock:
                self._pos = int(ratio * len(self._audio_orig))

    def get_progress(self) -> float:
        if self._audio_orig is not None and len(self._audio_orig) > 0:
            return self._pos / len(self._audio_orig)
        return 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# TRACK DATA STORE
# ═══════════════════════════════════════════════════════════════════════════════

class TrackDataStore:

    def __init__(self):
        self.tracks = OrderedDict()
        self._proj_dir = None

    def set_project_dir(self, d: str):
        self._proj_dir = d

    def load(self):
        pass 

    def save(self):
        pass 

    def get_track(self, fp: str) -> dict:
        if fp not in self.tracks:
            self.tracks[fp] = {
                "metadata": {}, "lyrics": ""}
        return self.tracks[fp]

    def set_metadata(self, fp: str, d: dict):
        self.get_track(fp)["metadata"].update(d)

    def get_metadata(self, fp: str) -> dict:
        return self.get_track(fp).get("metadata", {})

    def set_lyrics(self, fp: str, lyr: str):
        self.get_track(fp)["lyrics"] = lyr

    def get_lyrics(self, fp: str) -> str:
        return self.get_track(fp).get("lyrics", "")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class MasterEOR:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("EOR Tagger — Edged Out Records")
        self.root.geometry("1100x780")
        self.root.minsize(900, 600)

        self._theme_name   = "light"
        self.t             = LIGHT
        self.file_list     = []
        self.cover_art_path = None
        self.cover_photo   = None
        self.meta_vars     = {} 
        self._meta_timers  = {}
        self.processing    = False
        self._current_path = None
        self._player_dur   = 0.0
        self._poll_id      = None
        self._ignore_meta_trace = False
        
        # Hardened Drag & Drop tracking
        self._drag_data = {
            "item": None, 
            "start_idx": -1, 
            "start_y": 0, 
            "start_x": 0,
            "dragging": False, 
            "target": None
        }

        self.player       = AudioPlayer()
        self.store        = TrackDataStore()
        
        self.naming_formula = tk.StringVar(value="{artist} - [{year}] {album} - {track_number} - {title}")

        self._apply_theme()
        self._build_ui()
        self._bind_scroll()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self._save_current_to_store()
        self.player.stop()
        self.store.save()
        self.root.destroy()

    def _apply_theme(self):
        self.t = LIGHT if self._theme_name == "light" else DARK
        t = self.t
        self.root.configure(bg=t["bg"])
        s = ttk.Style(); s.theme_use("clam")
        s.configure(".", background=t["bg"], foreground=t["fg"],
                    fieldbackground=t["input"], bordercolor=t["border"],
                    insertcolor=t["fg"], selectbackground=t["accent"],
                    selectforeground=t["bg"])
        s.configure("TFrame",       background=t["bg"])
        s.configure("TLabel",       background=t["bg"], foreground=t["fg"],   font=("Segoe UI",10))
        s.configure("H.TLabel",     background=t["bg"], foreground=t["accent"],font=("Segoe UI",12,"bold"))
        s.configure("S.TLabel",     background=t["bg"], foreground=t["muted"], font=("Segoe UI",9))
        s.configure("TLabelframe",  background=t["bg"], foreground=t["muted"])
        s.configure("TLabelframe.Label", background=t["bg"], foreground=t["muted"], font=("Segoe UI",9,"bold"))
        s.configure("TButton",      background=t["bg2"], foreground=t["fg"], padding=(8,4), font=("Segoe UI",10))
        s.map("TButton",            background=[("active",t["bg3"]),("disabled",t["bg"])])
        s.configure("A.TButton",    background=t["accent"], foreground=t["select_fg"], padding=(10,5), font=("Segoe UI",10,"bold"))
        s.map("A.TButton",          background=[("active",t["accent_hover"]),("disabled",t["muted"])])
        s.configure("P.TButton",    background=t["green"], foreground=t["select_fg"], padding=(8,4), font=("Segoe UI",10,"bold"))
        s.map("P.TButton",          background=[("active",t["accent_hover"])])
        s.configure("X.TButton",    background=t["red"], foreground="#fff", padding=(8,4), font=("Segoe UI",10,"bold"))
        s.map("X.TButton",          background=[("active",t["red"])])
        s.configure("TEntry",       fieldbackground=t["input"], foreground=t["fg"], insertcolor=t["fg"], padding=3)
        s.configure("TCombobox",    fieldbackground=t["input"], foreground=t["fg"])
        s.map("TCombobox",          fieldbackground=[("readonly",t["input"])])
        s.configure("Treeview",     background=t["tree_bg"], foreground=t["tree_fg"],
                    fieldbackground=t["tree_bg"], rowheight=24, font=("Segoe UI",10))
        s.configure("Treeview.Heading", background=t["bg2"], foreground=t["muted"], font=("Segoe UI",9,"bold"))
        s.map("Treeview",           background=[("selected",t["accent"])],
                                    foreground=[("selected",t["select_fg"])])
        s.configure("TProgressbar", background=t["accent"], troughcolor=t["bg2"])
        s.configure("TScale",       background=t["bg"], troughcolor=t["bg2"])
        s.configure("TCheckbutton", background=t["bg"], foreground=t["fg"])
        s.configure("TRadiobutton", background=t["bg"], foreground=t["fg"])
        s.configure("TSeparator",   background=t["border"])
        
        # Tags for drag & drop
        s.configure("Treeview", fieldbackground=t["tree_bg"])
        try:
            self.tree.tag_configure("dragging", background=t["drag_bg"], foreground=t["tree_fg"])
            self.tree.tag_configure("drag_target", background=t["select_bg"], foreground=t["select_fg"])
        except: pass

    def _toggle_theme(self):
        self._theme_name = "dark" if self._theme_name == "light" else "light"
        self._apply_theme()
        if hasattr(self, "theme_btn"):
            self.theme_btn.configure(text="☀" if self._theme_name == "dark" else "🌙")
        t = self.t
        for w in (self.lyrics_text,):
            try: w.configure(bg=t["input"], fg=t["fg"], insertbackground=t["fg"])
            except: pass

    def _bind_scroll(self):
        def _scroll(event):
            w = event.widget
            while w:
                if isinstance(w, (tk.Canvas, tk.Text)):
                    if event.num == 4 or event.delta > 0: w.yview_scroll(-1, "units")
                    else: w.yview_scroll(1, "units")
                    break
                w = w.master
        self.root.bind_all("<MouseWheel>", _scroll)
        self.root.bind_all("<Button-4>",   _scroll)
        self.root.bind_all("<Button-5>",   _scroll)

    def _build_ui(self):
        top = ttk.Frame(self.root); top.pack(fill="x", padx=10, pady=(7,2))
        ttk.Label(top, text="EOR TAGGER",   style="H.TLabel").pack(side="left")
        ttk.Label(top, text="  Edged Out Records — Powered by Koryuai",
                  style="S.TLabel").pack(side="left")
        self.theme_btn = ttk.Button(top, text="🌙", width=3, command=self._toggle_theme)
        self.theme_btn.pack(side="right")
        ttk.Separator(self.root).pack(fill="x", padx=10, pady=2)

        pw = ttk.PanedWindow(self.root, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=6, pady=(0,2))

        left = ttk.Frame(pw, width=270);  pw.add(left,  weight=0)
        right = ttk.Frame(pw);            pw.add(right, weight=1)

        self._build_library_pane(left)
        self._build_work_pane(right)

        bot = ttk.Frame(self.root); bot.pack(fill="x", padx=10, pady=(0,4))
        self.status_var = tk.StringVar(value="Ready — Add files to begin")
        ttk.Label(bot, textvariable=self.status_var, style="S.TLabel").pack(side="left")
        self.progress = ttk.Progressbar(bot, mode="determinate", length=180)
        self.progress.pack(side="right")

    def _build_library_pane(self, p):
        hf = ttk.Frame(p); hf.pack(fill="x", padx=4, pady=(4,2))
        ttk.Label(hf, text="Library", style="H.TLabel").pack(side="left")
        ttk.Button(hf, text="+ Add Files", style="A.TButton",
                   command=self._add_files).pack(side="right")

        tf = ttk.Frame(p); tf.pack(fill="both", expand=True, padx=2)
        
        self.tree = ttk.Treeview(tf, columns=("num",), show="tree headings",
                                  selectmode="extended")
        self.tree.heading("#0",  text="Track",  anchor="w")
        self.tree.heading("num", text="#",       anchor="center")
        self.tree.column("#0",   width=175, minwidth=80)
        self.tree.column("num",  width=40,  stretch=False)
        
        scr = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scr.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scr.pack(side="right", fill="y")

        self.tree.tag_configure("dragging", background=self.t["drag_bg"], foreground=self.t["tree_fg"])
        self.tree.tag_configure("drag_target", background=self.t["select_bg"], foreground=self.t["select_fg"])

        self.tree.bind("<<TreeviewSelect>>",  self._on_select)
        self.tree.bind("<ButtonPress-1>",     self._on_drag_start)
        self.tree.bind("<B1-Motion>",         self._on_drag_motion)
        self.tree.bind("<ButtonRelease-1>",   self._on_drag_release)

        bf = ttk.Frame(p); bf.pack(fill="x", padx=2, pady=(3,0))
        ttk.Button(bf, text="↑", width=3, command=lambda: self._move_track(-1)).pack(side="left", padx=1)
        ttk.Button(bf, text="↓", width=3, command=lambda: self._move_track(1)).pack(side="left", padx=1)
        ttk.Button(bf, text="All", width=4, command=lambda: self.tree.selection_set(self.tree.get_children())).pack(side="left", padx=1)
        ttk.Button(bf, text="Clear", command=self._clear_files).pack(side="left", padx=1)
        self.fcount = tk.StringVar(value="0 files")
        ttk.Label(bf, textvariable=self.fcount, style="S.TLabel").pack(side="right", padx=3)

        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=4, pady=6)

        ttk.Label(p, text="Save Destination", style="H.TLabel").pack(anchor="w", padx=4, pady=(0,3))

        of = ttk.Frame(p); of.pack(fill="x", padx=4, pady=1)
        ttk.Label(of, text="Output Folder:", font=("Segoe UI",9)).pack(anchor="w")
        of2 = ttk.Frame(p); of2.pack(fill="x", padx=4, pady=1)
        self.out_var = tk.StringVar(value=get_desktop())
        ttk.Entry(of2, textvariable=self.out_var, font=("Segoe UI",9)).pack(side="left", fill="x", expand=True)
        ttk.Button(of2, text="Browse", command=self._browse_output).pack(side="right", padx=(3,0))

        nf = ttk.Frame(p); nf.pack(fill="x", padx=4, pady=(4,2))
        ttk.Label(nf, text="Filename formula:", font=("Segoe UI",8,"bold"),
                  foreground=self.t["muted"]).pack(anchor="w")
        ttk.Entry(nf, textvariable=self.naming_formula, font=("Courier",8), foreground=self.t["accent"], width=50).pack(anchor="w", pady=(1,0), fill="x")
        ttk.Label(nf, text="Available variables: {artist}, {year}, {album}, {track_number}, {title}",
                  style="S.TLabel", wraplength=240).pack(anchor="w")

        xf = ttk.Frame(p); xf.pack(fill="x", padx=4, pady=(6,2))
        self.btn_exp_sel = ttk.Button(xf, text="Save Selected", style="A.TButton",
                                      command=lambda: self._export(sel_only=True))
        self.btn_exp_sel.pack(side="left", padx=(0,2))
        self.btn_exp_all = ttk.Button(xf, text="Save All",
                                      command=lambda: self._export(sel_only=False))
        self.btn_exp_all.pack(side="left")

    def _refresh_tree(self):
        sel = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        total = len(self.file_list)
        for i, fp in enumerate(self.file_list):
            num = f"{i+1}/{total}"
            self.store.set_metadata(str(fp), {"track_number": num})
            if self._current_path and fp == self._current_path:
                if "track_number" in self.meta_vars:
                    self._ignore_meta_trace = True
                    self.meta_vars["track_number"].set(num)
                    self._ignore_meta_trace = False
            self.tree.insert("", "end", iid=str(fp), text=fp.name, values=(num,))
        self.fcount.set(f"{total} files")
        self.btn_exp_all.configure(text=f"Save All ({total})")
        for s in sel:
            if self.tree.exists(s): self.tree.selection_add(s)

    def _build_work_pane(self, p):
        wf = ttk.Frame(p); wf.pack(fill="both", expand=True)
        self._work_canvas = tk.Canvas(wf, bg=self.t["bg"], highlightthickness=0)
        wsb = ttk.Scrollbar(wf, orient="vertical", command=self._work_canvas.yview)
        self._work_inner  = ttk.Frame(self._work_canvas)
        self._work_inner.bind("<Configure>", lambda e: self._work_canvas.configure(scrollregion=self._work_canvas.bbox("all")))
        self._work_canvas.create_window((0,0), window=self._work_inner, anchor="nw")
        self._work_canvas.bind("<Configure>", lambda e: self._work_canvas.itemconfig(1, width=e.width))
        self._work_canvas.configure(yscrollcommand=wsb.set)
        self._work_canvas.pack(side="left", fill="both", expand=True)
        wsb.pack(side="right", fill="y")

        pf = ttk.LabelFrame(self._work_inner, text="Preview Player")
        pf.pack(fill="x", padx=6, pady=(4,2))

        tr = ttk.Frame(pf); tr.pack(fill="x", padx=6, pady=2)
        ttk.Button(tr, text="▶ Play",  style="P.TButton", command=self._play,  width=7).pack(side="left", padx=(0,2))
        ttk.Button(tr, text="⏸ Pause", command=self._pause, width=7).pack(side="left", padx=(0,2))
        ttk.Button(tr, text="■ Stop",  style="X.TButton", command=self._stop,  width=7).pack(side="left", padx=(0,8))
        self.time_var = tk.StringVar(value="0:00 / 0:00")
        ttk.Label(tr, textvariable=self.time_var, font=("Courier",10)).pack(side="left")

        self.seek_var = tk.DoubleVar(value=0)
        sc = ttk.Scale(pf, from_=0, to=1000, variable=self.seek_var, orient="horizontal", command=self._on_seek)
        sc.pack(fill="x", padx=6, pady=(0,6))
        def _click_seek(e):
            ratio = e.x / sc.winfo_width()
            self.seek_var.set(ratio * 1000)
            if self.player._audio_orig is not None:
                self.player.seek_ratio(ratio)
        sc.bind("<Button-1>", _click_seek)

        ttk.Separator(self._work_inner).pack(fill="x", padx=6, pady=8)
        self._build_metadata_section(self._work_inner)

    def _build_metadata_section(self, p):
        self.stamp_var = tk.StringVar(value="Stamp Template...")

        hf = ttk.Frame(p); hf.pack(fill="x", padx=6, pady=(2,4))
        ttk.Label(hf, text="Tag & Analyze", style="H.TLabel").pack(side="left")
        
        tf = ttk.Frame(hf)
        tf.pack(side="left", padx=20)
        
        self.template_dropdown = ttk.Combobox(tf, textvariable=self.stamp_var, state="readonly", width=25)
        self.template_dropdown.pack(side="left", padx=4)
        self.template_dropdown.bind("<<ComboboxSelected>>", self._on_template_dropdown_select)

        ttk.Button(hf, text="Manage Templates", style="P.TButton",
                   command=self._open_templates_modal).pack(side="right", padx=2)

        ttk.Label(p,
            text="Select multiple tracks in the library — editing any field updates all of them simultaneously.",
            style="S.TLabel").pack(anchor="w", padx=6, pady=(0,3))

        self._refresh_template_dropdown()

        self.meta_vars = {}
        grid = ttk.Frame(p); grid.pack(fill="x", padx=6, pady=2)
        row_frame = None
        for i, (key, label, _, _) in enumerate(METADATA_FIELDS):
            if i % 2 == 0:
                row_frame = ttk.Frame(grid); row_frame.pack(fill="x", pady=1)
            ff = ttk.Frame(row_frame); ff.pack(side="left", fill="x", expand=True, padx=(0,3))
            ttk.Label(ff, text=label+":", width=13, anchor="e", font=("Segoe UI",9)).pack(side="left", padx=(0,2))
            
            var = tk.StringVar()
            self.meta_vars[key] = var
            var.trace_add("write", lambda *a, k=key: self._queue_meta_change(k))
            
            e = ttk.Entry(ff, textvariable=var, width=24, font=("Segoe UI",9))
            e.pack(side="left", fill="x", expand=True)

        ttk.Separator(p).pack(fill="x", padx=6, pady=5)
        tool_row = ttk.Frame(p); tool_row.pack(fill="x", padx=6, pady=2)

        ttk.Label(tool_row, text="Cover Art:", width=10, anchor="e",
                  font=("Segoe UI",9)).pack(side="left", padx=(0,2))
        self.covlbl = ttk.Label(tool_row, text="No image", style="S.TLabel")
        self.covlbl.pack(side="left", padx=(0,3))
        ttk.Button(tool_row, text="Browse",  command=self._browse_cov,  width=7).pack(side="left", padx=1)
        ttk.Button(tool_row, text="Clear",   command=self._clear_cov,   width=5).pack(side="left", padx=1)
        self.covprev = ttk.Label(tool_row); self.covprev.pack(side="left", padx=4)

        ttk.Button(tool_row, text="Analyze Lyrics", style="A.TButton",
                   command=self._analyze_lyrics).pack(side="right", padx=(2,0))
        
        ttk.Button(tool_row, text="Detect BPM / Key / Character",
                   command=self._detect_bpm_key).pack(side="right", padx=2)

        ttk.Separator(p).pack(fill="x", padx=6, pady=5)
        ttk.Label(p, text="Lyrics  (written to file tags)",
                  font=("Segoe UI",10,"bold")).pack(anchor="w", padx=6, pady=(0,2))
        self.lyrics_text = scrolledtext.ScrolledText(
            p, wrap="word", font=("Segoe UI",10),
            bg=self.t["input"], fg=self.t["fg"],
            insertbackground=self.t["fg"], relief="solid", bd=1, padx=6, pady=4, height=10)
        self.lyrics_text.pack(fill="x", padx=6, pady=(0,12))
        self.lyrics_text.bind("<FocusOut>", lambda e: self._on_lyrics_change())

    def _load_templates_data(self):
        p = Path.home() / "eor_metadata_templates.json"
        if p.exists():
            try:
                with open(p, "r") as f: return json.load(f)
            except: pass
        return {}

    def _refresh_template_dropdown(self):
        templates = self._load_templates_data()
        vals = ["Stamp Template..."] + list(templates.keys())
        self.template_dropdown.configure(values=vals)

    def _on_template_dropdown_select(self, event=None):
        name = self.stamp_var.get()
        if not name or name == "Stamp Template...": 
            return

        templates = self._load_templates_data()
        data = templates.get(name, {})
        if not data: 
            self.stamp_var.set("Stamp Template...")
            return
        
        targets = self._sel_paths()
        if not targets and self.file_list:
            targets = self.file_list
            
        if not targets:
            self.stamp_var.set("Stamp Template...")
            return

        for fp in targets:
            self.store.set_metadata(str(fp), data)
            
        if self._current_path and self._current_path in targets:
            self._ignore_meta_trace = True
            for k, v in data.items():
                if k in self.meta_vars:
                    self.meta_vars[k].set(v)
            self._ignore_meta_trace = False
            
        self._ss(f"Template '{name}' stamped onto {len(targets)} track(s).")
        
        self.stamp_var.set("Stamp Template...")
        self.root.focus_set() 

    def _open_templates_modal(self):
        top = tk.Toplevel(self.root)
        top.title("Manage Metadata Templates")
        top.geometry("550x650")
        top.configure(bg=self.t["bg"])
        top.transient(self.root)
        top.grab_set()

        templates_path = Path.home() / "eor_metadata_templates.json"

        def save_templates(data):
            try:
                with open(templates_path, "w") as f: json.dump(data, f, indent=2)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save templates:\n{e}", parent=top)

        templates = self._load_templates_data()

        tf = ttk.Frame(top); tf.pack(fill="x", padx=10, pady=10)
        ttk.Label(tf, text="Template Name:", font=("Segoe UI", 10, "bold")).pack(side="left")
        
        cb_var = tk.StringVar()
        cb = ttk.Combobox(tf, textvariable=cb_var, values=list(templates.keys()), width=30)
        cb.pack(side="left", padx=10)
        
        wf = ttk.Frame(top); wf.pack(fill="both", expand=True, padx=10, pady=5)
        canvas = tk.Canvas(wf, bg=self.t["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(wf, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(1, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        modal_vars = {}
        for key, label, _, _ in METADATA_FIELDS:
            row = ttk.Frame(scroll_frame); row.pack(fill="x", pady=2)
            ttk.Label(row, text=label+":", width=16, anchor="e").pack(side="left", padx=5)
            var = tk.StringVar()
            modal_vars[key] = var
            ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True, padx=5)

        def on_select(e=None):
            name = cb_var.get()
            if name in templates:
                data = templates[name]
                for k, v in modal_vars.items(): v.set(data.get(k, ""))
        cb.bind("<<ComboboxSelected>>", on_select)

        bf = ttk.Frame(top); bf.pack(fill="x", padx=10, pady=10)
        
        def do_save():
            name = cb_var.get().strip()
            if not name:
                messagebox.showwarning("Name Required", "Please enter a template name.", parent=top)
                return
            data = {k: v.get() for k, v in modal_vars.items() if v.get().strip()}
            templates[name] = data
            save_templates(templates)
            cb.configure(values=list(templates.keys()))
            self._refresh_template_dropdown()  
            messagebox.showinfo("Saved", f"Template '{name}' saved.", parent=top)
            
        def do_copy():
            current_name = cb_var.get().strip()
            if not current_name:
                messagebox.showwarning("Select Template", "Select or type a template name to copy first.", parent=top)
                return
            new_name = simpledialog.askstring("Copy Template", "Enter name for the new template:", parent=top)
            if new_name:
                new_name = new_name.strip()
                if new_name in templates:
                    if not messagebox.askyesno("Overwrite", f"Template '{new_name}' already exists. Overwrite?", parent=top):
                        return
                data = {k: v.get() for k, v in modal_vars.items() if v.get().strip()}
                templates[new_name] = data
                save_templates(templates)
                cb.configure(values=list(templates.keys()))
                cb_var.set(new_name)
                self._refresh_template_dropdown() 
                messagebox.showinfo("Copied", f"Template copied to '{new_name}'. You can now safely edit it.", parent=top)
            
        def do_del():
            name = cb_var.get().strip()
            if name in templates:
                del templates[name]
                save_templates(templates)
                cb.configure(values=list(templates.keys()))
                cb_var.set("")
                for v in modal_vars.values(): v.set("")
                self._refresh_template_dropdown() 

        ttk.Button(bf, text="Save / Update", command=do_save).pack(side="left", padx=2)
        ttk.Button(bf, text="Copy Template", command=do_copy).pack(side="left", padx=2)
        ttk.Button(bf, text="Delete", command=do_del).pack(side="left", padx=2)
        ttk.Button(bf, text="Close", command=top.destroy).pack(side="right", padx=2)

    def _add_files(self):
        files = filedialog.askopenfilenames(
            title="Add Audio Files",
            filetypes=[("Audio","*.wav *.flac *.mp3 *.aiff *.ogg"),("All","*.*")])
        n = 0
        for f in files:
            fp = Path(f)
            if fp not in self.file_list:
                self.file_list.append(fp)
                n += 1
                if self.store._proj_dir is None:
                    self.store.set_project_dir(str(fp.parent))
                    self.store.load()
        if n:
            self._refresh_tree()
            self._ss(f"Added {n} file(s)  ({len(self.file_list)} total)")

    def _clear_files(self):
        self._save_current_to_store()
        self.player.stop()
        self.tree.delete(*self.tree.get_children())
        self.file_list.clear()
        self.fcount.set("0 files")
        self._current_path = None
        self.btn_exp_sel.configure(text="Save Selected")
        self.btn_exp_all.configure(text="Save All")

    def _sel_paths(self):
        return [Path(i) for i in self.tree.selection()]

    def _move_track(self, direction: int):
        sel = self.tree.selection()
        if not sel: return
        idx = self.tree.index(sel[0])
        new_idx = idx + direction
        if 0 <= new_idx < len(self.file_list):
            self.file_list[idx], self.file_list[new_idx] = self.file_list[new_idx], self.file_list[idx]
            self._refresh_tree()
            new_iid = str(self.file_list[new_idx])
            if self.tree.exists(new_iid):
                self.tree.selection_set(new_iid)
                self.tree.see(new_iid)

    # Hardened Drag and Drop functionality
    def _on_drag_start(self, event):
        if self.tree.identify_region(event.x, event.y) not in ("cell", "tree"): 
            return
        item = self.tree.identify_row(event.y)
        if item:
            self._drag_data = {
                "item": item,
                "start_idx": self.file_list.index(Path(item)),
                "start_y": event.y,
                "start_x": event.x,
                "dragging": False,
                "target": None
            }

    def _on_drag_motion(self, event):
        if not self._drag_data.get("item"): return
        
        # Enforce threshold distance before initiating drag to avoid accidental grabs
        if not self._drag_data["dragging"]:
            if abs(event.y - self._drag_data["start_y"]) > 6 or abs(event.x - self._drag_data["start_x"]) > 6:
                self._drag_data["dragging"] = True
                self.tree.item(self._drag_data["item"], tags=("dragging",))
                self.root.configure(cursor="fleur")
            else:
                return

        target = self.tree.identify_row(event.y)
        
        # Clear visual indicator on old target
        prev_target = self._drag_data.get("target")
        if prev_target and prev_target != target and prev_target != self._drag_data["item"]:
            if self.tree.exists(prev_target):
                self.tree.item(prev_target, tags=())

        # Set visual indicator on new target
        if target and target != self._drag_data["item"]:
            self.tree.item(target, tags=("drag_target",))
            self._drag_data["target"] = target

    def _on_drag_release(self, event):
        src_item = self._drag_data.get("item")
        target_item = self._drag_data.get("target")
        dragging = self._drag_data.get("dragging", False)

        self.root.configure(cursor="")
        
        # Clean up UI tags
        if src_item and self.tree.exists(src_item):
            self.tree.item(src_item, tags=())
        if target_item and self.tree.exists(target_item):
            self.tree.item(target_item, tags=())

        # Execute List Reordering
        if dragging and src_item and target_item and src_item != target_item:
            try:
                src_idx = self.file_list.index(Path(src_item))
                target_idx = self.file_list.index(Path(target_item))
                
                # Pop and re-insert into array list
                item_val = self.file_list.pop(src_idx)
                self.file_list.insert(target_idx, item_val)
                
                self._refresh_tree()
                self.tree.selection_set(src_item)
            except ValueError:
                pass

        # Reset dragging payload state
        self._drag_data = {
            "item": None, "start_idx": -1, "start_y": 0, "start_x": 0,
            "dragging": False, "target": None
        }

    def _on_select(self, e=None):
        self._save_current_to_store()
        sel = self._sel_paths()
        if not sel: return
        n = len(sel)
        self.btn_exp_sel.configure(text=f"Save Selected ({n})")
        self._current_path = sel[0]
        self._load_track_into_ui(self._current_path)
        if n > 1:
            self._ss(f"{n} tracks selected — editing text fields will instantly apply to all of them.")

    def _load_track_into_ui(self, path):
        file_data  = MetadataHandler(path).read_all()
        store_data = self.store.get_metadata(str(path))
        
        merged     = {**file_data, **store_data}
        
        self._ignore_meta_trace = True
        for key, var in self.meta_vars.items():
            var.set(merged.get(key, ""))
        self._ignore_meta_trace = False

        lyrics = self.store.get_lyrics(str(path))
        self.lyrics_text.delete("1.0", "end")
        if lyrics:
            self.lyrics_text.insert("1.0", lyrics)
        else:
            file_lyr = MetadataHandler(path).read_lyrics()
            if file_lyr: self.lyrics_text.insert("1.0", file_lyr)

        img_data, _ = MetadataHandler(path).read_cover_art()
        if img_data:
            try:
                img = Image.open(io.BytesIO(img_data)); img.thumbnail((75,75))
                self.cover_photo = ImageTk.PhotoImage(img)
                self.covprev.configure(image=self.cover_photo)
                self.covlbl.configure(text="(embedded)")
            except: pass
        else:
            self._clear_cov()

        self.player.stop()
        self.player.load(path)
        try:    self._player_dur = sf.info(str(path)).duration
        except: self._player_dur = 0.0
        self.time_var.set(f"0:00 / {self._ft(self._player_dur)}")
        self.seek_var.set(0)
        self._ss(f"Selected: {path.name}")

    def _save_current_to_store(self):
        if not self._current_path: return
        meta   = {k: var.get() for k, var in self.meta_vars.items()}
        lyrics = self.lyrics_text.get("1.0", "end-1c")

        self.store.set_metadata(str(self._current_path), meta)
        self.store.set_lyrics(str(self._current_path), lyrics)

    def _queue_meta_change(self, key):
        if self._ignore_meta_trace: return
        if key in self._meta_timers:
            self.root.after_cancel(self._meta_timers[key])
        self._meta_timers[key] = self.root.after(400, lambda: self._commit_meta_change(key))

    def _commit_meta_change(self, key):
        val = self.meta_vars[key].get()
        targets = self._sel_paths()
        if not targets and self._current_path:
            targets = [self._current_path]
        for fp in targets:
            self.store.set_metadata(str(fp), {key: val})
        if len(targets) > 1:
            self._ss(f"Updated '{key.replace('_',' ').title()}' on {len(targets)} tracks.")

    def _on_lyrics_change(self):
        if self._current_path:
            self.store.set_lyrics(str(self._current_path),
                                  self.lyrics_text.get("1.0","end-1c"))

    def _play(self):
        if not self._current_path: return
        if not self.player.playing:
            try:
                self.player.play(); self._start_poll()
            except Exception as e:
                self._ss(f"Playback error: {e}")

    def _pause(self):
        self.player.pause(); self._stop_poll(); self._ss("Paused")

    def _stop(self):
        self.player.stop(); self.seek_var.set(0)
        self._stop_poll(); self._ss("Stopped")

    def _on_seek(self, v):
        if self.player.playing:
            self.player.seek_ratio(float(v) / 1000.0)

    def _start_poll(self):
        self._stop_poll()
        def poll():
            if self.player.playing:
                r = self.player.get_progress()
                self.seek_var.set(r * 1000)
                self.time_var.set(f"{self._ft(r*self._player_dur)} / {self._ft(self._player_dur)}")
                self._poll_id = self.root.after(100, poll)
            else:
                self.time_var.set(f"{self._ft(self._player_dur)} / {self._ft(self._player_dur)}")
        poll()

    def _stop_poll(self):
        if self._poll_id: self.root.after_cancel(self._poll_id); self._poll_id = None

    @staticmethod
    def _ft(s: float) -> str:
        return f"{int(s)//60}:{int(s)%60:02d}"

    def _detect_bpm_key(self):
        targets = self._sel_paths()
        if not targets and self._current_path: 
            targets = [self._current_path]
        if not targets: return
            
        if not HAS_LIBROSA:
            self._ss("librosa not installed — pip install librosa"); return
            
        def go():
            for i, fp in enumerate(targets):
                self._ss(f"Analyzing Audio Features for {fp.name} ({i+1}/{len(targets)})...")
                
                res = AudioAnalyzer.analyze_features(fp)
                meta_update = {k: v for k, v in res.items() if v}
                
                if meta_update:
                    self.store.set_metadata(str(fp), meta_update)
                    
                    if fp == self._current_path:
                        self.root.after(0, lambda mu=meta_update: [
                            self.meta_vars[k].set(v) for k, v in mu.items() if k in self.meta_vars
                        ])
                        
            self._ss(f"Batch audio analysis complete for {len(targets)} track(s).")
            
        threading.Thread(target=go, daemon=True).start()

    def _analyze_lyrics(self):
        targets = self._sel_paths()
        if not targets and self._current_path:
            targets = [self._current_path]
        if not targets:
            return

        def go():
            for i, fp in enumerate(targets):
                self._ss(f"Analyzing lyrics for {fp.name} ({i+1}/{len(targets)})...")

                if fp == self._current_path:
                    text = self.lyrics_text.get("1.0", "end-1c").strip()
                else:
                    text = self.store.get_lyrics(str(fp)).strip()

                if not text:
                    continue

                results = LyricalAnalyzer.analyze(text)
                meta_update = {k: v for k, v in results.items() if v}

                if meta_update:
                    self.store.set_metadata(str(fp), meta_update)
                    if fp == self._current_path:
                        self.root.after(0, lambda r=meta_update: [
                            self.meta_vars[k].set(v) for k, v in r.items() if k in self.meta_vars
                        ])

            self._ss(f"Batch lyrical analysis complete for {len(targets)} track(s).")

        threading.Thread(target=go, daemon=True).start()

    def _browse_cov(self):
        p = filedialog.askopenfilename(filetypes=[("Images","*.jpg *.jpeg *.png"),("All","*.*")])
        if p:
            self.cover_art_path = p
            self.covlbl.configure(text=Path(p).name)
            try:
                img = Image.open(p); img.thumbnail((75,75))
                self.cover_photo = ImageTk.PhotoImage(img)
                self.covprev.configure(image=self.cover_photo)
            except: pass

    def _clear_cov(self):
        self.cover_art_path = None
        self.covlbl.configure(text="No image")
        self.cover_photo = None
        self.covprev.configure(image="")

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select Output Folder", initialdir=self.out_var.get())
        if d: self.out_var.set(d)

    def _export(self, sel_only=True):
        if self.processing: return
        files = self._sel_paths() if sel_only else list(self.file_list)
        if not files: self._ss("No files selected"); return

        self._save_current_to_store()
        out_dir  = Path(self.out_var.get())
        total    = len(files)
        formula  = self.naming_formula.get()
        
        if not formula.strip():
            formula = "{artist} - [{year}] {album} - {track_number} - {title}"
            
        self.processing = True
        self.progress["value"] = 0
        self.player.stop()

        def go():
            errs = 0
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self._ss(f"Cannot create output dir: {e}")
                self.processing = False; return

            for i, fp in enumerate(files):
                try:
                    meta     = self.store.get_metadata(str(fp))
                    artist   = meta.get("artist","")  or fp.stem
                    year_raw = meta.get("date","")
                    year     = year_raw[:4] if year_raw else "0000"
                    album    = meta.get("album","")   or "Unknown Album"
                    raw_trk  = str(meta.get("track_number","1"))
                    trk_num  = raw_trk.split("/")[0].zfill(2)
                    title    = meta.get("title","")   or fp.stem
                    
                    try:
                        built = formula.format(
                            artist=artist, year=year, album=album, track_number=trk_num, title=title
                        )
                    except Exception:
                        built = f"{artist} - [{year}] {album} - {trk_num} - {title}"
                        
                    safe     = "".join(c for c in built if c not in '<>:"/\\|?*').strip()
                    
                    # Instead of altering extensions or exporting new files, copy identical extensions over
                    ext_out  = fp.suffix 
                    oname    = (safe or fp.stem) + ext_out
                    out_path = out_dir / oname

                    self._ss(f"Saving {i+1}/{total}: {oname}")

                    # Non-destructive copy
                    shutil.copy2(str(fp), str(out_path))

                    h = MetadataHandler(out_path)
                    h.copy_tags_from(fp, self.cover_art_path)
                    h.write_all(meta, self.cover_art_path, lyrics=self.store.get_lyrics(str(fp)))
                    h.write_wherefrom("EOR Tagger Application")

                except Exception as e:
                    errs += 1
                    print(f"Export error {fp.name}: {e}")

                self.root.after(0, lambda v=(i+1)/total*100: self.progress.configure(value=v))

            msg = f"Done! {total-errs}/{total} saved to {out_dir.name}/"
            if errs: msg += f" ({errs} error{'s' if errs>1 else ''})"
            self._ss(msg)
            self.processing = False
            self.root.after(2000, lambda: self.progress.configure(value=0))

        threading.Thread(target=go, daemon=True).start()

    def _ss(self, msg: str):
        self.root.after(0, lambda: self.status_var.set(msg))

# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    MasterEOR().run()
