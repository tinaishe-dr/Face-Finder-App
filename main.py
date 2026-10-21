"""FaceFinder – Windows desktop app. Pick reference photo(s), choose a scan type, get every photo of that person."""
from __future__ import annotations
import ctypes
import os
import shutil
import sys
import threading
import time
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt, QSize, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QRadioButton, QSizePolicy, QSlider, QVBoxLayout, QWidget)

from facefinder import config, settings
from facefinder.cache import Cache
from facefinder.clock import ScanClock, clock_text, human_text
from facefinder.engine import get_engine, load_image
from facefinder.history import History
from facefinder.pipeline import ScanJob
from facefinder.scanner import full_roots, normalise_roots, quick_roots


# ───────────────────────── helpers ─────────────────────────
def resource_path(rel: str) -> Path:
    """Bundled file location – works from source and from a PyInstaller build."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / rel


def app_icon() -> QIcon:
    icon = QIcon(str(resource_path("assets/icon.ico")))
    if icon.isNull():                                   # fall back to the PNG if the ICO plugin is missing
        icon = QIcon(str(resource_path("assets/icon.png")))
    return icon


def to_qimage(im: Image.Image) -> QImage:
    im = im.convert("RGB")
    return QImage(im.tobytes("raw", "RGB"), im.width, im.height, 3 * im.width,
                  QImage.Format.Format_RGB888).copy()


def qimage_to_png(q: QImage) -> bytes:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    q.save(buf, "PNG")
    return bytes(ba)


def png_to_qimage(b: bytes) -> QImage:
    q = QImage()
    q.loadFromData(b, "PNG")
    return q


def icon_of(q: QImage) -> QIcon:
    return QIcon(QPixmap.fromImage(q))


def crop_face(img: Image.Image, bbox) -> Image.Image:
    x1, y1, x2, y2 = bbox
    pad = 0.25 * max(x2 - x1, y2 - y1)
    c = img.crop((int(max(0, x1 - pad)), int(max(0, y1 - pad)), int(min(img.width, x2 + pad)), int(min(img.height, y2 + pad))))
    c.thumbnail((96, 96))
    return c


# ───────────────────────── workers ─────────────────────────
class RefWorker(QThread):
    """Detects ALL faces in each reference photo; the UI lets the user choose when there are several."""
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, paths):
        super().__init__()
        self.paths = paths

    def run(self):
        try:
            engine, out = get_engine(), []
            for p in self.paths:
                img = load_image(p)
                faces = sorted(engine.embed(img), key=lambda f: f.bbox[0]) if img else []   # left → right
                out.append({"name": os.path.basename(p),
                            "image": to_qimage(img) if len(faces) > 1 else None,
                            "faces": [{"bbox": f.bbox, "emb": f.emb, "crop": to_qimage(crop_face(img, f.bbox))}
                                      for f in faces]})
            self.done.emit(out)
        except Exception as e:
            self.failed.emit(str(e))


class ScanWorker(QThread):
    status = Signal(str)
    state = Signal(str)                  # "paused" | "running"
    progress = Signal(int, int, str)
    hit = Signal(str, float, QImage)
    done = Signal(bool, int, int)        # stopped, scanned, hits
    failed = Signal(str)

    def __init__(self, roots, refs, full):
        super().__init__()
        self.roots, self.refs, self.full = roots, refs, full
        self.stop, self.pause = threading.Event(), threading.Event()

    def run(self):
        cache = None
        try:
            self.status.emit("Loading AI model (first run downloads it, one time)…")
            engine = get_engine()
            cache = Cache(settings.cache_path(config.MODEL_NAME))
            self.state.emit("started")                     # model is ready: start the stopwatch
            self.status.emit("Scanning…")
            job = ScanJob(engine, cache, self.refs, self.roots, self.full, self.progress.emit,
                          lambda h: self.hit.emit(h.path, h.score, to_qimage(h.thumb)), self.stop,
                          self.pause, self.state.emit)
            scanned, hits = job.run()
            self.done.emit(self.stop.is_set(), scanned, hits)
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            if cache:
                cache.close()


class HistoryLoader(QThread):
    """Re-opens a past scan's results: just thumbnails, no AI."""
    hit = Signal(str, float, QImage)
    done = Signal(int, int)              # shown, missing

    def __init__(self, results):
        super().__init__()
        self.results, self.stop = results, threading.Event()

    def run(self):
        shown = missing = 0
        for path, score in self.results:
            if self.stop.is_set():
                return
            img = load_image(path) if os.path.exists(path) else None
            if img is None:
                missing += 1
                continue
            img.thumbnail((config.THUMB, config.THUMB))
            self.hit.emit(path, score, to_qimage(img))
            shown += 1
        self.done.emit(shown, missing)


# ───────────────────────── face picker ─────────────────────────
class FacePhoto(QLabel):
    """Photo with numbered face boxes; click a face to select it."""
    picked = Signal(int)

    def __init__(self, qimg: QImage, boxes):
        super().__init__()
        self.boxes = boxes
        base = QPixmap.fromImage(qimg)
        self.scaled = base.scaled(900, 560, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.k = self.scaled.width() / base.width()
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
        self.sel = max(range(len(boxes)), key=lambda i: area(boxes[i]))     # largest face preselected
        self.setFixedSize(self.scaled.size())
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.redraw()

    def _rect(self, b):
        return QRectF(b[0] * self.k, b[1] * self.k, (b[2] - b[0]) * self.k, (b[3] - b[1]) * self.k)

    def redraw(self):
        pm = self.scaled.copy()
        p = QPainter(pm)
        p.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        for i, b in enumerate(self.boxes):
            sel, r = i == self.sel, self._rect(b)
            col = QColor("#22c55e") if sel else QColor("#f59e0b")
            p.setPen(QPen(col, 5 if sel else 2))
            p.drawRect(r)
            tag = QRectF(r.x(), max(0, r.y() - 22), 26, 22)
            p.fillRect(tag, col)
            p.setPen(QColor("black"))
            p.drawText(tag, Qt.AlignmentFlag.AlignCenter, str(i + 1))
        p.end()
        self.setPixmap(pm)

    def mousePressEvent(self, e):
        pt = e.position()
        hits = [i for i, b in enumerate(self.boxes) if self._rect(b).contains(pt)]
        if hits:                                                         # smallest box wins when overlapping
            self.sel = min(hits, key=lambda i: self._rect(self.boxes[i]).width())
            self.redraw()
            self.picked.emit(self.sel)


class FacePickerDialog(QDialog):
    def __init__(self, analysis: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Who do you want to find?")
        faces = analysis["faces"]
        self.photo = FacePhoto(analysis["image"], [f["bbox"] for f in faces])
        self.index = self.photo.sel
        self.info = QLabel(f"Selected: face #{self.index + 1}")
        self.photo.picked.connect(self._picked)
        ok, skip = QPushButton("Use this face"), QPushButton("Skip this photo")
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        skip.clicked.connect(self.reject)
        row = QHBoxLayout()
        row.addWidget(self.info, 1)
        row.addWidget(skip)
        row.addWidget(ok)
        l = QVBoxLayout(self)
        l.addWidget(QLabel(f"<b>{analysis['name']}</b> contains {len(faces)} people – click the one you want to find."))
        l.addWidget(self.photo, alignment=Qt.AlignmentFlag.AlignCenter)
        l.addLayout(row)

    def _picked(self, i):
        self.index = i
        self.info.setText(f"Selected: face #{i + 1}")


# ───────────────────────── image viewer ─────────────────────────
class ImageLabel(QLabel):
    """Shows an image scaled to fit, re-fitting whenever the window is resized."""

    def __init__(self):
        super().__init__(alignment=Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(320, 240)
        self._pm = None

    def set_image(self, q: QImage | None, message: str = ""):
        self._pm = QPixmap.fromImage(q) if q is not None else None
        if self._pm is None:
            self.setText(message)
        self._fit()

    def _fit(self):
        if self._pm is not None:
            self.setPixmap(self._pm.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()


class ImageViewer(QDialog):
    """Opens a result and lets you move through the other results with ← / → (or the buttons)."""

    def __init__(self, items, index: int, threshold: float, parent=None):
        super().__init__(parent)
        self.items, self.i, self.threshold = items, index, threshold
        self.resize(1000, 720)
        self.img = ImageLabel()
        self.info = QLabel(wordWrap=True)
        self.prev_b, self.next_b = QPushButton("◀  Previous"), QPushButton("Next  ▶")
        open_b, folder_b, close_b = QPushButton("Open in default app"), QPushButton("Show in folder"), QPushButton("Close")
        for b in (self.prev_b, self.next_b, open_b, folder_b, close_b):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)                # keep arrow keys for navigation
        self.prev_b.clicked.connect(lambda: self.go(-1))
        self.next_b.clicked.connect(lambda: self.go(1))
        open_b.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.current().path)))
        folder_b.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(self.current().path))))
        close_b.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addWidget(self.prev_b)
        row.addWidget(self.next_b)
        row.addStretch()
        for b in (open_b, folder_b, close_b):
            row.addWidget(b)
        l = QVBoxLayout(self)
        l.addWidget(self.img, 1)
        l.addWidget(self.info)
        l.addLayout(row)
        self.show_index(index)

    def current(self):
        return self.items[self.i]

    def go(self, step: int):
        j = self.i + step
        if 0 <= j < len(self.items):
            self.show_index(j)

    def show_index(self, i: int):
        self.i, n, it = i, len(self.items), self.items[i]
        im = load_image(it.path) if os.path.exists(it.path) else None
        self.img.set_image(to_qimage(im) if im else None, "This file no longer exists or can't be read")
        tag = "  <span style='color:#f59e0b'>· below strictness – review manually</span>" if it.score < self.threshold else ""
        self.info.setText(f"<b>{i + 1} / {n}</b>  ·  score {it.score:.2f}{tag}<br>"
                          f"<span style='color:#9ca3af'>{it.path}</span>")
        self.setWindowTitle(f"{os.path.basename(it.path)} – FaceFinder")
        self.prev_b.setEnabled(i > 0)
        self.next_b.setEnabled(i < n - 1)

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key.Key_Left:
            self.go(-1)
        elif k == Qt.Key.Key_Right:
            self.go(1)
        elif k == Qt.Key.Key_Home:
            self.show_index(0)
        elif k == Qt.Key.Key_End:
            self.show_index(len(self.items) - 1)
        else:
            super().keyPressEvent(e)


# ───────────────────────── history dialog ─────────────────────────
class HistoryDialog(QDialog):
    view_requested = Signal(int)
    rerun_requested = Signal(int)

    def __init__(self, history: History, parent=None):
        super().__init__(parent)
        self.history = history
        self.setWindowTitle("Scan history")
        self.resize(640, 480)
        self.list = QListWidget()
        self.list.setIconSize(QSize(48, 48))
        self.list.itemDoubleClicked.connect(lambda _: self._emit(self.view_requested))
        view, rerun = QPushButton("View results"), QPushButton("Re-run with same settings")
        delete, clear, close = QPushButton("Delete"), QPushButton("Clear all"), QPushButton("Close")
        view.clicked.connect(lambda: self._emit(self.view_requested))
        rerun.clicked.connect(lambda: self._emit(self.rerun_requested))
        delete.clicked.connect(self.delete_sel)
        clear.clicked.connect(self.clear_all)
        close.clicked.connect(self.reject)
        row = QHBoxLayout()
        for b in (view, rerun, delete, clear):
            row.addWidget(b)
        row.addStretch()
        row.addWidget(close)
        l = QVBoxLayout(self)
        l.addWidget(self.list)
        l.addLayout(row)
        self.reload()

    def reload(self):
        self.list.clear()
        for s in self.history.list_scans():
            when = time.strftime("%b %d, %Y  %H:%M", time.localtime(s["started"]))
            scope = {"quick": "Quick", "full": "Full", "custom": "Custom"}.get(s["mode"], s["mode"])
            it = QListWidgetItem(f"{when}  ·  {scope} scan{' (stopped)' if s['cancelled'] else ''}\n"
                                 f"{s['scanned']:,} images in {human_text(s['elapsed'])}  ·  {s['matches']} matches at {s['threshold']:.2f}"
                                 if s["elapsed"] else f"{s['scanned']:,} images scanned  ·  {s['matches']} matches at {s['threshold']:.2f}")
            it.setData(Qt.ItemDataRole.UserRole, s["id"])
            crop = self.history.first_crop(s["id"])
            if crop:
                it.setIcon(icon_of(png_to_qimage(crop)))
            self.list.addItem(it)

    def _sel(self):
        it = self.list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it else None

    def _emit(self, sig):
        sid = self._sel()
        if sid is not None:
            sig.emit(sid)
            self.accept()

    def delete_sel(self):
        sid = self._sel()
        if sid is not None:
            self.history.delete(sid)
            self.reload()

    def clear_all(self):
        if QMessageBox.question(self, "Clear history", "Delete all scan history?") == QMessageBox.StandardButton.Yes:
            self.history.clear()
            self.reload()


# ───────────────────────── settings dialog ─────────────────────────
class SettingsDialog(QDialog):
    def __init__(self, parent, busy: bool):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(620)
        self.storage = QLineEdit(str(settings.storage_dir()), readOnly=True)
        self.export = QLineEdit(str(settings.export_dir()), readOnly=True)
        self.move_cb = QCheckBox("Move existing data (AI model, scan cache, history) to the new storage folder")
        self.move_cb.setChecked(True)
        mb = round(settings.cache_size() / 1048576, 1)
        self.cache_lbl = QLabel(f"Scan cache: {mb} MB (lets re-scans skip unchanged photos)")
        save, cancel = QPushButton("Save"), QPushButton("Cancel")
        clear, opn = QPushButton("Clear scan cache"), QPushButton("Open storage folder")
        save.clicked.connect(self.save)
        cancel.clicked.connect(self.reject)
        clear.clicked.connect(self.clear_cache)
        opn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.storage.text())))

        def row(title, edit):
            b = QPushButton("Browse…")
            b.clicked.connect(lambda: self.browse(edit, title))
            r = QHBoxLayout()
            r.addWidget(edit, 1)
            r.addWidget(b)
            return r

        l = QVBoxLayout(self)
        l.addWidget(QLabel("<b>Storage folder</b> – AI model, scan cache and history"))
        l.addLayout(row("Storage folder", self.storage))
        l.addWidget(self.move_cb)
        l.addWidget(QLabel("<b>Export folder</b> – where “Export matches” opens by default"))
        l.addLayout(row("Export folder", self.export))
        l.addWidget(self.cache_lbl)
        r = QHBoxLayout()
        r.addWidget(clear)
        r.addWidget(opn)
        r.addStretch()
        r.addWidget(cancel)
        r.addWidget(save)
        l.addLayout(r)
        if busy:
            for w in (save, clear, self.move_cb):
                w.setEnabled(False)
            l.insertWidget(0, QLabel("A scan is running – settings are locked until it finishes."))

    def browse(self, edit, title):
        d = QFileDialog.getExistingDirectory(self, title, edit.text())
        if d:
            edit.setText(str(Path(d)))

    def clear_cache(self):
        settings.clear_cache()
        self.cache_lbl.setText("Scan cache: 0 MB")

    def save(self):
        new_s, new_e = Path(self.storage.text()), Path(self.export.text())
        for p in (new_s, new_e):
            err = settings.validate_writable(p)
            if err:
                QMessageBox.warning(self, "FaceFinder", err)
                return
        try:
            export_followed_storage = new_e == settings.storage_dir() / "Exports"
            notes = settings.change_storage(new_s, self.move_cb.isChecked())
            settings.set_export_dir(None if export_followed_storage else new_e)
            if notes:
                QMessageBox.information(self, "FaceFinder", "\n".join(notes))
        except OSError as e:
            QMessageBox.warning(self, "FaceFinder", str(e))
            return
        self.accept()


# ───────────────────────── main window ─────────────────────────
class ScoreItem(QListWidgetItem):
    def __init__(self, path, score, icon):
        super().__init__(icon, f"{score:.2f}")
        self.path, self.score = path, score
        self.setToolTip(path)

    def __lt__(self, other):             # best matches first
        return self.score > other.score


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceFinder – find every photo of a person")
        self.resize(1180, 760)
        self.refs: list[tuple[QImage, object]] = []        # (face crop, embedding)
        self.folders: list[str] = []
        self.history = History(settings.history_path)
        self.ref_worker = self.scan_worker = self.loader = None
        self.ctx: dict = {}
        self.pause_req = self.paused = self.stopping = False
        self.counts = (0, 0)
        self.clock = ScanClock()
        self.tick = QTimer(self)
        self.tick.setInterval(250)
        self.tick.timeout.connect(self.update_stats)

        # ---- left panel ----
        g1 = QGroupBox("1  Reference face(s)")
        self.ref_list = QListWidget(viewMode=QListWidget.ViewMode.IconMode, fixedHeight=110,
                                    selectionMode=QListWidget.SelectionMode.ExtendedSelection)
        self.ref_list.setIconSize(QSize(72, 72))
        self.ref_note = QLabel("Pick a clear photo. If it has several people, you'll choose which face to find.", wordWrap=True)
        self.pick_b, rm_b, clr_b = QPushButton("Add photo(s)…"), QPushButton("Remove"), QPushButton("Clear")
        self.pick_b.clicked.connect(self.pick_refs)
        rm_b.clicked.connect(self.remove_refs)
        clr_b.clicked.connect(lambda: (self.refs.clear(), self.rebuild_refs()))
        rr = QHBoxLayout()
        for w in (self.pick_b, rm_b, clr_b):
            rr.addWidget(w)
        l = QVBoxLayout(g1)
        l.addLayout(rr)
        l.addWidget(self.ref_list)
        l.addWidget(self.ref_note)

        g2 = QGroupBox("2  Scan type")
        self.r_quick = QRadioButton("Quick scan – Pictures, Desktop, Downloads, Documents, OneDrive")
        self.r_full = QRadioButton("Full scan – every internal drive (can take hours)")
        self.r_custom = QRadioButton("Custom scan – folders I choose")
        self.r_quick.setChecked(True)
        self.dirs = QListWidget(fixedHeight=90)
        self.add_b, self.rem_b = QPushButton("Add folder…"), QPushButton("Remove")
        self.add_b.clicked.connect(self.add_folder)
        self.rem_b.clicked.connect(self.remove_folder)
        row = QHBoxLayout()
        row.addWidget(self.add_b)
        row.addWidget(self.rem_b)
        l = QVBoxLayout(g2)
        for w in (self.r_quick, self.r_full, self.r_custom, self.dirs):
            l.addWidget(w)
        l.addLayout(row)
        for r in (self.r_quick, self.r_full, self.r_custom):
            r.toggled.connect(self.refresh_ui)

        g3 = QGroupBox("3  Match strictness")
        self.slider = QSlider(Qt.Orientation.Horizontal, minimum=30, maximum=70, value=int(config.DEFAULT_THRESHOLD * 100))
        self.slider_lbl = QLabel()
        self.slider.valueChanged.connect(self.apply_threshold)
        l = QVBoxLayout(g3)
        l.addWidget(self.slider)
        l.addWidget(self.slider_lbl)
        l.addWidget(QLabel("Lower = more results (some wrong). Higher = only near-certain matches.", wordWrap=True))

        self.start_b = QPushButton("Start scan")
        self.pause_b, self.continue_b, self.stop_b = QPushButton("Pause"), QPushButton("Continue"), QPushButton("Stop")
        self.start_b.clicked.connect(self.start_scan)
        self.pause_b.clicked.connect(self.pause_scan)
        self.continue_b.clicked.connect(self.continue_scan)
        self.stop_b.clicked.connect(self.stop_scan)
        self.start_b.setMinimumHeight(38)
        ctl = QHBoxLayout()
        for w in (self.pause_b, self.continue_b, self.stop_b):
            ctl.addWidget(w)

        left = QVBoxLayout()
        for w in (g1, g2, g3, self.start_b):
            left.addWidget(w)
        left.addLayout(ctl)
        left.addStretch()
        lw = QWidget(maximumWidth=380)
        lw.setLayout(left)

        # ---- right panel ----
        self.status = QLabel("Choose a reference photo to begin.")
        hist_b, set_b = QPushButton("History"), QPushButton("Settings")
        hist_b.clicked.connect(self.open_history)
        set_b.clicked.connect(lambda: SettingsDialog(self, self.busy()).exec())
        top = QHBoxLayout()
        top.addWidget(self.status, 1)
        top.addWidget(hist_b)
        top.addWidget(set_b)
        self.bar = QProgressBar(textVisible=False, maximumHeight=8)
        self.stats = QLabel("")
        self.stats.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.view_cb = QComboBox()
        self.view_cb.addItems(["Matching strictness only", "All candidates (including below strictness)"])
        self.view_cb.setToolTip("Candidates scoring below your strictness are kept – switch here to review them.")
        self.view_cb.currentIndexChanged.connect(self.apply_threshold)
        fr = QHBoxLayout()
        fr.addWidget(QLabel("Show:"))
        fr.addWidget(self.view_cb)
        fr.addWidget(QLabel("<span style='color:#9ca3af'>Grey “below” labels = under your strictness, possible matches to review</span>"), 1)
        self.results = QListWidget(viewMode=QListWidget.ViewMode.IconMode, resizeMode=QListWidget.ResizeMode.Adjust,
                                   movement=QListWidget.Movement.Static,
                                   selectionMode=QListWidget.SelectionMode.ExtendedSelection)
        self.results.setIconSize(QSize(config.THUMB, config.THUMB))
        self.results.setSpacing(8)
        self.results.itemDoubleClicked.connect(self.open_viewer)
        ob, sb, eb = QPushButton("View"), QPushButton("Show in folder"), QPushButton("Export all visible (copy)…")
        ob.setToolTip("Double-click a result to view it; use ← / → to move between results")
        ob.clicked.connect(lambda: self.open_viewer())
        sb.clicked.connect(lambda: self.each_selected(lambda p: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(p)))))
        eb.clicked.connect(self.export)
        brow = QHBoxLayout()
        for w in (ob, sb, eb):
            brow.addWidget(w)
        right = QVBoxLayout()
        right.addLayout(top)
        right.addWidget(self.bar)
        right.addWidget(self.stats)
        right.addLayout(fr)
        right.addWidget(self.results, 1)
        right.addLayout(brow)

        root = QHBoxLayout()
        root.addWidget(lw)
        rw = QWidget()
        rw.setLayout(right)
        root.addWidget(rw, 1)
        c = QWidget()
        c.setLayout(root)
        self.setCentralWidget(c)
        self.apply_threshold()
        self.refresh_ui()

    # ---- helpers ----
    @property
    def threshold(self):
        return self.slider.value() / 100

    def busy(self):
        return bool(self.scan_worker and self.scan_worker.isRunning())

    def refresh_ui(self):
        custom = self.r_custom.isChecked()
        for w in (self.dirs, self.add_b, self.rem_b):
            w.setEnabled(custom and not self.busy())
        ok = bool(self.refs) and (not custom or bool(self.folders))
        self.start_b.setEnabled(ok and not self.busy())
        b = self.busy()
        self.pause_b.setEnabled(b and not self.pause_req and not self.stopping)
        self.continue_b.setEnabled(b and self.pause_req and not self.stopping)
        self.stop_b.setEnabled(b and not self.stopping)

    @property
    def show_all(self):
        return self.view_cb.currentIndex() == 1

    def style_item(self, it):
        """Hide/show + label an item according to strictness and the Show filter."""
        below = it.score < self.threshold
        it.setHidden(below and not self.show_all)
        it.setText(f"{it.score:.2f} · below" if below else f"{it.score:.2f}")
        it.setForeground(QBrush(QColor("#9ca3af")) if below else QBrush())

    def apply_threshold(self):
        self.slider_lbl.setText(f"Threshold: {self.threshold:.2f}  (default {config.DEFAULT_THRESHOLD:.2f})")
        total = self.results.count()
        matching = 0
        for i in range(total):
            it = self.results.item(i)
            self.style_item(it)
            matching += it.score >= self.threshold
        if total and not self.busy():
            below = total - matching
            if self.show_all:
                self.status.setText(f"Showing all {total} candidates – {matching} within strictness, {below} below it")
            else:
                hint = f" · {below} more below strictness (choose “All candidates” to review them)" if below else ""
                self.status.setText(f"{matching} matches at this strictness{hint}")

    def visible_items(self):
        return [it for it in (self.results.item(i) for i in range(self.results.count())) if not it.isHidden()]

    def open_viewer(self, item=None):
        """Open a result in the built-in viewer; ← / → walk through the results currently shown."""
        items = self.visible_items()
        if not items:
            return
        if item is None:
            chosen = {it.path for it in self.results.selectedItems()}
            item = next((it for it in items if it.path in chosen), items[0])
        idx = next((i for i, it in enumerate(items) if it.path == item.path), 0)
        dlg = ImageViewer(items, idx, self.threshold, self)
        dlg.exec()
        cur = dlg.current()                                 # leave the grid on the last picture viewed
        self.results.clearSelection()
        cur.setSelected(True)
        self.results.setCurrentItem(cur)
        self.results.scrollToItem(cur)

    def each_selected(self, fn):
        for it in self.results.selectedItems():
            fn(it.path)

    # ---- reference faces ----
    def pick_refs(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Reference photo(s)", str(Path.home() / "Pictures"),
                                                "Images (*.jpg *.jpeg *.png *.bmp *.webp *.heic)")
        if not files:
            return
        self.ref_note.setText("Analysing photo(s)… (first run downloads the AI model)")
        self.pick_b.setEnabled(False)
        self.start_b.setEnabled(False)
        self.ref_worker = RefWorker(files)
        self.ref_worker.done.connect(self.on_refs)
        self.ref_worker.failed.connect(self.on_ref_failed)
        self.ref_worker.start()

    def on_ref_failed(self, msg):
        self.pick_b.setEnabled(True)
        self.ref_note.setText(f"Error: {msg}")
        self.refresh_ui()

    def on_refs(self, analyses):
        notes = []
        for a in analyses:
            faces = a["faces"]
            if not faces:
                notes.append(f"No face found in {a['name']}")
                continue
            idx = 0
            if len(faces) > 1:
                dlg = FacePickerDialog(a, self)
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    notes.append(f"Skipped {a['name']}")
                    continue
                idx = dlg.index
            self.refs.append((faces[idx]["crop"], faces[idx]["emb"]))
        self.pick_b.setEnabled(True)
        self.ref_note.setText("\n".join(notes) if notes else f"{len(self.refs)} face(s) ready.")
        self.rebuild_refs()

    def rebuild_refs(self):
        self.ref_list.clear()
        for crop, _ in self.refs:
            self.ref_list.addItem(QListWidgetItem(icon_of(crop), ""))
        self.refresh_ui()

    def remove_refs(self):
        for r in sorted({self.ref_list.row(i) for i in self.ref_list.selectedItems()}, reverse=True):
            del self.refs[r]
        self.rebuild_refs()

    # ---- folders ----
    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add folder to scan")
        if d and d not in self.folders:
            self.folders.append(d)
            self.dirs.addItem(d)
            self.refresh_ui()

    def remove_folder(self):
        for it in self.dirs.selectedItems():
            self.folders.remove(it.text())
            self.dirs.takeItem(self.dirs.row(it))
        self.refresh_ui()

    # ---- scanning ----
    def _stop_loader(self):
        if self.loader and self.loader.isRunning():
            self.loader.stop.set()
            self.loader.wait(3000)

    def start_scan(self):
        full, custom = self.r_full.isChecked(), self.r_custom.isChecked()
        roots = full_roots() if full else normalise_roots(self.folders) if custom else quick_roots()
        if not roots:
            QMessageBox.warning(self, "FaceFinder", "No readable folders to scan.")
            return
        if full and QMessageBox.question(self, "Full scan", f"Scan {len(roots)} drive(s)? This can take hours "
                                         "the first time (later scans are much faster).") != QMessageBox.StandardButton.Yes:
            return
        self._stop_loader()
        self.ctx = {"mode": "full" if full else "custom" if custom else "quick", "roots": roots}
        self.results.clear()
        self.bar.setRange(0, 0)
        self.pause_req = self.paused = self.stopping = False
        self.counts = (0, 0)
        self.clock.reset()
        self.stats.setText("⏱ Preparing…")
        w = self.scan_worker = ScanWorker(roots, [e for _, e in self.refs], full)
        w.status.connect(self.status.setText)
        w.state.connect(self.on_state)
        w.progress.connect(self.on_progress)
        w.hit.connect(self.on_hit)
        w.done.connect(self.on_done)
        w.failed.connect(self.on_failed)
        w.start()
        self.refresh_ui()

    # ---- pause / continue / stop ----
    def on_progress(self, n, h, d):
        self.counts = (n, h)
        if not (self.pause_req or self.stopping):          # keep "Pausing…" / "Stopping…" visible
            self.status.setText(f"Scanned {n:,} images · {h} candidates · {d}")

    def update_stats(self):
        el, (n, h) = self.clock.elapsed(), self.counts
        rate = n / el if el >= 1 else 0.0
        tag = "   ⏸ PAUSED" if self.paused else ""
        self.stats.setText(f"⏱ Elapsed {clock_text(el)}   │   Scanned {n:,} images   │   "
                           f"{rate:.1f} img/s   │   Candidates {h}{tag}")

    def on_state(self, state):
        if state == "started":
            self.clock.start()
            self.tick.start()
            self.update_stats()
        elif state == "paused":
            self.clock.pause()
            self.paused = True
            self.update_stats()
            self.bar.setRange(0, 1)                        # stop the busy animation
            n, h = self.counts
            self.status.setText(f"Paused – {n:,} images scanned · {h} candidates. Continue to resume, or Stop to finish.")

    def pause_scan(self):
        if self.busy():
            self.pause_req = True
            self.scan_worker.pause.set()
            self.status.setText("Pausing…")
            self.refresh_ui()

    def continue_scan(self):
        if self.busy() and self.pause_req:
            self.pause_req = self.paused = False
            self.bar.setRange(0, 0)
            self.status.setText("Scanning…")
            self.clock.resume()
            self.update_stats()
            self.scan_worker.pause.clear()
            self.refresh_ui()

    def stop_scan(self):
        if self.busy():
            self.stopping = True
            self.scan_worker.stop.set()                    # also wakes a paused scan
            self.status.setText("Stopping…")
            self.refresh_ui()

    def on_hit(self, path, score, qimg):
        it = ScoreItem(path, score, icon_of(qimg))
        self.results.addItem(it)
        self.style_item(it)

    def on_done(self, cancelled, scanned, hits):
        elapsed = self.clock.stop()
        self.tick.stop()
        self.counts = (scanned, hits)
        self.pause_req = self.paused = self.stopping = False
        self.update_stats()
        self.bar.setRange(0, 1)
        self.results.sortItems()
        results = [(self.results.item(i).path, self.results.item(i).score) for i in range(self.results.count())]
        try:
            self.history.add_scan(self.ctx["mode"], self.ctx["roots"], self.threshold, scanned, cancelled,
                                  [(qimage_to_png(c), e) for c, e in self.refs], results, elapsed)
        except Exception as e:
            print("history save failed:", e)
        self.refresh_ui()
        self.apply_threshold()
        self.status.setText(("Stopped – " if cancelled else "Done – ") + f"{scanned:,} images scanned in {human_text(elapsed)}. " + self.status.text())

    def on_failed(self, msg):
        self.clock.stop()
        self.tick.stop()
        self.pause_req = self.paused = self.stopping = False
        self.bar.setRange(0, 1)
        self.refresh_ui()
        QMessageBox.critical(self, "FaceFinder", msg)

    # ---- history ----
    def open_history(self):
        dlg = HistoryDialog(self.history, self)
        dlg.view_requested.connect(self.view_history)
        dlg.rerun_requested.connect(self.rerun_history)
        dlg.exec()

    def view_history(self, sid):
        if self.busy():
            QMessageBox.information(self, "FaceFinder", "Wait for the running scan to finish first.")
            return
        scan = self.history.get_scan(sid)
        if not scan:
            return
        self._stop_loader()
        self.results.clear()
        self.stats.setText(f"🕘 Saved scan · {time.strftime('%b %d, %Y %H:%M', time.localtime(scan['started']))}"
                           f" · {scan['scanned']:,} images in {human_text(scan['elapsed'])}")
        self.slider.setValue(int(round(scan["threshold"] * 100)))
        self.status.setText("Loading saved results…")
        self.bar.setRange(0, 0)
        self.loader = HistoryLoader(self.history.get_results(sid))
        self.loader.hit.connect(self.on_hit)
        self.loader.done.connect(self.on_history_loaded)
        self.loader.start()

    def on_history_loaded(self, shown, missing):
        self.bar.setRange(0, 1)
        self.results.sortItems()
        self.apply_threshold()
        if missing:
            self.status.setText(self.status.text() + f"  ({missing} saved file(s) no longer exist)")

    def rerun_history(self, sid):
        if self.busy():
            QMessageBox.information(self, "FaceFinder", "Wait for the running scan to finish first.")
            return
        scan = self.history.get_scan(sid)
        if not scan:
            return
        self.refs = [(png_to_qimage(c), e) for c, e in self.history.get_refs(sid)]
        self.rebuild_refs()
        {"quick": self.r_quick, "full": self.r_full, "custom": self.r_custom}[scan["mode"]].setChecked(True)
        if scan["mode"] == "custom":
            self.folders = [r for r in scan["roots"] if os.path.isdir(r)]
            self.dirs.clear()
            self.dirs.addItems(self.folders)
        self.slider.setValue(int(round(scan["threshold"] * 100)))
        self.refresh_ui()
        self.status.setText("Settings restored from history – press Start scan.")

    # ---- export (copy only: never touches originals) ----
    def export(self):
        items = self.visible_items()
        if not items:
            return
        start = settings.export_dir()
        start.mkdir(parents=True, exist_ok=True)
        d = QFileDialog.getExistingDirectory(self, "Export matches to…", str(start))
        if not d:
            return
        out = Path(d) / f"FaceFinder {time.strftime('%Y-%m-%d %H%M%S')}"
        out.mkdir(parents=True, exist_ok=True)
        for it in items:
            dst, n = out / Path(it.path).name, 1
            while dst.exists():
                dst = out / f"{Path(it.path).stem} ({n}){Path(it.path).suffix}"
                n += 1
            shutil.copy2(it.path, dst)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    def closeEvent(self, e):
        self._stop_loader()
        if self.busy():
            self.scan_worker.stop.set()
            self.scan_worker.wait(5000)
        e.accept()


if __name__ == "__main__":
    if sys.platform == "win32":                         # own taskbar identity so Windows shows OUR icon, not python.exe's
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("FaceFinder.App")
    app = QApplication(sys.argv)
    app.setApplicationName("FaceFinder")
    icon = app_icon()
    app.setWindowIcon(icon)                             # inherited by every dialog too
    win = MainWindow()
    win.setWindowIcon(icon)
    win.show()
    sys.exit(app.exec())
