"""FaceFinder – Windows desktop app. Pick reference photo(s), choose a scan type, get every photo of that person."""
from __future__ import annotations
import os
import shutil
import sys
import threading
import time
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt, QSize, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QRadioButton, QSlider, QVBoxLayout, QWidget)

from facefinder import config, settings
from facefinder.cache import Cache
from facefinder.engine import get_engine, load_image
from facefinder.history import History
from facefinder.pipeline import ScanJob
from facefinder.scanner import full_roots, normalise_roots, quick_roots


# ───────────────────────── helpers ─────────────────────────
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
    progress = Signal(int, int, str)
    hit = Signal(str, float, QImage)
    done = Signal(bool, int, int)        # cancelled, scanned, hits
    failed = Signal(str)

    def __init__(self, roots, refs, full):
        super().__init__()
        self.roots, self.refs, self.full = roots, refs, full
        self.stop = threading.Event()

    def run(self):
        cache = None
        try:
            self.status.emit("Loading AI model (first run downloads it, one time)…")
            engine = get_engine()
            cache = Cache(settings.cache_path(config.MODEL_NAME))
            self.status.emit("Scanning…")
            job = ScanJob(engine, cache, self.refs, self.roots, self.full, self.progress.emit,
                          lambda h: self.hit.emit(h.path, h.score, to_qimage(h.thumb)), self.stop)
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
            it = QListWidgetItem(f"{when}  ·  {scope} scan{' (cancelled)' if s['cancelled'] else ''}\n"
                                 f"{s['scanned']:,} images scanned  ·  {s['matches']} matches at {s['threshold']:.2f}")
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

        self.start_b, self.cancel_b = QPushButton("Start scan"), QPushButton("Cancel")
        self.start_b.clicked.connect(self.start_scan)
        self.cancel_b.clicked.connect(lambda: self.scan_worker and self.scan_worker.stop.set())
        self.start_b.setMinimumHeight(38)

        left = QVBoxLayout()
        for w in (g1, g2, g3, self.start_b, self.cancel_b):
            left.addWidget(w)
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
        self.results = QListWidget(viewMode=QListWidget.ViewMode.IconMode, resizeMode=QListWidget.ResizeMode.Adjust,
                                   movement=QListWidget.Movement.Static,
                                   selectionMode=QListWidget.SelectionMode.ExtendedSelection)
        self.results.setIconSize(QSize(config.THUMB, config.THUMB))
        self.results.setSpacing(8)
        self.results.itemDoubleClicked.connect(lambda i: QDesktopServices.openUrl(QUrl.fromLocalFile(i.path)))
        ob, sb, eb = QPushButton("Open"), QPushButton("Show in folder"), QPushButton("Export all visible (copy)…")
        ob.clicked.connect(lambda: self.each_selected(lambda p: QDesktopServices.openUrl(QUrl.fromLocalFile(p))))
        sb.clicked.connect(lambda: self.each_selected(lambda p: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(p)))))
        eb.clicked.connect(self.export)
        brow = QHBoxLayout()
        for w in (ob, sb, eb):
            brow.addWidget(w)
        right = QVBoxLayout()
        right.addLayout(top)
        right.addWidget(self.bar)
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
        self.cancel_b.setEnabled(self.busy())

    def apply_threshold(self):
        self.slider_lbl.setText(f"Threshold: {self.threshold:.2f}  (default {config.DEFAULT_THRESHOLD:.2f})")
        shown = 0
        for i in range(self.results.count()):
            it = self.results.item(i)
            it.setHidden(it.score < self.threshold)
            shown += not it.isHidden()
        if self.results.count() and not self.busy():
            self.status.setText(f"{shown} matches at this strictness ({self.results.count()} candidates found)")

    def visible_items(self):
        return [it for it in (self.results.item(i) for i in range(self.results.count())) if not it.isHidden()]

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
        w = self.scan_worker = ScanWorker(roots, [e for _, e in self.refs], full)
        w.status.connect(self.status.setText)
        w.progress.connect(lambda n, h, d: self.status.setText(f"Scanned {n:,} images · {h} candidates · {d}"))
        w.hit.connect(self.on_hit)
        w.done.connect(self.on_done)
        w.failed.connect(self.on_failed)
        w.start()
        self.refresh_ui()

    def on_hit(self, path, score, qimg):
        it = ScoreItem(path, score, icon_of(qimg))
        it.setHidden(score < self.threshold)
        self.results.addItem(it)

    def on_done(self, cancelled, scanned, hits):
        self.bar.setRange(0, 1)
        self.results.sortItems()
        results = [(self.results.item(i).path, self.results.item(i).score) for i in range(self.results.count())]
        try:
            self.history.add_scan(self.ctx["mode"], self.ctx["roots"], self.threshold, scanned, cancelled,
                                  [(qimage_to_png(c), e) for c, e in self.refs], results)
        except Exception as e:
            print("history save failed:", e)
        self.refresh_ui()
        self.apply_threshold()
        self.status.setText(("Cancelled – " if cancelled else "Done – ") + f"{scanned:,} images scanned. " + self.status.text())

    def on_failed(self, msg):
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
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
