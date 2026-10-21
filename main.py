"""FaceFinder – Windows desktop app. Pick reference photo(s), choose a scan type, get every photo of that person."""
from __future__ import annotations
import os
import shutil
import sys
import threading
import time
from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt, QSize, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
                               QRadioButton, QSlider, QVBoxLayout, QWidget)

from facefinder import config
from facefinder.cache import Cache
from facefinder.engine import get_engine, load_image
from facefinder.pipeline import Hit, ScanJob
from facefinder.scanner import full_roots, normalise_roots, quick_roots


def to_qimage(im: Image.Image) -> QImage:
    im = im.convert("RGB")
    return QImage(im.tobytes("raw", "RGB"), im.width, im.height, 3 * im.width,
                  QImage.Format.Format_RGB888).copy()


class RefWorker(QThread):
    """Detects the face in each reference photo (largest face wins)."""
    done = Signal(list, list, list)      # crops, embeddings, notes
    failed = Signal(str)

    def __init__(self, paths):
        super().__init__()
        self.paths = paths

    def run(self):
        try:
            engine = get_engine()
            crops, embs, notes = [], [], []
            for p in self.paths:
                name = os.path.basename(p)
                img = load_image(p)
                faces = engine.embed(img) if img else []
                if not faces:
                    notes.append(f"No face found in {name}")
                    continue
                f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                if len(faces) > 1:
                    notes.append(f"{name}: {len(faces)} faces found – using the largest")
                x1, y1, x2, y2 = f.bbox
                pad = 0.25 * max(x2 - x1, y2 - y1)
                crop = img.crop((max(0, x1 - pad), max(0, y1 - pad), min(img.width, x2 + pad), min(img.height, y2 + pad)))
                crop.thumbnail((96, 96))
                crops.append(to_qimage(crop))
                embs.append(f.emb)
            self.done.emit(crops, embs, notes)
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
            cache = Cache(config.DATA_DIR / f"cache_{config.MODEL_NAME}.db")
            self.status.emit("Scanning…")
            job = ScanJob(engine, cache, self.refs, self.roots, self.full,
                          self.progress.emit,
                          lambda h: self.hit.emit(h.path, h.score, to_qimage(h.thumb)),
                          self.stop)
            scanned, hits = job.run()
            self.done.emit(self.stop.is_set(), scanned, hits)
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            if cache:
                cache.close()


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
        self.resize(1180, 740)
        self.ref_embs, self.folders = [], []
        self.ref_worker = self.scan_worker = None

        # ---- left panel ----
        g1 = QGroupBox("1  Reference photo(s)")
        self.ref_list = QListWidget(viewMode=QListWidget.ViewMode.IconMode, fixedHeight=110)
        self.ref_list.setIconSize(QSize(72, 72))
        self.ref_note = QLabel("Use a clear, front-facing photo. Add 2–3 for better accuracy.", wordWrap=True)
        b = QPushButton("Choose photo(s)…")
        b.clicked.connect(self.pick_refs)
        l = QVBoxLayout(g1)
        for w in (b, self.ref_list, self.ref_note):
            l.addWidget(w)

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
        self.slider = QSlider(Qt.Orientation.Horizontal, minimum=30, maximum=70,
                              value=int(config.DEFAULT_THRESHOLD * 100))
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
        self.bar = QProgressBar(textVisible=False, maximumHeight=8)
        self.results = QListWidget(viewMode=QListWidget.ViewMode.IconMode, resizeMode=QListWidget.ResizeMode.Adjust,
                                   movement=QListWidget.Movement.Static, selectionMode=QListWidget.SelectionMode.ExtendedSelection)
        self.results.setIconSize(QSize(config.THUMB, config.THUMB))
        self.results.setSpacing(8)
        self.results.itemDoubleClicked.connect(lambda i: QDesktopServices.openUrl(QUrl.fromLocalFile(i.path)))
        ob, sb, eb = QPushButton("Open"), QPushButton("Show in folder"), QPushButton("Export all visible (copy)…")
        ob.clicked.connect(lambda: self.each_selected(lambda p: QDesktopServices.openUrl(QUrl.fromLocalFile(p))))
        sb.clicked.connect(self.show_in_folder)
        eb.clicked.connect(self.export)
        brow = QHBoxLayout()
        for w in (ob, sb, eb):
            brow.addWidget(w)
        right = QVBoxLayout()
        right.addWidget(self.status)
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
        ok = bool(self.ref_embs) and (not custom or bool(self.folders))
        self.start_b.setEnabled(ok and not self.busy())
        self.cancel_b.setEnabled(self.busy())

    def apply_threshold(self):
        self.slider_lbl.setText(f"Threshold: {self.threshold:.2f}  (default {config.DEFAULT_THRESHOLD:.2f})")
        shown = 0
        for i in range(self.results.count()):
            it = self.results.item(i)
            it.setHidden(it.score < self.threshold)
            shown += not it.isHidden()
        if self.results.count():
            self.status.setText(f"{shown} matches at this strictness ({self.results.count()} candidates found)")

    def visible_items(self):
        return [self.results.item(i) for i in range(self.results.count()) if not self.results.item(i).isHidden()]

    def each_selected(self, fn):
        for it in self.results.selectedItems():
            fn(it.path)

    def show_in_folder(self):
        self.each_selected(lambda p: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(p))))

    # ---- reference photos ----
    def pick_refs(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Reference photo(s)", str(Path.home() / "Pictures"),
                                                "Images (*.jpg *.jpeg *.png *.bmp *.webp *.heic)")
        if not files:
            return
        self.ref_list.clear()
        self.ref_embs = []
        self.ref_note.setText("Analysing reference photo(s)… (first run downloads the AI model)")
        self.start_b.setEnabled(False)
        self.ref_worker = RefWorker(files)
        self.ref_worker.done.connect(self.on_refs)
        self.ref_worker.failed.connect(lambda m: self.ref_note.setText(f"Error: {m}"))
        self.ref_worker.start()

    def on_refs(self, crops, embs, notes):
        for q in crops:
            self.ref_list.addItem(QListWidgetItem(QIcon(QPixmap.fromImage(q)), ""))
        self.ref_embs = embs
        self.ref_note.setText("\n".join(notes) if notes else f"{len(embs)} face(s) ready.")
        self.refresh_ui()

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
    def start_scan(self):
        full = self.r_full.isChecked()
        roots = full_roots() if full else normalise_roots(self.folders) if self.r_custom.isChecked() else quick_roots()
        if not roots:
            QMessageBox.warning(self, "FaceFinder", "No readable folders to scan.")
            return
        if full and QMessageBox.question(self, "Full scan", f"Scan {len(roots)} drive(s)? This can take hours "
                                         "the first time (later scans are much faster).") != QMessageBox.StandardButton.Yes:
            return
        self.results.clear()
        self.bar.setRange(0, 0)
        w = self.scan_worker = ScanWorker(roots, self.ref_embs, full)
        w.status.connect(self.status.setText)
        w.progress.connect(lambda n, h, d: self.status.setText(f"Scanned {n:,} images · {h} candidates · {d}"))
        w.hit.connect(self.on_hit)
        w.done.connect(self.on_done)
        w.failed.connect(self.on_failed)
        w.start()
        self.refresh_ui()

    def on_hit(self, path, score, qimg):
        it = ScoreItem(path, score, QIcon(QPixmap.fromImage(qimg)))
        it.setHidden(score < self.threshold)
        self.results.addItem(it)

    def on_done(self, cancelled, scanned, hits):
        self.bar.setRange(0, 1)
        self.results.sortItems()
        self.refresh_ui()
        self.apply_threshold()
        self.status.setText(("Cancelled – " if cancelled else "Done – ") + f"{scanned:,} images scanned. " + self.status.text())

    def on_failed(self, msg):
        self.bar.setRange(0, 1)
        self.refresh_ui()
        QMessageBox.critical(self, "FaceFinder", msg)

    # ---- export (copy only: never touches originals) ----
    def export(self):
        items = self.visible_items()
        if not items:
            return
        d = QFileDialog.getExistingDirectory(self, "Export matches to…")
        if not d:
            return
        out = Path(d) / f"FaceFinder {time.strftime('%Y-%m-%d %H%M%S')}"
        out.mkdir(parents=True, exist_ok=True)
        for it in items:
            dst = out / Path(it.path).name
            n = 1
            while dst.exists():
                dst = out / f"{Path(it.path).stem} ({n}){Path(it.path).suffix}"
                n += 1
            shutil.copy2(it.path, dst)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    def closeEvent(self, e):
        if self.busy():
            self.scan_worker.stop.set()
            self.scan_worker.wait(5000)
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
