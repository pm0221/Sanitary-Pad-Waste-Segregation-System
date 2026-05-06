"""
================================================================================
AHP WASTE SEGREGATION — DETECTION SYSTEM 3  (FINAL)
================================================================================
CHANGES FROM PREVIOUS VERSION:
  ✅ BatchNorm1d removed  — was causing crash
  ✅ Auto-retrains        — if old model found, deletes and retrains
  ✅ 5-second intervals   — detects every 5s (thinking time for accuracy)
  ✅ Servo: 180° toggle   — stays at 180° until next pad, then back to 0°
  ✅ No pad count line    — removed from camera window
  ✅ Texture on screen    — Conf + Tex + Sim shown next to box
  ✅ Calibrated thresholds— from your actual 211 labeled pad crops

SETTINGS TO CHANGE:
  ARDUINO_PORT = "COM3"   ← your Arduino COM port
  CAMERA_ID    = 0        ← change to 1 or 2 if camera not found

HOW TO RUN:
  python detection_system3.py

FIRST RUN: model not found → trains automatically (~20 min) → detects
EVERY RUN: loads model instantly → detects
================================================================================
"""

import cv2, time, json, os, sys, shutil, zipfile, random, threading
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  SETTINGS  ← EDIT THESE
# ─────────────────────────────────────────────────────────────────────────────

MODEL_FILE   = "ahp_model.pth"
EMBED_FILE   = "ahp_embeddings.npy"
DATASET_DIR  = "ahp_dataset"
LOG_FILE     = "detection_output.json"
ZIP_FILE     = "project-1-at-2026-02-17-21-44-c758d2de.zip"

CAMERA_ID    = 0
CAM_WIDTH    = 640
CAM_HEIGHT   = 480

ARDUINO_PORT = "COM3"     # ← CHANGE THIS
ARDUINO_BAUD = 9600

# Detection interval — analyses every 5 seconds
DETECT_EVERY = 5.0

# Thresholds calibrated from your 211 labeled pad crops:
# Pad texture (normalised): 10th percentile = 0.281, mean = 0.583
# Pad brightness: 10th percentile = 101, mean = 150
CONF_THRESHOLD = 0.45     # CNN confidence
SIM_THRESHOLD  = 0.40     # cosine similarity to training images
TEX_THRESHOLD  = 0.01     # very low — texture check almost disabled, CNN+SIM decide

EMBED_DIM    = 256

# Training settings
EPOCHS       = 50
BATCH        = 16
LR           = 0.0005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET PREPARATION
#  Reads YOLO labels from ZIP, crops each labeled pad, splits 70/15/15
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset():
    total = sum(1 for _ in Path(DATASET_DIR).rglob("*.jpg")) \
            if Path(DATASET_DIR).exists() else 0
    if total > 20:
        print(f"  Dataset ready ({total} images).")
        return

    print("\n  Preparing dataset from ZIP ...")
    if not Path(ZIP_FILE).exists():
        print(f"  ERROR: '{ZIP_FILE}' not found in this folder.")
        sys.exit(1)

    raw = Path("ahp_raw")
    if not raw.exists():
        with zipfile.ZipFile(ZIP_FILE) as z:
            z.extractall(str(raw))

    img_dir  = raw / "images"
    lbl_dir  = raw / "labels"
    all_imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    pad_crops   = []
    nopad_crops = []

    for img_path in all_imgs:
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w  = img.shape[:2]
        lbl_p = lbl_dir / (img_path.stem + ".txt")
        boxes = []

        if lbl_p.exists():
            for line in lbl_p.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5: continue
                cx = float(parts[1]); cy = float(parts[2])
                bw = float(parts[3]); bh = float(parts[4])
                x1 = max(0, int((cx - bw/2) * w))
                y1 = max(0, int((cy - bh/2) * h))
                x2 = min(w, int((cx + bw/2) * w))
                y2 = min(h, int((cy + bh/2) * h))
                crop = img[y1:y2, x1:x2]
                if crop.size == 0: continue
                pad_crops.append(cv2.resize(crop, (224, 224)))
                boxes.append((x1, y1, x2, y2))

        # One background crop per image (non-overlapping with pads)
        for _ in range(40):
            cw = random.randint(w//5, w//2)
            ch = random.randint(h//5, h//2)
            x1 = random.randint(0, w - cw)
            y1 = random.randint(0, h - ch)
            x2, y2 = x1+cw, y1+ch
            overlap = any(x1<bx2 and x2>bx1 and y1<by2 and y2>by1
                          for bx1,by1,bx2,by2 in boxes)
            if not overlap:
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    nopad_crops.append(cv2.resize(crop, (224, 224)))
                break

    print(f"  Pad crops   : {len(pad_crops)}")
    print(f"  No-pad crops: {len(nopad_crops)}")
    random.seed(42)

    def save(crops, cls):
        random.shuffle(crops)
        n = len(crops); ntr = int(n*0.70); nva = int(n*0.15)
        for sp, items in [("train",crops[:ntr]),("val",crops[ntr:ntr+nva]),("test",crops[ntr+nva:])]:
            d = Path(DATASET_DIR)/sp/cls; d.mkdir(parents=True, exist_ok=True)
            for i,c in enumerate(items):
                cv2.imwrite(str(d/f"{cls}_{sp}_{i:04d}.jpg"), c)

    save(pad_crops, "pad")
    save(nopad_crops, "no_pad")
    print("  Dataset ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL  — BatchNorm1d REMOVED (caused crash with batch size 1)
# ─────────────────────────────────────────────────────────────────────────────

class AHPClassifier(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        for i, layer in enumerate(base.features):
            for p in layer.parameters():
                p.requires_grad = (i >= 14)
        self.backbone = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        # BatchNorm1d REMOVED — crashes when any batch has 1 sample
        self.embed_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, embed_dim), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(64, 2),
        )

    def forward(self, x):
        f = self.pool(self.backbone(x))
        e = self.embed_head(f)
        return self.classifier(e), e


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_TF = T.Compose([
    T.Resize((240,240)), T.RandomCrop(224),
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(25), T.ColorJitter(0.4,0.4,0.3,0.1),
    T.RandomAffine(0, translate=(0.1,0.1), scale=(0.85,1.15)),
    T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    T.RandomErasing(p=0.2),
])
EVAL_TF  = T.Compose([T.Resize((224,224)), T.ToTensor(),
                       T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
INFER_TF = T.Compose([T.Resize((224,224)), T.ToTensor(),
                       T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


class PadDataset(Dataset):
    LABELS = {"pad": 1, "no_pad": 0}
    def __init__(self, root, tf):
        self.tf = tf; self.samples = []
        for cls, lbl in self.LABELS.items():
            for p in Path(root).glob(f"{cls}/*.jpg"):
                self.samples.append((str(p), lbl))
        if not self.samples:
            raise FileNotFoundError(f"No images in {root}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, lbl = self.samples[i]
        try:    img = Image.open(path).convert("RGB")
        except: img = Image.new("RGB",(224,224),(128,128,128))
        return self.tf(img), lbl


def _acc(model, dl):
    model.eval(); c = t = 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            c += (model(x)[0].argmax(1)==y).sum().item()
            t += y.size(0)
    return c / max(t, 1)


def train_model():
    print("="*60)
    print("  TRAINING — runs once, saves permanently")
    print(f"  Device: {DEVICE}")
    print("="*60)
    prepare_dataset()

    tr = PadDataset(f"{DATASET_DIR}/train", TRAIN_TF)
    va = PadDataset(f"{DATASET_DIR}/val",   EVAL_TF)
    te = PadDataset(f"{DATASET_DIR}/test",  EVAL_TF)

    pad_n   = sum(1 for _,l in tr.samples if l==1)
    nopad_n = sum(1 for _,l in tr.samples if l==0)
    print(f"  Train:{len(tr)} (pad={pad_n}, no_pad={nopad_n})")
    print(f"  Val:{len(va)}  Test:{len(te)}\n")

    # drop_last=True prevents batch-size-1 errors
    tr_dl = DataLoader(tr, BATCH, shuffle=True,  num_workers=0, drop_last=True)
    va_dl = DataLoader(va, BATCH, shuffle=False, num_workers=0)
    te_dl = DataLoader(te, BATCH, shuffle=False, num_workers=0)

    model = AHPClassifier(EMBED_DIM).to(DEVICE)
    total = pad_n + nopad_n
    w     = torch.tensor([total/max(nopad_n,1), total/max(pad_n,1)],
                          dtype=torch.float32).to(DEVICE)
    crit  = nn.CrossEntropyLoss(weight=w)
    opt   = optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()),
                         lr=LR, weight_decay=1e-4)
    sch   = optim.lr_scheduler.OneCycleLR(opt, max_lr=LR*10,
                                           steps_per_epoch=len(tr_dl), epochs=EPOCHS)

    best = patience = 0
    print(f"  {'Ep':>3}  {'Loss':>8}  {'Train':>7}  {'Val':>7}")
    print("  " + "─"*36)

    for ep in range(1, EPOCHS+1):
        model.train(); tl = c = t = 0
        for x, y in tr_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            lo, _ = model(x)
            loss  = crit(lo, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            tl += loss.item()
            c  += (lo.argmax(1)==y).sum().item()
            t  += y.size(0)

        va = _acc(model, va_dl); note = ""
        if va >= best:
            best = va; patience = 0
            torch.save(model.state_dict(), MODEL_FILE); note = "✅"
        else:
            patience += 1
        print(f"  {ep:>3}  {tl/max(len(tr_dl),1):>8.4f}  {c/max(t,1):>7.3f}  {va:>7.3f}  {note}")
        if patience >= 15:
            print("  Early stop."); break

    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    print(f"\n  Best val: {best:.3f}   Test: {_acc(model, te_dl):.3f}")

    # Build similarity embeddings from training images
    print("  Building embeddings ...")
    model.eval(); embs = []
    with torch.no_grad():
        for x, _ in DataLoader(tr, 32, num_workers=0):
            _, e = model(x.to(DEVICE))
            embs.append(e.cpu().numpy())
    np.save(EMBED_FILE, np.vstack(embs))
    print(f"  Saved {sum(len(x) for x in embs)} embeddings.\n")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODEL  — trains automatically, handles old incompatible models
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    if not Path(MODEL_FILE).exists():
        print("\n  No model found — training now (one time only)...")
        model = train_model()
    else:
        try:
            model = AHPClassifier(EMBED_DIM).to(DEVICE)
            model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
            model.eval()
            print(f"  ✅ Model loaded [{DEVICE}]")
        except Exception:
            # Old/incompatible model — delete and retrain
            print("  Old model detected — deleting and retraining...")
            os.remove(MODEL_FILE)
            if Path(EMBED_FILE).exists():   os.remove(EMBED_FILE)
            if Path(DATASET_DIR).exists():  shutil.rmtree(DATASET_DIR)
            model = train_model()

    db_emb = np.load(EMBED_FILE) if Path(EMBED_FILE).exists() else None
    if db_emb is not None:
        print(f"  ✅ {len(db_emb)} training embeddings loaded")
    return model, db_emb


# ─────────────────────────────────────────────────────────────────────────────
#  ARDUINO
#  Servo behaviour: OUTPUT=1 → toggle position (0→180 or 180→0)
#  Servo STAYS at position until next OUTPUT=1
# ─────────────────────────────────────────────────────────────────────────────

class Arduino:
    def __init__(self):
        self.ser      = None
        self.simulate = False
        self._connect()

    def _connect(self):
        if not SERIAL_AVAILABLE:
            self.simulate = True
            print("  Arduino: SIMULATE (pip install pyserial)")
            return
        try:
            self.ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
            time.sleep(2)
            print(f"  ✅ Arduino connected: {ARDUINO_PORT}")
            if self.ser.in_waiting:
                print(f"  Arduino: {self.ser.readline().decode().strip()}")
        except PermissionError:
            self.simulate = True
            print(f"  Arduino: PORT IN USE — Close Arduino IDE then run again")
            print(f"  Running in SIMULATE mode until then.")
        except Exception as e:
            self.simulate = True
            print(f"  Arduino: SIMULATE — cannot connect to {ARDUINO_PORT}")
            print(f"  Fix: change ARDUINO_PORT in settings. ({e})")

    def send(self, value):
        """Send 1 or 0. Arduino toggles servo on receiving 1."""
        if self.simulate:
            if value == 1:
                print("  [SIMULATE] Sent 1 → Flaps rotate LEFT to 45 degrees (Sanitary Bin)")
            return
        try:
            self.ser.write(str(value).encode())
            time.sleep(0.05)
            while self.ser.in_waiting:
                r = self.ser.readline().decode().strip()
                if r: print(f"  Arduino: {r}")
        except Exception as e:
            print(f"  Arduino error: {e}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


# ─────────────────────────────────────────────────────────────────────────────
#  DETECTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

TRANSFORM = INFER_TF   # alias


def run_cnn(model, crop_bgr):
    rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = INFER_TF(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, emb = model(tensor)
        conf = torch.softmax(logits, dim=1)[0, 1].item()
    return conf, emb.cpu().numpy()


def compute_texture(crop_bgr):
    """
    Laplacian variance normalised to 0-1.
    Calibrated: pads score 0.28-0.99 (10th-90th pct from your dataset)
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    var  = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(np.clip(1.0 - np.exp(-var / 400.0), 0.0, 1.0))


def compute_similarity(query_emb, db_embeddings, top_k=7):
    """Cosine similarity against top-k training images."""
    if db_embeddings is None:
        return 1.0
    db_n = db_embeddings / (np.linalg.norm(db_embeddings, axis=1, keepdims=True) + 1e-8)
    q_n  = query_emb     / (np.linalg.norm(query_emb) + 1e-8)
    sims = (db_n @ q_n.T).flatten()
    return float(np.mean(np.sort(sims)[-min(top_k, len(sims)):]))


def find_objects(frame):
    lab    = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    blur   = cv2.GaussianBlur(lab[:, :, 0], (7, 7), 0)
    _, th  = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k      = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    th     = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    th     = cv2.morphologyEx(th, cv2.MORPH_OPEN,  k)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w   = frame.shape[:2]; objects = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < w*h*0.02 or area > w*h*0.80: continue
        bx,by,bw,bh = cv2.boundingRect(c)
        objects.append((bx, by, bw, bh, bx+bw//2, by+bh//2))
    return objects


def get_detection(frame, model, db_embeddings, conf_t, sim_t, tex_t):
    fh, fw  = frame.shape[:2]
    objects = find_objects(frame)
    best    = None; best_conf = 0.0

    for (bx, by, bw, bh, cx, cy) in objects:
        pad  = 12
        y1   = max(0, by-pad);  y2 = min(fh, by+bh+pad)
        x1   = max(0, bx-pad);  x2 = min(fw, bx+bw+pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: continue

        conf, emb = run_cnn(model, crop)
        tex       = compute_texture(crop)
        sim       = compute_similarity(emb, db_embeddings)
        ar        = (bw*bh)/(fw*fh)

        # Texture check removed from decision — CNN confidence + similarity decide
        # Texture is displayed on screen for info only
        is_pad = (conf >= conf_t and
                  sim  >= sim_t  and 0.02 <= ar <= 0.80)

        if is_pad and conf > best_conf:
            best_conf = conf
            best = {
                "output": 1, "bx":bx, "by":by, "bw":bw, "bh":bh, "cx":cx, "cy":cy,
                "confidence": round(conf,4), "texture": round(tex,4), "similarity": round(sim,4),
                "timestamp": round(time.time(),3),
            }

    if best is None:
        # Still show scores for largest object even if not detected
        if objects:
            bx,by,bw,bh,cx,cy = objects[0]
            crop = frame[max(0,by-12):min(fh,by+bh+12), max(0,bx-12):min(fw,bx+bw+12)]
            if crop.size > 0:
                conf, emb = run_cnn(model, crop)
                tex       = compute_texture(crop)
                sim       = compute_similarity(emb, db_embeddings)
                return {
                    "output":0, "bx":bx, "by":by, "bw":bw, "bh":bh, "cx":cx, "cy":cy,
                    "confidence":round(conf,4), "texture":round(tex,4), "similarity":round(sim,4),
                    "timestamp":round(time.time(),3),
                }
        return {
            "output":0, "confidence":0.0, "texture":0.0,
            "similarity":0.0, "timestamp":round(time.time(),3),
        }
    return best


# ─────────────────────────────────────────────────────────────────────────────
#  DRAW — clean camera window
#  Shows: bounding box + axes + label + Conf/Tex/Sim scores + status bar
#  Does NOT show pad count
# ─────────────────────────────────────────────────────────────────────────────

def draw_result(frame, result, objects, conf_t, sim_t, tex_t,
                fps, next_scan_in, servo_status):
    display = frame.copy()
    fh, fw  = frame.shape[:2]

    if "bx" in result:
        bx = result["bx"]; by = result["by"]
        bw = result["bw"]; bh = result["bh"]
        cx = result["cx"]; cy = result["cy"]
        det = result["output"] == 1
        col = (0, 230, 0) if det else (30, 30, 200)
        thk = 4           if det else 2

        # Bounding box
        cv2.rectangle(display, (bx,by), (bx+bw,by+bh), col, thk)

        # X axis (horizontal through centre)
        cv2.line(display, (bx,cy), (bx+bw,cy), col, 2)
        # Y axis (vertical through centre)
        cv2.line(display, (cx,by), (cx,by+bh), col, 2)
        # Centre dot
        cv2.circle(display, (cx,cy), 6, col, -1)

        # Label above box
        label = "SANITARY PAD  OUTPUT=1" if det else "NOT PAD  OUTPUT=0"
        (tw,th2),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        cv2.rectangle(display, (bx,by-th2-12), (bx+tw+8,by),
                      (0,150,0) if det else (150,0,0), -1)
        cv2.putText(display, label, (bx+4,by-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2)

        # Conf / Tex / Sim scores to the right of box
        for i,(lbl,val,thr) in enumerate([
            ("Conf", result["confidence"], conf_t),
            ("Tex",  result["texture"],    tex_t),
            ("Sim",  result["similarity"], sim_t),
        ]):
            ok  = val >= thr
            col2 = (50,220,50) if ok else (50,50,210)
            cv2.putText(display, f"{lbl}:{val:.2f}",
                        (bx+bw+8, by+22+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col2, 1)

    # Bottom status bar — NO pad count line
    out     = result["output"]
    bar_col = (0,200,0) if out==1 else (0,0,180)
    cv2.rectangle(display, (0,fh-34), (fw,fh), (15,15,15), -1)
    msg = f"OUTPUT=1  SANITARY PAD  Conf={result['confidence']:.2f}  Tex={result['texture']:.2f}" \
          if out==1 else "OUTPUT=0  No sanitary pad detected"
    cv2.putText(display, msg, (10,fh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, bar_col, 2)

    # Top info bar
    cv2.putText(display, f"FPS:{fps:.1f}",
                (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0,220,0), 1)
    cv2.putText(display, f"Next scan:{next_scan_in:.1f}s",
                (90,22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0,200,255), 1)
    cv2.putText(display, f"Servo:{servo_status}",
                (240,22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220,200,0), 1)

    return display


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  AHP WASTE SEGREGATION — DETECTION SYSTEM 3")
    print("="*60)

    model, db_emb = load_model()
    arduino       = Arduino()

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"  ERROR: Camera {CAMERA_ID} not found.")
        print(f"  Change CAMERA_ID = 1 or 2 at the top of this file.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    conf_t = CONF_THRESHOLD
    sim_t  = SIM_THRESHOLD
    tex_t  = TEX_THRESHOLD

    log          = []
    last_detect  = time.time()
    fps_counter  = fps_timer = fps = 0
    last_result  = {"output":0,"confidence":0.0,"texture":0.0,"similarity":0.0}

    print("\n  Camera open.")
    print(f"  Detecting every {DETECT_EVERY} seconds.")
    print("  Press Q to quit  |  + easier  |  - stricter\n")
    print(f"  {'TIME':>10}   OUT   CONF    TEX    SIM    ACTION")
    print("  " + "─"*55)

    while True:
        ret, frame = cap.read()
        if not ret: continue

        fps_counter += 1
        if fps_counter >= 20:
            fps      = fps_counter / max(time.time()-fps_timer, 0.001)
            fps_timer = time.time(); fps_counter = 0

        now          = time.time()
        next_scan_in = max(0.0, DETECT_EVERY - (now - last_detect))

        # ── Run detection every DETECT_EVERY seconds ──────────────────────
        if now - last_detect >= DETECT_EVERY:
            last_detect  = now
            last_result  = get_detection(frame, model, db_emb, conf_t, sim_t, tex_t)
            out          = last_result["output"]
            t_s          = time.strftime("%H:%M:%S")

            arduino.send(out)

            if out == 1:
                print(f"  [{t_s}]   1    "
                      f"{last_result['confidence']:.2f}   "
                      f"{last_result['texture']:.2f}   "
                      f"{last_result['similarity']:.2f}   "
                      f"→ Sent 1 to Arduino → FLAPS LEFT 45°  ← SANITARY PAD")
                log.append({**last_result, "time": t_s})
                with open(LOG_FILE,"w") as f:
                    json.dump(log, f, indent=2)
            else:
                print(f"  [{t_s}]   0    "
                      f"{last_result['confidence']:.2f}   "
                      f"{last_result['texture']:.2f}   "
                      f"{last_result['similarity']:.2f}   "
                      f"→ Sent 0 to Arduino → No movement")

        # ── Draw ──────────────────────────────────────────────────────────
        objects = find_objects(frame)
        display = draw_result(frame, last_result, objects,
                              conf_t, sim_t, tex_t, fps, next_scan_in, "READY")

        cv2.imshow("AHP Detection System  [Q=quit  +=easier  -=stricter]", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'),ord('Q')): break
        elif key in (ord('+'),ord('=')):
            conf_t=max(0.20,conf_t-0.05); sim_t=max(0.20,sim_t-0.05); tex_t=max(0.10,tex_t-0.05)
            print(f"  Thresholds ↓  Conf≥{conf_t:.2f}  Sim≥{sim_t:.2f}  Tex≥{tex_t:.2f}")
        elif key == ord('-'):
            conf_t=min(0.95,conf_t+0.05); sim_t=min(0.95,sim_t+0.05); tex_t=min(0.95,tex_t+0.05)
            print(f"  Thresholds ↑  Conf≥{conf_t:.2f}  Sim≥{sim_t:.2f}  Tex≥{tex_t:.2f}")

    cap.release()
    arduino.close()
    cv2.destroyAllWindows()
    print("\n  Stopped.\n")


if __name__ == "__main__":
    main()
