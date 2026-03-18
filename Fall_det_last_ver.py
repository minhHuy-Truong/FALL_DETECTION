import os, sys, cv2, time, argparse, warnings, threading, math
import numpy as np
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "2"
os.environ["OPENCV_LOG_LEVEL"]       = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

CONFIG = {
    # ── Paths ─────────────────────────────────────────────────────────
    "dataset_dir":           "dataset_v8",
    "real_test_dir":         "dataset_v8/real_test_v8",
    "processed_dir":         "data_processed_v8",
    "hard_neg_dir":          "dataset_v8/hard_negatives",
    "model_path":            "model_v8/best_model.keras",
    "tflite_path":           "model_v8/best_model.tflite",
    "log_path":              "fall_log_v8.csv",

    # ── Sequence ──────────────────────────────────────────────────────
    "sequence_length":       60,
    "extract_stride":        5,
    "min_visibility":        0.20,   # nới lỏng hơn v7 (0.25) cho điều kiện tối

    # ── Training ──────────────────────────────────────────────────────
    "epochs":                40,
    "batch_size":            128,
    "learning_rate":         0.001,
    "early_stop_patience":   10,

    # ── Focal Loss (tăng alpha vì fall là class thiểu số nặng) ────────
    "focal_alpha":           0.60,   # v7: 0.4 → v8: 0.6 (300 fall vs 1500 not_fall = 1:5)
    "focal_gamma":           2.5,    # v7: 2.0 → v8: 2.5 (focus harder on hard examples)

    # ── Label smoothing + Mixup ───────────────────────────────────────
    "label_smoothing":       0.08,   # nới nhẹ để không mất signal fall
    "mixup_alpha":           0.15,   # nhỏ hơn v7 (0.2) vì dataset imbalanced

    # ── SMOTE-Temporal ────────────────────────────────────────────────
    # Tạo fall sequences tổng hợp bằng cách nội suy giữa 2 fall sequences
    "smote_temporal":        True,
    "smote_k_neighbors":     5,      # k nearest neighbors trong không gian feature
    "smote_ratio":           2.0,    # tạo thêm 2x số fall sequences hiện có

    # ── LR schedule ───────────────────────────────────────────────────
    "lr_schedule":           "cosine_warm",  # mới: cosine với warm restart
    "lr_warmup_epochs":      3,

    # ── Confidence & confirm ──────────────────────────────────────────
    "confidence_threshold":  0.60,   # v7: 0.65 → v8: 0.60 (nhạy hơn, rule engine bù)
    "confirm_frames":        2,
    "alert_debounce_sec":    2.0,
    "accum_threshold":       0.68,   # v7: 0.70 → v8: 0.68

    # ── Bayesian smoother ─────────────────────────────────────────────
    # Làm mượt xác suất theo thời gian: p_smooth = α*p_raw + (1-α)*p_prev
    "bayes_alpha":           0.35,   # 0=không cập nhật, 1=raw prob thuần túy
    "bayes_rise_alpha":      0.60,   # khi prob tăng nhanh → phản ứng nhanh hơn

    # ── Post-fall ─────────────────────────────────────────────────────
    "stillness_window":      45,
    "stillness_threshold":   0.04,
    "stillness_min_frames":  10,
    "recovery_window_sec":   3.0,

    # ── Jerk & window stats ───────────────────────────────────────────
    "jerk_scale":            30,
    "head_rise_frames":      20,
    "head_rise_threshold":   0.05,

    # ── Ensemble ──────────────────────────────────────────────────────
    "ensemble_weights":      [0.72, 0.28],  # v7: [0.75, 0.25] → rule engine mạnh hơn

    # ── Multi-person ──────────────────────────────────────────────────
    "max_persons":           3,

    # ── Scene-aware threshold ─────────────────────────────────────────
    # Tự động điều chỉnh threshold theo nhận diện scene
    # top_down: camera nhìn từ trên xuống → threshold cao hơn (nhiều FP)
    # low_light: ánh sáng yếu → threshold thấp hơn (visibility giảm)
    "scene_aware_threshold": True,
    "scene_topdown_boost":   0.08,   # tăng threshold khi top-down camera
    "scene_lowlight_reduce": 0.05,   # giảm threshold khi tối

    # ── Auto threshold (F2-score) ─────────────────────────────────────
    "auto_threshold":        True,

    # ── Telegram ──────────────────────────────────────────────────────
    "telegram_token":        "",
    "telegram_chat_id":      "",
    "telegram_send_image":   True,

    # ── Hard negative mining ──────────────────────────────────────────
    "hard_neg_threshold":    0.50,
    "hard_neg_clip_sec":     3.0,
    "hard_neg_filter_prob":  0.80,

    # ── Dataset recorder ──────────────────────────────────────────────
    "recorder_dir":          "dataset_recorder_v8",
    "recorder_ud_dir":       "dataset_v8",
    "recorder_start_idx":    0,

    # ── Incremental learning ──────────────────────────────────────────
    "incremental_lr":        0.00008,  # v7: 0.0001 → nhỏ hơn để bảo toàn knowledge
    "incremental_epochs":    10,
    "incremental_patience":  4,
    "extract_registry":      "data_processed_v8/extracted_files.txt",

    # ── Fine-tune real ────────────────────────────────────────────────
    "finetune_real_lr":           0.00004,  # v7: 0.00005
    "finetune_real_epochs":       20,       # v7: 15
    "finetune_real_patience":     6,
    "finetune_real_replay_ratio": 0.40,
    "finetune_real_max_seqs":     6000,
    "finetune_real_model_path":   "model_v8/best_model_ft_real.keras",

    # ── Model architecture ────────────────────────────────────────────
    # "ms_tcn" : Multi-scale TCN + CBAM (MỚI, recommended)
    # "tcn"    : TCN thuần (v7 cũ, fallback)
    # "bilstm" : BiLSTM+Attention (legacy)
    "model_arch":            "ms_tcn",

    # ── Multi-scale TCN params ────────────────────────────────────────
    "tcn_filters":           96,     # v7: 64 → v8: 96 (nhiều features hơn)
    "tcn_kernels":           [3, 5, 7],    # multi-scale: ngắn/trung/dài
    "tcn_dilations":         [1, 2, 4, 8], # receptive field: 3*(1+2+4+8)*2 = 90 frames
    "tcn_dropout":           0.30,
    "cbam_reduction":        8,      # CBAM channel reduction ratio

    # ── Auxiliary loss ────────────────────────────────────────────────
    # Dự đoán thêm "frame nào là điểm ngã" (binary per-frame)
    # Giúp model học feature temporal tốt hơn
    "aux_loss_weight":       0.20,   # trọng số auxiliary loss trong total loss
    "aux_fall_window":       10,     # số frame quanh điểm ngã được label = 1

    # ── FP/FN Confusion tracker ───────────────────────────────────────
    "confusion_log":         "model_v8/confusion_log.csv",
    "fp_rule_log":           "model_v8/fp_rule_stats.csv",

    # ── Device & inference ────────────────────────────────────────────
    "device":                "laptop",
    "use_lite_pose":         False,
    "use_tflite":            False,

    # ── Frame processing ──────────────────────────────────────────────
    "frame_resize":          (640, 480),  # tăng từ 480x360 để display sắc nét hơn
    "display_size":          (960, 720),

    # ── Classes ───────────────────────────────────────────────────────
    "classes":               ["not_fall", "fall"],
}

# ======================================================================
#  FEATURE SCHEMA  (38 features)
#
#  [0-18]  2D features (19) — giống v7
#  [19-23] 3D features (5)  — giống v7
#  [24]    near_camera_score
#  [25]    max_v_hip_15f
#  [26]    std_spine_15f
#  [27]    descent_ratio
#  ── MỚI v8 ──────────────────────────────────────────
#  [28]    hip_accel_mag    — độ lớn gia tốc hip 3D (sqrt(ax²+ay²+az²))
#                             nga: peak cao đột ngột; ngồi: tăng dần
#  [29]    asymmetry_score  — bất đối xứng trái/phải (|left-right| velocity)
#                             nga sang một bên: cao; ngồi thẳng: thấp
#  [30]    pose_energy      — tổng energy của tất cả joints trong window 20f
#                             trước ngã: cao (chuẩn bị); trong ngã: rất cao
#  [31]    fall_impulse     — tích phân velocity*acceleration (impulse)
#                             ngã thật: spike mạnh trong ~5 frame
#                             ngồi: phân bố đều
#  [32]    lateral_tilt_rate— tốc độ thay đổi sh_tilt theo thời gian
#                             ngã sang bên: cao; ngã về trước: thấp
#  [33]    knee_extension_rate — tốc độ knee_avg thay đổi
#                             ngồi: gập nhanh; ngã: duỗi đột ngột
#  [34]    arm_flail        — biên độ cổ tay so với vai (mất thăng bằng)
#                             ngã: tay vung ra; đứng: tay gần người
#  [35]    head_accel_mag   — gia tốc đầu (3D magnitude)
#                             ngã đầu xuống trước: spike đặc trưng
#  [36]    body_sway        — dao động ngang của hip trong 30f
#                             ngã do chóng mặt: cao; ngồi: thấp
#  [37]    impact_score     — jerk peak sau velocity peak (va chạm sàn)
#                             ngã thật: có impact rõ; ngồi: không có
# ======================================================================
N_FEATURES = 38

LM = {
    "nose":0, "l_ear":7, "r_ear":8,
    "l_shoulder":11, "r_shoulder":12,
    "l_elbow":13,    "r_elbow":14,
    "l_wrist":15,    "r_wrist":16,
    "l_hip":23,      "r_hip":24,
    "l_knee":25,     "r_knee":26,
    "l_ankle":27,    "r_ankle":28,
}
KEY_VIS = [0, 11, 12, 23, 24, 25, 26, 27, 28]

def banner(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")
def step(m):   print(f"\n  >  {m}")
def ok(m):     print(f"  OK {m}")
def warn(m):   print(f"  !  {m}")
def err(m):    print(f"  X  {m}")


# ======================================================================
#  TELEGRAM ALERT
# ======================================================================
def _telegram_send(message: str, image_bgr=None):
    token   = CONFIG.get("telegram_token", "").strip()
    chat_id = CONFIG.get("telegram_chat_id", "").strip()
    if not token or not chat_id:
        return

    def _send():
        try:
            import urllib.request, urllib.parse, json
            data = urllib.parse.urlencode({
                "chat_id": chat_id, "text": message, "parse_mode": "HTML"
            }).encode()
            urllib.request.urlopen(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data=data, timeout=10)
            if image_bgr is not None and CONFIG.get("telegram_send_image", True):
                ok_flag, buf = cv2.imencode(".jpg", image_bgr,
                                            [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok_flag:
                    img_bytes = buf.tobytes()
                    boundary  = b"----FallDetBoundary"
                    body  = b"--" + boundary + b"\r\n"
                    body += b'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
                    body += chat_id.encode() + b"\r\n"
                    body += b"--" + boundary + b"\r\n"
                    body += b'Content-Disposition: form-data; name="photo"; filename="fall.jpg"\r\n'
                    body += b"Content-Type: image/jpeg\r\n\r\n"
                    body += img_bytes + b"\r\n"
                    body += b"--" + boundary + b"--\r\n"
                    req = urllib.request.Request(
                        f"https://api.telegram.org/bot{token}/sendPhoto",
                        data=body,
                        headers={"Content-Type":
                                 f"multipart/form-data; boundary={boundary.decode()}"})
                    urllib.request.urlopen(req, timeout=15)
        except Exception as e:
            warn(f"Telegram send failed: {e}")

    threading.Thread(target=_send, daemon=True).start()


# ======================================================================
#  FRAME ENHANCEMENT  (giữ nguyên từ v7, cải tiến nhỏ)
# ======================================================================
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def _is_backlit(frame):
    h, w   = frame.shape[:2]
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    center = gray[h//4:3*h//4, w//4:3*w//4].mean()
    border = (gray[:h//4].mean() + gray[3*h//4:].mean()) / 2
    return center < border * 0.6

def _is_yellow_light(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (15, 40, 40), (35, 255, 255))
    yellow_ratio = mask.sum() / (mask.shape[0] * mask.shape[1] * 255.0)
    return yellow_ratio > 0.25

def _is_greenish_fluorescent(frame):
    """Phát hiện đèn huỳnh quang (văn phòng, bệnh viện) — màu xanh lá nhạt."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 15, 50), (85, 80, 255))
    return mask.sum() / (mask.shape[0] * mask.shape[1] * 255.0) > 0.30

def enhance_frame(frame):
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_mean = gray.mean()
    dark_level = float(np.clip(1.0 - gray_mean / 80.0, 0.0, 1.0))

    if gray_mean > 80 and not _is_yellow_light(frame) and not _is_greenish_fluorescent(frame):
        return frame

    # Bước 1: White balance
    if _is_yellow_light(frame):
        b_ch, g_ch, r_ch = cv2.split(frame.astype(np.float32))
        r_ch = np.clip(r_ch * 0.85, 0, 255)
        g_ch = np.clip(g_ch * 0.90, 0, 255)
        b_ch = np.clip(b_ch * 1.15, 0, 255)
        frame = cv2.merge([b_ch, g_ch, r_ch]).astype(np.uint8)
    elif _is_greenish_fluorescent(frame):
        b_ch, g_ch, r_ch = cv2.split(frame.astype(np.float32))
        g_ch = np.clip(g_ch * 0.92, 0, 255)
        r_ch = np.clip(r_ch * 1.05, 0, 255)
        frame = cv2.merge([b_ch, g_ch, r_ch]).astype(np.uint8)

    # Bước 2: Xử lý ngược sáng
    if _is_backlit(frame):
        frame = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.1)

    # Bước 3: CLAHE trên kênh L (LAB)
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    if dark_level > 0.6:
        clahe_strong = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        l = clahe_strong.apply(l)
    else:
        l = _clahe.apply(l)
    frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Bước 4: Gamma correction
    if dark_level > 0.3:
        gamma = 1.0 + dark_level * 0.8
        inv_g = 1.0 / gamma
        lut   = np.array([((i / 255.0) ** inv_g) * 255
                          for i in range(256)], dtype=np.uint8)
        frame = cv2.LUT(frame, lut)

    return frame


# ======================================================================
#  MEDIAPIPE POSE SETUP
# ======================================================================
def _safe_model_path(lite=False):
    safe_dir = r"C:\mediapipe_models" if sys.platform == "win32" else "/tmp/mediapipe_models"
    os.makedirs(safe_dir, exist_ok=True)
    fname = "pose_landmarker_lite.task" if lite else "pose_landmarker_full.task"
    return os.path.join(safe_dir, fname)

def _ensure_pose_model(lite=False):
    model_path = _safe_model_path(lite)
    if not os.path.exists(model_path):
        variant = "lite" if lite else "full"
        url = (f"https://storage.googleapis.com/mediapipe-models/"
               f"pose_landmarker/pose_landmarker_{variant}/float16/latest/"
               f"pose_landmarker_{variant}.task")
        step(f"Downloading pose model ({variant}) -> {model_path}")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, model_path)
            ok(f"Downloaded: {model_path}")
        except Exception as e:
            err(f"Download failed: {e}\n  Manual: {url}\n  Save to: {model_path}")
            sys.exit(1)
    return model_path

def _make_pose_detector(running_mode_str="IMAGE"):
    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        err("pip install mediapipe>=0.10.0"); sys.exit(1)

    lite = CONFIG.get("use_lite_pose", False)
    model_path = _ensure_pose_model(lite)

    mode_map = {
        "IMAGE": mp_vision.RunningMode.IMAGE,
        "VIDEO": mp_vision.RunningMode.VIDEO,
        "LIVE":  mp_vision.RunningMode.LIVE_STREAM,
    }
    running_mode = mode_map.get(running_mode_str, mp_vision.RunningMode.IMAGE)

    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=running_mode,
        num_poses=CONFIG.get("max_persons", 3),
        min_pose_detection_confidence=0.45,   # v7: 0.5 → nhạy hơn trong môi trường tối
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)

def _run_pose_on_frame(detector, frame, timestamp_ms=None, use_video_mode=False):
    try:
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if use_video_mode and timestamp_ms is not None:
            result = detector.detect_for_video(mp_img, int(timestamp_ms))
        else:
            result = detector.detect(mp_img)
        if not result.pose_landmarks:
            return None
        # Trả về list (landmarks_2d, world_landmarks) cho từng người
        out = []
        for lm2d, lmw in zip(result.pose_landmarks,
                              result.pose_world_landmarks or [None]*len(result.pose_landmarks)):
            out.append((lm2d, lmw))
        return out if out else None
    except Exception:
        return None


# ======================================================================
#  EXTRACT FEATURES  (38 features)
# ======================================================================
def extract_features(landmark_pair, prev_state, feat_history=None):
    """
    landmark_pair: (lm2d, lm_world) từ MediaPipe
    prev_state: dict chứa giá trị frame trước
    feat_history: list các feature vector gần đây (tối đa 30)
    Trả về: (feature_vector[38], new_state)
    """
    lm2d, lm_world = landmark_pair if isinstance(landmark_pair, tuple) else (landmark_pair, None)

    def pt2(idx):
        p = lm2d[idx]
        return np.array([p.x, p.y], dtype=np.float32)

    def vis(idx):
        return float(lm2d[idx].visibility or 0)

    def pt3(idx):
        if lm_world is None:
            return np.zeros(3, dtype=np.float32)
        p = lm_world[idx]
        return np.array([p.x, p.y, p.z], dtype=np.float32)

    # ── Landmark 2D ──────────────────────────────────────────────────
    nose      = pt2(LM["nose"])
    l_sh      = pt2(LM["l_shoulder"]); r_sh  = pt2(LM["r_shoulder"])
    l_hip     = pt2(LM["l_hip"]);      r_hip = pt2(LM["r_hip"])
    l_kn      = pt2(LM["l_knee"]);     r_kn  = pt2(LM["r_knee"])
    l_an      = pt2(LM["l_ankle"]);    r_an  = pt2(LM["r_ankle"])
    l_wr      = pt2(LM["l_wrist"]);    r_wr  = pt2(LM["r_wrist"])
    l_el      = pt2(LM["l_elbow"]);    r_el  = pt2(LM["r_elbow"])

    mid_sh    = (l_sh + r_sh)   / 2
    mid_hip   = (l_hip + r_hip) / 2
    mid_kn    = (l_kn + r_kn)   / 2
    mid_an    = (l_an + r_an)   / 2

    # ── Landmark 3D ──────────────────────────────────────────────────
    sh3_l  = pt3(LM["l_shoulder"]); sh3_r  = pt3(LM["r_shoulder"])
    hip3_l = pt3(LM["l_hip"]);      hip3_r = pt3(LM["r_hip"])
    kn3_l  = pt3(LM["l_knee"]);     kn3_r  = pt3(LM["r_knee"])
    nose3  = pt3(LM["nose"])

    mid_sh3  = (sh3_l + sh3_r)   / 2
    mid_hip3 = (hip3_l + hip3_r) / 2

    # ── Spine vector ──────────────────────────────────────────────────
    spine_vec   = mid_hip - mid_sh
    spine_len   = float(np.linalg.norm(spine_vec)) + 1e-6
    spine_angle = float(abs(spine_vec[0]) / spine_len)   # 0=thẳng đứng, 1=nằm ngang
    spine_horiz = float(abs(spine_vec[0]) / (abs(spine_vec[1]) + 1e-6))
    spine_horiz = float(np.clip(spine_horiz / 3.0, 0, 1))

    # ── Tilt ──────────────────────────────────────────────────────────
    sh_tilt   = float(abs(l_sh[1]  - r_sh[1]))
    hip_tilt  = float(abs(l_hip[1] - r_hip[1]))

    # ── Bounding box ──────────────────────────────────────────────────
    all_pts  = np.stack([nose, l_sh, r_sh, l_hip, r_hip,
                         l_kn, r_kn, l_an, r_an])
    xmin, ymin = all_pts.min(0)
    xmax, ymax = all_pts.max(0)
    bx_w = float(xmax - xmin) + 1e-6
    bx_h = float(ymax - ymin) + 1e-6
    bbox_ar     = float(np.clip(bx_w / bx_h, 0, 5))
    body_h_norm = float(np.clip(bx_h / bx_w, 0, 5))

    # ── Relative distances (normalized bằng spine 3D) ─────────────────
    spine3_vec = mid_hip3 - mid_sh3
    spine3_len = float(np.linalg.norm(spine3_vec)) + 1e-6
    norm_f     = spine3_len * 2.0

    head_hip_dy  = float(abs(nose[1]   - mid_hip[1]) / (bx_h + 1e-6))
    sh_hip_dy    = float(abs(mid_sh[1] - mid_hip[1]) / (bx_h + 1e-6))
    hip_knee_dy  = float(abs(mid_hip[1]- mid_kn[1])  / (bx_h + 1e-6))

    # ── Knee angles ───────────────────────────────────────────────────
    def _knee_angle(hip, knee, ankle):
        v1 = hip - knee; v2 = ankle - knee
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6: return 0.5
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        return float(np.clip(np.arccos(cos_a) / np.pi, 0, 1))

    knee_l   = _knee_angle(l_hip, l_kn, l_an)
    knee_r   = _knee_angle(r_hip, r_kn, r_an)
    knee_avg = (knee_l + knee_r) / 2.0

    # ── Visibility ────────────────────────────────────────────────────
    all_vis      = [vis(i) for i in range(33)]
    mean_vis     = float(np.mean(all_vis))
    min_core_vis = float(min(vis(i) for i in KEY_VIS))

    # ── 3D features ───────────────────────────────────────────────────
    if lm_world is not None:
        sh_z_diff      = float(abs(sh3_l[2]  - sh3_r[2]))
        hip_z_diff     = float(abs(hip3_l[2] - hip3_r[2]))
        spine_depth_tilt = float(np.clip((sh_z_diff + hip_z_diff) / 2.0, 0, 1))

        Y_axis = np.array([0, 1, 0], dtype=np.float32)
        sp3    = spine3_vec / (spine3_len + 1e-6)
        cos_sp = float(np.clip(np.dot(sp3, Y_axis), -1, 1))
        spine_angle_3d = float(np.clip(1.0 - abs(cos_sp), 0, 1))

        trunk3_vec = mid_sh3 - mid_hip3
        trunk3_len = float(np.linalg.norm(trunk3_vec)) + 1e-6
        t3n = trunk3_vec / trunk3_len
        trunk_angle_3d = float(np.clip(abs(t3n[2]), 0, 1))
    else:
        spine_depth_tilt = 0.0
        spine_angle_3d   = 0.0
        trunk_angle_3d   = 0.0

    # ── Near-camera score ──────────────────────────────────────────────
    body_frac        = float(bx_w * bx_h)
    near_camera_score= float(np.clip((body_frac - 0.15) / 0.35, 0, 1))

    # ── State-based velocity / acceleration ───────────────────────────
    cur_hip_y  = float(mid_hip[1])
    cur_nose_y = float(nose[1])
    cur_hip_z  = float(mid_hip3[2]) if lm_world else 0.0
    cur_nose_3 = nose3.copy()
    cur_mid_hip3 = mid_hip3.copy()

    if prev_state is None:
        v_hip_y = v_nose_y = a_hip_y = a_nose_y = j_hip_y = 0.0
        v_hip_z = a_hip_z = 0.0
        v_nose_3 = np.zeros(3)
        v_hip_3  = np.zeros(3)
        a_hip_3  = np.zeros(3)
    else:
        v_hip_y  = cur_hip_y  - prev_state["hip_y"]
        v_nose_y = cur_nose_y - prev_state["nose_y"]
        v_hip_z  = cur_hip_z  - prev_state["hip_z"]

        pv_hip  = prev_state.get("v_hip_y",  0.0)
        pv_nose = prev_state.get("v_nose_y", 0.0)
        pv_hip_z= prev_state.get("v_hip_z",  0.0)
        a_hip_y  = v_hip_y  - pv_hip
        a_nose_y = v_nose_y - pv_nose
        a_hip_z  = v_hip_z  - pv_hip_z

        pa_hip  = prev_state.get("a_hip_y",  0.0)
        j_hip_y = a_hip_y - pa_hip

        v_nose_3 = cur_nose_3  - prev_state.get("nose_3", cur_nose_3)
        v_hip_3  = cur_mid_hip3 - prev_state.get("hip_3", cur_mid_hip3)
        pa_hip_3 = prev_state.get("a_hip_3", np.zeros(3))
        a_hip_3  = v_hip_3  - prev_state.get("v_hip_3", np.zeros(3))

    js  = CONFIG.get("jerk_scale", 30)

    # Cập nhật state
    cur = {
        "hip_y":   cur_hip_y,  "nose_y": cur_nose_y,
        "hip_z":   cur_hip_z,
        "v_hip_y": v_hip_y,    "v_nose_y": v_nose_y,
        "v_hip_z": v_hip_z,
        "a_hip_y": a_hip_y,    "a_nose_y": a_nose_y,
        "a_hip_z": a_hip_z,
        "nose_3":  cur_nose_3, "hip_3": cur_mid_hip3,
        "v_nose_3":v_nose_3,   "v_hip_3": v_hip_3,
        "a_hip_3": a_hip_3 if prev_state else np.zeros(3),
    }

    # ── Window statistics (features [25-27]) ──────────────────────────
    if feat_history and len(feat_history) >= 5:
        hist_arr      = np.array(feat_history[-30:], dtype=np.float32)
        hist_v_hip    = hist_arr[:, 12]
        hist_spine    = hist_arr[:, 1]

        max_v_hip_15f = float(np.max(hist_arr[-15:, 12])) if len(hist_arr) >= 15 \
                        else float(np.max(hist_v_hip))
        max_v_hip_15f = float(np.clip(max_v_hip_15f, 0, 1))

        std_spine_15f = float(np.std(hist_arr[-15:, 1])) if len(hist_arr) >= 15 \
                        else float(np.std(hist_spine))
        std_spine_15f = float(np.clip(std_spine_15f * 5, 0, 1))

        n_descending  = float(np.sum(hist_v_hip > 0.015))
        descent_ratio = float(np.clip(n_descending / max(len(hist_v_hip), 1), 0, 1))
    else:
        max_v_hip_15f = std_spine_15f = descent_ratio = 0.0

    # ── NEW features [28-37] ──────────────────────────────────────────
    # [28] hip_accel_mag: 3D acceleration magnitude
    if prev_state is not None:
        a_hip_3_cur = cur.get("a_hip_3", np.zeros(3))
        hip_accel_mag = float(np.clip(np.linalg.norm(a_hip_3_cur) * 20, 0, 1))
    else:
        hip_accel_mag = 0.0

    # [29] asymmetry_score: bất đối xứng chuyển động trái/phải
    # Dùng sự khác biệt vị trí Y của l_hip và r_hip thay đổi theo thời gian
    if feat_history and len(feat_history) >= 3:
        # Ước lượng từ sh_tilt + hip_tilt biến động trong 10f
        tilt_arr = np.array([f[2] for f in feat_history[-10:]], dtype=np.float32)  # sh_tilt
        asymmetry_score = float(np.clip(np.std(tilt_arr) * 8 + (sh_tilt + hip_tilt) / 2, 0, 1))
    else:
        asymmetry_score = float(np.clip((sh_tilt + hip_tilt) / 2, 0, 1))

    # [30] pose_energy: tổng kinetic energy của tất cả joints
    if feat_history and len(feat_history) >= 5:
        # Sử dụng velocity fields (idx 12,13) + 3D (idx 22,23) như proxy
        energy_arr = np.array(feat_history[-20:], dtype=np.float32)
        v_cols = [12, 13, 14, 15, 16, 22, 23]  # velocity và acceleration fields
        valid_cols = [c for c in v_cols if c < energy_arr.shape[1]]
        pose_energy = float(np.clip(np.mean(energy_arr[:, valid_cols] ** 2) * 10, 0, 1))
    else:
        pose_energy = 0.0

    # [31] fall_impulse: tích phân |v * a| trong 5 frame = dấu hiệu va chạm đột ngột
    if feat_history and len(feat_history) >= 5:
        imp_arr = np.array(feat_history[-5:], dtype=np.float32)
        v5 = imp_arr[:, 12]  # v_hip_y
        a5 = imp_arr[:, 14]  # a_hip_y
        fall_impulse = float(np.clip(np.sum(np.abs(v5 * a5)) * 15, 0, 1))
    else:
        fall_impulse = 0.0

    # [32] lateral_tilt_rate: tốc độ thay đổi sh_tilt (ngã sang bên: rate cao)
    if feat_history and len(feat_history) >= 3:
        tilt_last3 = [f[2] for f in feat_history[-3:]]
        lateral_tilt_rate = float(np.clip(abs(tilt_last3[-1] - tilt_last3[0]) * 5, 0, 1))
    else:
        lateral_tilt_rate = 0.0

    # [33] knee_extension_rate: tốc độ duỗi gối (ngã: gối duỗi đột ngột)
    if feat_history and len(feat_history) >= 5:
        kn_arr   = np.array([f[11] for f in feat_history[-5:]], dtype=np.float32)  # knee_avg
        kn_early = float(np.mean(kn_arr[:2]))
        kn_late  = float(np.mean(kn_arr[-2:]))
        # duỗi = tăng; gập = giảm → ngã: duỗi đột ngột
        knee_extension_rate = float(np.clip((kn_late - kn_early) * 4, 0, 1))
    else:
        knee_extension_rate = 0.0

    # [34] arm_flail: biên độ vung tay (mất thăng bằng trước khi ngã)
    # Khoảng cách cổ tay so với thân người (mid_sh)
    l_wr_dist = float(np.linalg.norm(l_wr - mid_sh))
    r_wr_dist = float(np.linalg.norm(r_wr - mid_sh))
    arm_flail = float(np.clip((l_wr_dist + r_wr_dist) / 2.0 / (bx_h + 1e-6), 0, 1))

    # [35] head_accel_mag: gia tốc đầu 3D (ngã đầu xuống: spike đặc trưng)
    if prev_state is not None:
        v_nose_3_cur = cur.get("v_nose_3", np.zeros(3))
        pv_nose_3    = prev_state.get("v_nose_3", np.zeros(3))
        a_nose_3_mag = float(np.linalg.norm(v_nose_3_cur - pv_nose_3))
        head_accel_mag = float(np.clip(a_nose_3_mag * 15, 0, 1))
    else:
        head_accel_mag = 0.0

    # [36] body_sway: dao động ngang của hip trong 30f (ngã do chóng mặt)
    if feat_history and len(feat_history) >= 10:
        # Sử dụng v_hip_x proxy: biến động của spine_horiz trong 30f
        sway_arr = np.array([f[1] for f in feat_history[-30:]], dtype=np.float32)  # spine_horiz
        body_sway = float(np.clip(np.std(sway_arr) * 6, 0, 1))
    else:
        body_sway = 0.0

    # [37] impact_score: jerk peak sau velocity peak (va chạm sàn)
    if feat_history and len(feat_history) >= 10:
        vhip_10 = np.array([f[12] for f in feat_history[-10:]], dtype=np.float32)
        jhip_10 = np.array([f[16] for f in feat_history[-10:]], dtype=np.float32)
        v_peak_idx = int(np.argmax(vhip_10))
        # Impact = jerk cao SAU khi velocity peak (va chạm với mặt đất)
        if v_peak_idx < len(jhip_10) - 1:
            post_jerk = float(np.max(np.abs(jhip_10[v_peak_idx:])))
        else:
            post_jerk = 0.0
        impact_score = float(np.clip(post_jerk * 2, 0, 1))
    else:
        impact_score = 0.0

    feats = np.array([
        # ── 2D (19) ───────────────────────────────────────────────────
        spine_angle,          # [0]
        spine_horiz,          # [1]
        sh_tilt,              # [2]
        hip_tilt,             # [3]
        bbox_ar,              # [4]
        body_h_norm,          # [5]
        head_hip_dy,          # [6]
        sh_hip_dy,            # [7]
        hip_knee_dy,          # [8]
        knee_l, knee_r, knee_avg,  # [9,10,11]
        np.clip(v_hip_y  * 10, -1, 1),   # [12]
        np.clip(v_nose_y * 10, -1, 1),   # [13]
        np.clip(a_hip_y  * 20, -1, 1),   # [14]
        np.clip(a_nose_y * 20, -1, 1),   # [15]
        np.clip(j_hip_y  * js, -1, 1),   # [16]
        mean_vis, min_core_vis,           # [17,18]
        # ── 3D (5) ────────────────────────────────────────────────────
        spine_depth_tilt,     # [19]
        spine_angle_3d,       # [20]
        trunk_angle_3d,       # [21]
        np.clip(v_hip_z * 30, -1, 1),    # [22]
        np.clip(a_hip_z * 50, -1, 1),    # [23]
        # ── Guards & window stats (4) ──────────────────────────────────
        near_camera_score,    # [24]
        max_v_hip_15f,        # [25]
        std_spine_15f,        # [26]
        descent_ratio,        # [27]
        # ── NEW features v8 (10) ──────────────────────────────────────
        hip_accel_mag,        # [28]
        asymmetry_score,      # [29]
        pose_energy,          # [30]
        fall_impulse,         # [31]
        lateral_tilt_rate,    # [32]
        knee_extension_rate,  # [33]
        arm_flail,            # [34]
        head_accel_mag,       # [35]
        body_sway,            # [36]
        impact_score,         # [37]
    ], dtype=np.float32)

    assert len(feats) == N_FEATURES, f"Feature mismatch: {len(feats)} vs {N_FEATURES}"
    return feats, cur


# ======================================================================
#  RULE-BASED ENSEMBLE v3
#
#  10 Negative Gates + 14 Positive Rules
#  Thiết kế mới: mỗi gate có trọng số, tổng gate = weighted suppression
# ======================================================================
def rule_based_score(buf_feats: list) -> float:
    if len(buf_feats) < 10:
        return 0.0

    n_buf   = len(buf_feats)
    buf_arr = np.array(buf_feats, dtype=np.float32)
    recent  = buf_arr[-15:]
    cur     = buf_feats[-1]

    # ── Feature indices ───────────────────────────────────────────────
    I_SPINE_HORIZ  = 1
    I_SH_TILT      = 2
    I_HIP_TILT     = 3
    I_BBOX_AR      = 4
    I_HEAD_HIP_DY  = 6
    I_SH_HIP_DY    = 7
    I_KNEE_AVG     = 11
    I_V_HIP_Y      = 12
    I_V_NOSE_Y     = 13
    I_A_HIP_Y      = 14
    I_A_NOSE_Y     = 15
    I_J_HIP_Y      = 16
    I_SPINE_DTILT  = 19
    I_SPINE_A3D    = 20
    I_TRUNK_A3D    = 21
    I_V_HIP_Z      = 22
    I_NEAR_CAM     = 24
    I_HIP_ACCEL    = 28
    I_ASYMMETRY    = 29
    I_POSE_ENERGY  = 30
    I_FALL_IMPULSE = 31
    I_LAT_TILT_RT  = 32
    I_KN_EXT_RT    = 33
    I_ARM_FLAIL    = 34
    I_HEAD_ACCEL   = 35
    I_BODY_SWAY    = 36
    I_IMPACT       = 37

    # ── Scalars ───────────────────────────────────────────────────────
    bbox_ar       = float(cur[I_BBOX_AR])
    spine_horiz   = float(cur[I_SPINE_HORIZ])
    knee_avg      = float(cur[I_KNEE_AVG])
    head_hip_dy   = float(cur[I_HEAD_HIP_DY])
    v_nose        = float(cur[I_V_NOSE_Y])
    a_nose        = float(cur[I_A_NOSE_Y])
    j_hip         = float(cur[I_J_HIP_Y])
    sh_tilt       = float(cur[I_SH_TILT])
    hip_tilt      = float(cur[I_HIP_TILT])
    spine_dtilt   = float(cur[I_SPINE_DTILT])
    spine_a3d     = float(cur[I_SPINE_A3D])
    trunk_a3d     = float(cur[I_TRUNK_A3D])
    v_hip_z       = float(cur[I_V_HIP_Z])
    near_cam      = float(cur[I_NEAR_CAM])
    hip_accel     = float(cur[I_HIP_ACCEL])   if len(cur) > I_HIP_ACCEL  else 0.0
    asymmetry     = float(cur[I_ASYMMETRY])   if len(cur) > I_ASYMMETRY  else 0.0
    pose_energy   = float(cur[I_POSE_ENERGY]) if len(cur) > I_POSE_ENERGY else 0.0
    fall_impulse  = float(cur[I_FALL_IMPULSE])if len(cur) > I_FALL_IMPULSE else 0.0
    lat_tilt_rt   = float(cur[I_LAT_TILT_RT]) if len(cur) > I_LAT_TILT_RT else 0.0
    kn_ext_rt     = float(cur[I_KN_EXT_RT])   if len(cur) > I_KN_EXT_RT  else 0.0
    arm_flail     = float(cur[I_ARM_FLAIL])    if len(cur) > I_ARM_FLAIL  else 0.0
    head_accel    = float(cur[I_HEAD_ACCEL])   if len(cur) > I_HEAD_ACCEL else 0.0
    body_sway     = float(cur[I_BODY_SWAY])    if len(cur) > I_BODY_SWAY  else 0.0
    impact        = float(cur[I_IMPACT])       if len(cur) > I_IMPACT     else 0.0

    # ── Windows ───────────────────────────────────────────────────────
    hip_y_r   = recent[:, I_V_HIP_Y]
    v_hip_r   = recent[:, I_V_HIP_Y]
    sh_tilt_r = recent[:, I_SH_TILT]
    knee_r    = recent[:, I_KNEE_AVG]
    j_hip_r   = recent[:, I_J_HIP_Y]

    # ══════════════════════════════════════════════════════════════════
    #  TẦNG 1: NEGATIVE GATES (10 gates)
    # ══════════════════════════════════════════════════════════════════
    gate_scores = {}

    # G1 — Controlled descent (ngồi từ từ)
    if n_buf >= 20:
        v20   = buf_arr[-20:, I_V_HIP_Y]
        j20   = buf_arr[-20:, I_J_HIP_Y]
        v_std = float(np.std(v20))
        v_max = float(np.max(v20))
        j_max = float(np.max(np.abs(j20)))
        kn20  = buf_arr[-20:, I_KNEE_AVG]
        kn_drop = float(kn20[0] - kn20[-1])
        is_controlled = (
            v_std < 0.07 and v_max < 0.22 and j_max < 0.14
            and float(np.mean(v20[-8:])) > 0.01
            and sh_tilt < 0.08
        )
        if is_controlled:
            gate_scores["G1"] = 0.70

    # G2 — Prolonged descent (> 22/30 frame)
    if n_buf >= 30:
        v30 = buf_arr[-30:, I_V_HIP_Y]
        n_descending = int(np.sum(v30 > 0.018))
        if n_descending >= 22:
            gate_scores["G2"] = 0.75

    # G3 — Recovery pattern (đứng dậy sau khi cúi)
    if len(hip_y_r) >= 8:
        peak_idx = int(np.argmax(hip_y_r))
        if peak_idx <= len(hip_y_r) - 4:
            recovery_vel = float(np.mean(hip_y_r[peak_idx:]))
            if recovery_vel < -0.02:
                gate_scores["G3"] = 0.85

    # G4 — Symmetric descent (ngồi thẳng)
    mean_sh_tilt = float(np.mean(sh_tilt_r))
    mean_v_hip   = float(np.mean(v_hip_r))
    if (mean_sh_tilt < 0.060 and mean_v_hip > 0.012 and spine_horiz < 0.50):
        gate_scores["G4"] = 0.60

    # G5 — Knee-bend finish (ngồi bệt xuống sàn)
    if len(knee_r) >= 8:
        knee_early = float(np.mean(knee_r[:4]))
        knee_late  = float(np.mean(knee_r[-4:]))
        knee_drop  = knee_early - knee_late
        if knee_drop > 0.16 and knee_late < 0.68:
            gate_scores["G5"] = 0.75

    # G6 — Static low posture (nằm tĩnh lâu)
    if n_buf >= 30:
        v_hip_long  = buf_arr[-30:, I_V_HIP_Y]
        spine_long  = buf_arr[-30:, I_SPINE_HORIZ]
        no_spike    = float(np.max(np.abs(v_hip_long))) < 0.18
        spine_low_all = float(np.mean(spine_long)) < 0.40
        had_spike = False
        if n_buf >= 60:
            had_spike = float(np.max(buf_arr[-60:, I_V_HIP_Y])) > 0.25
        if no_spike and spine_low_all and not had_spike:
            gate_scores["G6"] = 0.72

    # G7 — Fast sit-down (ngồi nhanh bị nhầm là ngã)
    if n_buf >= 15:
        v15      = buf_arr[-15:, I_V_HIP_Y]
        knee15   = buf_arr[-15:, I_KNEE_AVG]
        spine15  = buf_arr[-15:, I_SPINE_HORIZ]
        v_peak   = float(np.max(v15))
        knee_end = float(np.mean(knee15[-4:]))
        knee_start = float(np.mean(knee15[:4]))
        spine_end  = float(np.mean(spine15[-4:]))
        knee_bending  = (knee_start - knee_end) > 0.10
        spine_upright = spine_end > 0.55
        fast_drop     = v_peak > 0.18
        if fast_drop and knee_bending and spine_upright:
            gate_scores["G7"] = 0.78

    # G8 — Stretching / exercise (duỗi người, yoga, exercise)
    # Đặc điểm: spine_horiz thấp nhưng có knee_extension_rate cao
    # + arm_flail cao (cánh tay duỗi ra khi tập)
    # + KHÔNG có spike velocity (không ngã đột ngột)
    if n_buf >= 20:
        v20_g8   = buf_arr[-20:, I_V_HIP_Y]
        max_v_g8 = float(np.max(np.abs(v20_g8)))
        kn_ext_g8 = float(kn_ext_rt) if len(cur) > I_KN_EXT_RT else 0.0
        arm_g8    = float(arm_flail)  if len(cur) > I_ARM_FLAIL  else 0.0
        if (max_v_g8 < 0.15          # không có spike velocity
                and spine_horiz < 0.45  # spine tương đối ngang
                and kn_ext_g8 > 0.20    # đang duỗi gối (tập thể dục)
                and arm_g8 > 0.30):     # tay vung ra (duỗi người)
            gate_scores["G8"] = 0.65

    # G9 — Crawling / all-fours (bò trên tất cả 4 chi)
    # Đặc điểm: cả vai lẫn hông đều thấp + spine ngang + nhưng knee gập mạnh
    if n_buf >= 15:
        sh_hip_15 = buf_arr[-15:, I_SH_HIP_DY] if I_SH_HIP_DY < buf_arr.shape[1] else None
        if sh_hip_15 is not None:
            mean_sh_hip = float(np.mean(sh_hip_15))
            mean_spine_15 = float(np.mean(buf_arr[-15:, I_SPINE_HORIZ]))
            mean_knee_15  = float(np.mean(buf_arr[-15:, I_KNEE_AVG]))
            # Khi bò: sh_hip_dy thấp (vai-hông gần nhau theo chiều đứng),
            # spine ngang, knee gập (knee_avg thấp)
            is_crawling = (
                mean_sh_hip < 0.20
                and mean_spine_15 < 0.40
                and mean_knee_15  < 0.55
            )
            if is_crawling:
                gate_scores["G9"] = 0.68

    # G10 — Dance / aerobics (nhảy, aerobic)
    # Đặc điểm: pose_energy rất cao NHƯNG body_sway thấp (nhảy có kiểm soát)
    # + không có prolonged descent (không nằm xuống)
    # + asymmetry cao xen kẽ (chuyển động nhịp nhàng)
    if n_buf >= 20:
        energy_20 = buf_arr[-20:, I_POSE_ENERGY] if I_POSE_ENERGY < buf_arr.shape[1] else None
        if energy_20 is not None:
            mean_energy = float(np.mean(energy_20))
            sway_g10    = float(body_sway) if len(cur) > I_BODY_SWAY else 0.0
            v20_g10     = buf_arr[-20:, I_V_HIP_Y]
            max_v_g10   = float(np.max(np.abs(v20_g10)))
            asym_std_g10 = float(np.std(buf_arr[-20:, I_ASYMMETRY])) \
                           if I_ASYMMETRY < buf_arr.shape[1] else 0.0
            is_dancing = (
                mean_energy > 0.30      # nhiều chuyển động
                and sway_g10 < 0.25     # không có sway đột ngột
                and max_v_g10 < 0.28    # không có velocity spike lớn
                and asym_std_g10 > 0.05 # asymmetry biến đổi nhịp nhàng (không cố định)
            )
            if is_dancing:
                gate_scores["G10"] = 0.60

    # Tổng hợp gate: lấy max của các gate có trọng số
    gate = max(gate_scores.values()) if gate_scores else 0.0

    # Nếu gate rất cao → tra về ngay
    if gate >= 0.85:
        return 0.0

    # ══════════════════════════════════════════════════════════════════
    #  TẦNG 2: POSITIVE RULES (14 rules)
    # ══════════════════════════════════════════════════════════════════
    scores = []

    # R1 — Horizontal body (ngã nằm ngang rõ ràng)
    if bbox_ar > 1.2 and spine_horiz < 0.35:
        s = (bbox_ar - 1.2) * 2.0 + (0.35 - spine_horiz) * 2.0
        scores.append(("R1", min(1.0, s)))

    # R2 — Rapid drop + velocity spike
    sum_v_hip = float(np.sum(v_hip_r))
    max_v_hip = float(np.max(v_hip_r))
    if sum_v_hip > 0.30 and max_v_hip > 0.15:
        scores.append(("R2", min(1.0, sum_v_hip * 3.0)))

    # R3 — Lying + straight knees + shoulder tilt
    if spine_horiz < 0.30 and knee_avg > 0.80 and sh_tilt > 0.03:
        scores.append(("R3", min(1.0, 0.65 + (0.30 - spine_horiz) * 1.5)))

    # R4 — Head at hip level
    if head_hip_dy < 0.15:
        scores.append(("R4", min(1.0, (0.15 - head_hip_dy) * 10.0)))

    # R5 — Head free-fall
    if v_nose > 0.05 and a_nose > 0.0:
        scores.append(("R5", min(1.0, v_nose * 8.0)))

    # R6 — 3D forward/backward fall
    if spine_dtilt > 0.35 or trunk_a3d > 0.45:
        s6 = max(
            min(1.0, (spine_dtilt - 0.35) * 3.0) if spine_dtilt > 0.35 else 0.0,
            min(1.0, (trunk_a3d  - 0.45) * 3.0) if trunk_a3d  > 0.45 else 0.0,
        )
        if s6 > 0.1:
            scores.append(("R6", s6))

    # R7 — 3D lying flat
    if spine_a3d > 0.50:
        scores.append(("R7", min(1.0, (spine_a3d - 0.50) * 4.0)))

    # R8 — 3D hip Z velocity
    if abs(v_hip_z) > 0.30:
        scores.append(("R8", min(1.0, abs(v_hip_z) * 1.5)))

    # R9 — Asymmetric impact (nghiêng + jerk khi chạm đất)
    tilt_avg = (sh_tilt + hip_tilt) / 2.0
    if tilt_avg > 0.055 and abs(j_hip) > 0.18:
        scores.append(("R9", min(1.0, tilt_avg * 4.5 + abs(j_hip) * 1.5)))

    # R10 — Slow lean fall (ngã dựa tường, ngã chậm có điểm tựa)
    if n_buf >= 20:
        spine20   = buf_arr[-20:, I_SPINE_HORIZ]
        sh_tilt20 = buf_arr[-20:, I_SH_TILT]
        knee20    = buf_arr[-20:, I_KNEE_AVG]
        mean_spine20 = float(np.mean(spine20))
        mean_tilt20  = float(np.mean(sh_tilt20))
        mean_knee20  = float(np.mean(knee20))
        lean_fall = (
            0.25 < mean_spine20 < 0.65
            and mean_tilt20  > 0.07
            and mean_knee20  > 0.70
            and spine_a3d    > 0.38
            and head_hip_dy  < 0.30
        )
        if lean_fall:
            s10 = min(1.0, mean_tilt20 * 5.0 + (0.65 - mean_spine20) * 1.5)
            scores.append(("R10", s10))

    # R11 — Stumble / trip (vấp ngã nhanh)
    # Đặc điểm: impact_score cao + hip_accel_mag cao + xảy ra trong < 10 frame
    if impact > 0.30 and hip_accel > 0.25:
        s11 = min(1.0, (impact + hip_accel) * 1.5)
        scores.append(("R11", s11))

    # R12 — Slow lean elderly (ngã từ từ của người già)
    # Đặc điểm: spine_horiz tăng dần rất chậm, descent_ratio trung bình,
    # NHƯNG head_hip_dy cuối cùng rất thấp
    if n_buf >= 40:
        spine40 = buf_arr[-40:, I_SPINE_HORIZ]
        head40  = buf_arr[-40:, I_HEAD_HIP_DY] if I_HEAD_HIP_DY < buf_arr.shape[1] else None
        if head40 is not None:
            spine_early_40 = float(np.mean(spine40[:10]))
            spine_late_40  = float(np.mean(spine40[-10:]))
            head_final     = float(np.mean(head40[-5:]))
            slow_lean = (
                spine_late_40 > spine_early_40 + 0.20  # spine dần nằm ngang
                and head_final < 0.25                   # đầu đã xuống gần hông
                and spine_late_40 > 0.35                # đang nằm ngang đáng kể
            )
            if slow_lean:
                s12 = min(1.0, (spine_late_40 - spine_early_40) * 2.5)
                scores.append(("R12", s12))

    # R13 — Lateral side-fall (ngã sang một bên)
    # Đặc điểm: lateral_tilt_rate cao + asymmetry cao + spine_angle_3d cao
    if lat_tilt_rt > 0.20 and asymmetry > 0.25 and spine_a3d > 0.30:
        s13 = min(1.0, lat_tilt_rt * 2 + asymmetry * 1.5)
        scores.append(("R13", s13))

    # R14 — Forward momentum fall (ngã về phía trước do mất đà)
    # Đặc điểm: head_accel_mag cao + trunk_angle_3d cao
    # + velocity mũi xuống (nose đi về trước)
    if head_accel > 0.30 and trunk_a3d > 0.35 and v_nose > 0.03:
        s14 = min(1.0, head_accel * 1.5 + trunk_a3d)
        scores.append(("R14", s14))

    # ── Top-down camera detection ──────────────────────────────────────
    _td_bbox   = float(np.clip(1.0 - abs(bbox_ar - 1.0) * 2.0, 0, 1))
    _td_spine  = float(np.clip((spine_horiz - 0.45) / 0.40, 0, 1))
    _td_head   = float(np.clip(1.0 - head_hip_dy / 0.20, 0, 1))
    top_down_score = float(np.clip((_td_bbox * 0.35 + _td_spine * 0.40 + _td_head * 0.25), 0, 1))
    is_top_down    = top_down_score > 0.45

    # R11-topdown: camera nhìn từ trên với 3D signals
    if is_top_down:
        gate = gate * 0.6  # giảm gate khi top-down (nhiều FP hơn)
        if spine_a3d > 0.40 and (v_hip_z > 0.20 or spine_dtilt > 0.25):
            s_td = min(1.0, spine_a3d * 1.5 + spine_dtilt)
            scores.append(("Rtd", s_td))

    # ── Positive score ────────────────────────────────────────────────
    if not scores:
        return 0.0

    pos_score = sum(s for _, s in scores) / max(len(scores), 1)
    pos_score = min(1.0, pos_score * 1.2)  # boost nhẹ khi có nhiều rules

    # ── Near-camera suppression ───────────────────────────────────────
    near_cam_factor = float(np.clip((near_cam - 0.4) / 0.6, 0, 0.3))

    # ── Final: gate * (1 - near_cam) ──────────────────────────────────
    final = pos_score * (1.0 - gate) * (1.0 - near_cam_factor)
    return float(np.clip(final, 0, 1))


# ======================================================================
#  SMOTE-TEMPORAL: Tạo fall sequences tổng hợp
#
#  Thay vì SMOTE trên feature space, ta nội suy trực tiếp giữa 2
#  sequences trong không gian thời gian:
#    seq_syn = lam * seq_a + (1-lam) * seq_b
#  với lam ~ Beta(0.5, 0.5) thiên về 0 và 1 (gần với data thật)
#
#  Khác Mixup: SMOTE-Temporal chỉ mix fall+fall (cùng class),
#  không mix với not_fall → không tạo label mơ hồ
# ======================================================================
def _smote_temporal(X_fall, ratio=2.0, k_neighbors=5, seed=42):
    """
    Tạo fall sequences tổng hợp.
    X_fall: shape (N, T, F) — chỉ các sequences có label = fall
    ratio: tạo thêm ratio * N sequences (tổng = N + ratio*N)
    Trả về: X_synthetic shape (round(N*ratio), T, F)
    """
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(seed)
    N, T, F = X_fall.shape
    n_generate = int(N * ratio)

    if N < 2:
        warn("SMOTE-Temporal: không đủ fall samples (< 2), bỏ qua.")
        return np.empty((0, T, F), dtype=np.float32)

    # Flatten sequences để tính kNN trong feature space
    X_flat = X_fall.reshape(N, -1).astype(np.float32)

    # Tính kNN (dùng cosine distance → robust với scale)
    k = min(k_neighbors, N - 1)
    dist_mat = cdist(X_flat, X_flat, metric="cosine")
    np.fill_diagonal(dist_mat, np.inf)
    knn_idx  = np.argsort(dist_mat, axis=1)[:, :k]

    synthetics = []
    for _ in range(n_generate):
        # Chọn ngẫu nhiên 1 seed sample
        i = rng.integers(0, N)
        # Chọn ngẫu nhiên 1 trong k neighbors
        j = knn_idx[i, rng.integers(0, k)]

        # Nội suy: lam ~ Beta(0.5, 0.5) thiên về extremes
        lam = float(rng.beta(0.5, 0.5))

        seq_a = X_fall[i]  # (T, F)
        seq_b = X_fall[j]  # (T, F)
        seq_syn = lam * seq_a + (1.0 - lam) * seq_b

        # Thêm noise nhỏ để tránh exact duplication
        noise = rng.normal(0, 0.005, seq_syn.shape).astype(np.float32)
        seq_syn = np.clip(seq_syn + noise, -1.5, 3.0)

        synthetics.append(seq_syn)

    result = np.array(synthetics, dtype=np.float32)
    ok(f"SMOTE-Temporal: tạo {len(result)} fall sequences tổng hợp từ {N} sequences gốc")
    return result


# ======================================================================
#  AUGMENTATION v3  (20 variants)
# ======================================================================
def _augment(X, y, max_multiplier=5):
    rng = np.random.default_rng(42)
    n, T, F = X.shape

    def aug_noise(Xb, sigma=0.010):
        return np.clip(Xb + rng.normal(0, sigma, Xb.shape).astype(np.float32), -1.5, 3)

    def aug_timewarp(Xb, speed=1.0):
        n_s, T_, F_ = Xb.shape
        new_t = np.clip(int(T_ * speed), T_ // 2, T_ * 2)
        Xr = np.zeros((n_s, T_, F_), dtype=np.float32)
        for i in range(n_s):
            src = Xb[i]
            idx = np.linspace(0, T_ - 1, new_t)
            warped = np.array([np.interp(idx, np.arange(T_), src[:, f])
                               for f in range(F_)], dtype=np.float32).T
            if new_t >= T_:
                start = rng.integers(0, new_t - T_ + 1)
                Xr[i] = warped[start:start + T_]
            else:
                Xr[i, :new_t] = warped
                Xr[i, new_t:] = warped[-1]
        return Xr

    def aug_scale(Xb, factor=1.0):
        # Scale chỉ velocity/accel features, giữ nguyên geometry
        Xr = Xb.copy()
        vel_feats = [12, 13, 14, 15, 16, 22, 23, 25, 28, 31, 35]
        for fi in [f for f in vel_feats if f < F]:
            Xr[:, :, fi] = np.clip(Xr[:, :, fi] * factor, -1.5, 1.5)
        return Xr

    def aug_rotation(Xb, delta=0.0):
        Xr = Xb.copy()
        Xr[:, :, 1] = np.clip(Xr[:, :, 1] + delta, 0, 1)
        Xr[:, :, 0] = np.clip(Xr[:, :, 0] + delta * 0.5, 0, 1)
        return Xr

    def aug_zflip(Xb):
        Xr = Xb.copy()
        Xr[:, :, 19] = np.clip(1.0 - Xr[:, :, 19], 0, 1)
        Xr[:, :, 22] = -Xr[:, :, 22]
        Xr[:, :, 23] = -Xr[:, :, 23]
        return Xr

    def aug_nearcam(Xb):
        Xr = Xb.copy()
        scale = rng.uniform(1.1, 1.4, (len(Xb), 1)).astype(np.float32)
        for fi in [4, 6, 7, 8]:
            if fi < F:
                Xr[:, :, fi] = np.clip(Xr[:, :, fi] * scale, 0, 3)
        Xr[:, :, 24] = np.clip(Xr[:, :, 24] + rng.uniform(0.2, 0.4, (len(Xb), T)), 0, 1)
        return Xr

    def aug_lowlight(Xb):
        Xr = Xb.copy()
        vis_drop = rng.uniform(0.3, 0.6, (len(Xb), 1)).astype(np.float32)
        Xr[:, :, 17] = np.clip(Xr[:, :, 17] * vis_drop, 0, 1)
        Xr[:, :, 18] = np.clip(Xr[:, :, 18] * vis_drop * 0.9, 0, 1)
        return Xr

    def aug_topdown_cam(Xb):
        Xr = Xb.copy()
        Xr[:, :, 1]  = np.clip(Xr[:, :, 1]  + rng.uniform(0.15, 0.35, (len(Xb), T)), 0, 1)
        pull = rng.uniform(0.3, 0.6, (len(Xb), T)).astype(np.float32)
        Xr[:, :, 4]  = Xr[:, :, 4] * (1 - pull) + 1.0 * pull
        Xr[:, :, 6]  = np.clip(Xr[:, :, 6]  * rng.uniform(0.4, 0.70, (len(Xb), T)), 0, 1)
        Xr[:, :, 7]  = np.clip(Xr[:, :, 7]  * rng.uniform(0.4, 0.70, (len(Xb), T)), 0, 1)
        if 20 < F: Xr[:, :, 20] = np.clip(Xr[:, :, 20] + rng.uniform(0.10, 0.25, (len(Xb), T)), 0, 1)
        if 21 < F: Xr[:, :, 21] = np.clip(Xr[:, :, 21] + rng.uniform(0.05, 0.18, (len(Xb), T)), 0, 1)
        return Xr

    def aug_body_shape(Xb):
        Xr = Xb.copy()
        body_sc  = rng.uniform(0.70, 1.35, (len(Xb), 1)).astype(np.float32)
        width_sc = rng.uniform(0.80, 1.40, (len(Xb), 1)).astype(np.float32)
        for fi in [6, 7, 8]:
            if fi < F: Xr[:, :, fi] = np.clip(Xr[:, :, fi] * body_sc, 0, 1)
        for fi in [9, 10, 11]:
            if fi < F: Xr[:, :, fi] = np.clip(Xr[:, :, fi] * rng.uniform(0.85, 1.15, (len(Xb), 1)), 0, 1)
        if 4 < F: Xr[:, :, 4] = np.clip(Xr[:, :, 4] * width_sc, 0, 3)
        for fi in [2, 3]:
            if fi < F: Xr[:, :, fi] = np.clip(Xr[:, :, fi] * rng.uniform(0.7, 1.5, (len(Xb), 1)), 0, 1)
        return Xr

    def aug_oblique_cam(Xb):
        Xr = Xb.copy()
        tilt_bias  = rng.uniform(-0.12, 0.12, (len(Xb), 1)).astype(np.float32)
        spine_bias = rng.uniform(-0.12, 0.20, (len(Xb), 1)).astype(np.float32)
        for fi in [2, 3]:
            if fi < F: Xr[:, :, fi] = np.clip(Xr[:, :, fi] + tilt_bias, 0, 1)
        if 1 < F:  Xr[:, :, 1]  = np.clip(Xr[:, :, 1]  + spine_bias, 0, 1)
        if 19 < F: Xr[:, :, 19] = np.clip(Xr[:, :, 19] + rng.uniform(0.05, 0.18, (len(Xb), 1)), 0, 1)
        return Xr

    # ── 6 NEW augmentation variants v8 ──────────────────────────────

    def aug_fisheye(Xb):
        """Mô phỏng lens fisheye (CCTV góc rộng): bbox_ar bị méo,
        tọa độ biên (tay, chân) bị kéo giãn ra ngoài."""
        Xr = Xb.copy()
        fisheye_scale = rng.uniform(1.1, 1.5, (len(Xb), 1)).astype(np.float32)
        # bbox rộng hơn so với thực
        if 4 < F: Xr[:, :, 4]  = np.clip(Xr[:, :, 4]  * fisheye_scale, 0, 4)
        # head_hip_dy giảm (chân và đầu bị kéo xa ra)
        if 6 < F: Xr[:, :, 6]  = np.clip(Xr[:, :, 6]  * rng.uniform(0.7, 0.9, (len(Xb), 1)), 0, 1)
        # arm_flail tăng (tay trông vung hơn)
        if 34 < F: Xr[:, :, 34] = np.clip(Xr[:, :, 34] * fisheye_scale * 0.8, 0, 1)
        # near_camera_score tăng (fisheye làm người trông gần hơn)
        if 24 < F: Xr[:, :, 24] = np.clip(Xr[:, :, 24] + 0.15, 0, 1)
        return Xr

    def aug_rain_fog(Xb):
        """Mô phỏng mưa / sương mù: visibility giảm mạnh, pose confidence thấp."""
        Xr = Xb.copy()
        # Drop visibility đột ngột (mưa to)
        vis_drop = rng.uniform(0.2, 0.55, (len(Xb), 1)).astype(np.float32)
        if 17 < F: Xr[:, :, 17] = np.clip(Xr[:, :, 17] * vis_drop, 0, 1)
        if 18 < F: Xr[:, :, 18] = np.clip(Xr[:, :, 18] * vis_drop * 0.8, 0, 1)
        # Thêm noise lớn hơn vào landmark positions (keypoint jitter do sương)
        noise_scale = rng.uniform(0.015, 0.030, (len(Xb), T)).astype(np.float32)
        geom_feats = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11]
        for fi in [f for f in geom_feats if f < F]:
            Xr[:, :, fi] = np.clip(Xr[:, :, fi] + rng.normal(0, 1, (len(Xb), T)).astype(np.float32) * noise_scale, 0, 1)
        return Xr

    def aug_crowd_occlusion(Xb):
        """Mô phỏng bị che khuất một phần bởi đám đông.
        Random mask một số features (set = 0 hoặc neutral value) trong các window ngắn."""
        Xr = Xb.copy()
        for i in range(len(Xb)):
            # Chọn ngẫu nhiên 1-2 window bị occlusion (mỗi window 5-15 frame)
            n_occ = rng.integers(1, 3)
            for _ in range(n_occ):
                occ_start = rng.integers(0, max(1, T - 15))
                occ_len   = rng.integers(5, min(15, T - occ_start))
                occ_end   = occ_start + occ_len
                # Giảm visibility trong window này
                if 17 < F: Xr[i, occ_start:occ_end, 17] *= 0.3
                if 18 < F: Xr[i, occ_start:occ_end, 18] *= 0.2
                # Set một số geometry features về neutral
                if 4 < F:  Xr[i, occ_start:occ_end, 4]  *= 0.5
        return Xr

    def aug_elderly_gait(Xb):
        """Mô phỏng dáng đi/chuyển động của người già:
        - Chuyển động chậm hơn (timewarp nhỏ)
        - Spine_horiz cao hơn (còng lưng)
        - Knee_avg thấp hơn (gối gập khi đi)
        - Body_sway cao hơn (dao động ngang nhiều hơn)"""
        Xr = Xb.copy()
        # Chậm hơn: timewarp xuống 0.7-0.85
        speed_factor = rng.uniform(0.70, 0.85)
        Xr = aug_timewarp(Xr, speed=speed_factor)
        # Còng lưng: spine_angle cao hơn
        if 0 < F: Xr[:, :, 0]  = np.clip(Xr[:, :, 0]  + rng.uniform(0.05, 0.15, (len(Xr), T)), 0, 1)
        # Gối gập nhẹ khi đứng
        if 11 < F: Xr[:, :, 11] = np.clip(Xr[:, :, 11] - rng.uniform(0.05, 0.12, (len(Xr), T)), 0, 1)
        # Sway cao hơn
        if 36 < F: Xr[:, :, 36] = np.clip(Xr[:, :, 36] + rng.uniform(0.05, 0.15, (len(Xr), T)), 0, 1)
        return Xr

    def aug_fast_action(Xb):
        """Mô phỏng chuyển động nhanh (chạy, nhảy):
        - Timewarp lên 1.3-1.5x
        - Velocity spike cao hơn
        - Arm_flail cao hơn"""
        Xr = Xb.copy()
        speed_factor = rng.uniform(1.30, 1.50)
        Xr = aug_timewarp(Xr, speed=speed_factor)
        if 12 < F: Xr[:, :, 12] = np.clip(Xr[:, :, 12] * rng.uniform(1.2, 1.5, (len(Xr), T)), -1, 1)
        if 34 < F: Xr[:, :, 34] = np.clip(Xr[:, :, 34] + rng.uniform(0.1, 0.3, (len(Xr), T)), 0, 1)
        return Xr

    def aug_lateral_cam(Xb):
        """Mô phỏng camera nhìn từ một bên (không phải trực diện):
        - sh_tilt và hip_tilt có bias dương đáng kể (một bên luôn cao hơn)
        - spine_depth_tilt cao hơn (depth rõ hơn)
        - Asymmetry cao hơn"""
        Xr = Xb.copy()
        bias = rng.uniform(0.08, 0.20, (len(Xb), 1)).astype(np.float32)
        for fi in [2, 3]:
            if fi < F: Xr[:, :, fi] = np.clip(Xr[:, :, fi] + bias, 0, 1)
        if 19 < F: Xr[:, :, 19] = np.clip(Xr[:, :, 19] + rng.uniform(0.10, 0.25, (len(Xb), 1)), 0, 1)
        if 29 < F: Xr[:, :, 29] = np.clip(Xr[:, :, 29] + rng.uniform(0.05, 0.15, (len(Xb), 1)), 0, 1)
        return Xr

    def aug_slowdescent(Xb, yb):
        """Chỉ cho not_fall: mô phỏng nằm từ từ."""
        mask = (yb == 0)
        if mask.sum() == 0:
            return None, None
        Xr = Xb[mask].copy()
        if 27 < F: Xr[:, :, 27] = np.clip(Xr[:, :, 27] + rng.uniform(0.2, 0.5, (mask.sum(), T)), 0, 1)
        if 25 < F: Xr[:, :, 25] = np.clip(Xr[:, :, 25] * 0.5, 0, 1)
        return Xr, yb[mask]

    # ── Danh sách tất cả 20 augmentation ─────────────────────────────
    all_augs = [
        # v7 originals (14)
        ("zflip",        lambda Xb, _: aug_zflip(Xb)),
        ("noise_s",      lambda Xb, _: aug_noise(Xb, 0.008)),
        ("noise_l",      lambda Xb, _: aug_noise(Xb, 0.015)),
        ("timewarp_s",   lambda Xb, _: aug_timewarp(Xb, 0.75)),
        ("timewarp_f",   lambda Xb, _: aug_timewarp(Xb, 1.30)),
        ("scale_d",      lambda Xb, _: aug_scale(Xb, 0.85)),
        ("scale_u",      lambda Xb, _: aug_scale(Xb, 1.15)),
        ("rot_l",        lambda Xb, _: aug_rotation(Xb, -0.08)),
        ("rot_r",        lambda Xb, _: aug_rotation(Xb,  0.08)),
        ("nearcam",      lambda Xb, _: aug_nearcam(Xb)),
        ("lowlight",     lambda Xb, _: aug_lowlight(Xb)),
        ("topdown_cam",  lambda Xb, _: aug_topdown_cam(Xb)),
        ("body_shape",   lambda Xb, _: aug_body_shape(Xb)),
        ("oblique_cam",  lambda Xb, _: aug_oblique_cam(Xb)),
        # v8 new (6)
        ("fisheye",      lambda Xb, _: aug_fisheye(Xb)),
        ("rain_fog",     lambda Xb, _: aug_rain_fog(Xb)),
        ("crowd_occ",    lambda Xb, _: aug_crowd_occlusion(Xb)),
        ("elderly_gait", lambda Xb, _: aug_elderly_gait(Xb)),
        ("fast_action",  lambda Xb, _: aug_fast_action(Xb)),
        ("lateral_cam",  lambda Xb, _: aug_lateral_cam(Xb)),
    ]

    n_extra = max_multiplier - 1
    n_extra = min(n_extra, len(all_augs))

    if n < 10000:
        chosen_augs = all_augs
    else:
        chosen_idx  = rng.choice(len(all_augs), size=n_extra, replace=False)
        chosen_augs = [all_augs[i] for i in sorted(chosen_idx)]
        print(f"    Aug variants ({n_extra}/{len(all_augs)}): "
              f"{', '.join(a[0] for a in chosen_augs)}")

    parts = [X]
    lbls  = [y]
    for aug_name, aug_fn in chosen_augs:
        parts.append(aug_fn(X, y))
        lbls.append(y)

    # Slow descent cho not_fall
    Xsd, ysd = aug_slowdescent(X, y)
    if Xsd is not None:
        parts.append(Xsd); lbls.append(ysd)

    Xa  = np.concatenate(parts)
    ya  = np.concatenate(lbls)
    idx = rng.permutation(len(Xa))
    return Xa[idx], ya[idx]


# ======================================================================
#  LABEL SMOOTHING & MIXUP
# ======================================================================
def _apply_label_smoothing(Y, smoothing=0.1):
    n_classes = Y.shape[-1]
    return Y * (1.0 - smoothing) + (smoothing / n_classes)

def _mixup_batch(X, Y, alpha=0.15, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(X)
    lam = rng.beta(alpha, alpha, size=n).astype(np.float32)
    lam = np.maximum(lam, 1.0 - lam)  # lam >= 0.5 để giữ label chính
    idx = rng.permutation(n)
    lam_r = lam[:, None, None]
    X_mix = lam_r * X + (1.0 - lam_r) * X[idx]
    lam_y = lam[:, None]
    Y_int = Y if Y.ndim == 2 else None
    if Y_int is not None:
        Y_mix = lam_y * Y + (1.0 - lam_y) * Y[idx]
    else:
        Y_mix = Y
    return X_mix.astype(np.float32), Y_mix


# ======================================================================
#  FOCAL LOSS
# ======================================================================
def _focal_loss(alpha=0.60, gamma=2.5):
    try:
        import tensorflow as tf

        def focal_loss(y_true, y_pred):
            y_pred    = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            ce_loss   = -y_true * tf.math.log(y_pred)
            p_t       = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
            alpha_t   = tf.where(y_true[:, 1:2] > 0.5,
                                 tf.ones_like(p_t) * alpha,
                                 tf.ones_like(p_t) * (1 - alpha))
            focal_w   = alpha_t * tf.pow(1.0 - p_t, gamma)
            loss      = tf.reduce_sum(focal_w * ce_loss, axis=-1)
            return tf.reduce_mean(loss)

        return focal_loss
    except ImportError:
        return "categorical_crossentropy"


# ======================================================================
#  MODEL: Multi-Scale TCN + CBAM Attention + Auxiliary Loss
# ======================================================================
def build_ms_tcn_model(n_features, seq_len, config=None):
    """
    Multi-Scale Temporal Convolutional Network với:
    1. Multi-scale convolutions: kernel 3, 5, 7 chạy song song
       → bắt được patterns ngắn hạn (spike) và dài hạn (slow lean)
    2. CBAM (Convolutional Block Attention Module):
       - Channel attention: tự động trọng số hóa features quan trọng
       - Temporal attention: focus vào frames quan trọng (điểm ngã)
    3. Auxiliary output: dự đoán "frame nào là điểm ngã"
       → giúp model học temporal localization tốt hơn
    4. Dilation tăng lên 1,2,4,8 → receptive field lớn hơn

    Architecture:
      Input → LayerNorm
      → [MS-TCN Block x4 (dilation 1,2,4,8)]
         Mỗi block: parallel Conv(k=3,5,7) → concat → CBAM → residual
      → GlobalAvgPool + GlobalMaxPool (concat)
      → Dense(64) → Dense(32) → [main_out(2), aux_out(seq_len)]
    """
    cfg = config or CONFIG
    try:
        import tensorflow as tf
        import keras
        from keras import layers, Model, regularizers
    except ImportError:
        err("pip install tensorflow"); sys.exit(1)

    filters     = cfg.get("tcn_filters",   96)
    kernels     = cfg.get("tcn_kernels",   [3, 5, 7])
    dilations   = cfg.get("tcn_dilations", [1, 2, 4, 8])
    dropout     = cfg.get("tcn_dropout",   0.30)
    cbam_red    = cfg.get("cbam_reduction",8)
    aux_weight  = cfg.get("aux_loss_weight", 0.20)

    def channel_attention(x, reduction=8, name=""):
        """CBAM channel attention."""
        C = x.shape[-1]
        # Average-pool và Max-pool theo time axis
        avg_pool = keras.ops.mean(x, axis=1, keepdims=True)  # (B,1,C)
        max_pool = keras.ops.max(x,  axis=1, keepdims=True)  # (B,1,C)

        # Shared MLP
        dense1 = layers.Dense(max(C // reduction, 1), activation="relu",
                               name=f"{name}_ca_d1", use_bias=False)
        dense2 = layers.Dense(C, name=f"{name}_ca_d2", use_bias=False)

        avg_out = dense2(dense1(avg_pool))
        max_out = dense2(dense1(max_pool))
        scale   = keras.ops.sigmoid(avg_out + max_out)   # (B,1,C)
        return x * scale

    def temporal_attention(x, name=""):
        """CBAM spatial (temporal) attention."""
        avg_pool = keras.ops.mean(x, axis=-1, keepdims=True)   # (B,T,1)
        max_pool = keras.ops.max(x,  axis=-1, keepdims=True)   # (B,T,1)
        concat   = keras.ops.concatenate([avg_pool, max_pool], axis=-1)  # (B,T,2)
        attn     = layers.Conv1D(1, kernel_size=7, padding="same",
                                 activation="sigmoid",
                                 name=f"{name}_ta_conv")(concat) # (B,T,1)
        return x * attn

    def ms_tcn_block(x, filters, dilations_list, dropout_rate, name=""):
        """
        Multi-Scale TCN block: chạy 3 kernel sizes song song,
        concat, CBAM attention, residual connection.
        """
        branches = []
        for k in kernels:
            h = layers.Conv1D(filters // len(kernels), k,
                              padding="causal",
                              dilation_rate=dilations_list,
                              kernel_regularizer=regularizers.l2(1e-4),
                              name=f"{name}_k{k}")(x)
            h = layers.BatchNormalization(name=f"{name}_bn_k{k}")(h)
            h = layers.Activation("relu")(h)
            branches.append(h)

        # Concat tất cả kernel branches
        if len(branches) > 1:
            h = layers.Concatenate(name=f"{name}_concat")(branches)
        else:
            h = branches[0]

        # Điều chỉnh về filters nếu cần (sau concat số channels = filters)
        if h.shape[-1] != filters:
            h = layers.Conv1D(filters, 1, name=f"{name}_align")(h)

        h = layers.SpatialDropout1D(dropout_rate)(h)

        # CBAM attention
        h = channel_attention(h, reduction=cbam_red, name=name)
        h = temporal_attention(h, name=name)

        # Residual: project input nếu cần
        if x.shape[-1] != filters:
            x = layers.Conv1D(filters, 1, name=f"{name}_res")(x)

        return layers.Add()([x, h])

    # ── Input ────────────────────────────────────────────────────────
    inp = layers.Input(shape=(seq_len, n_features), name="sequence")
    x   = layers.LayerNormalization()(inp)

    # ── 4 MS-TCN blocks với dilation 1,2,4,8 ──────────────────────────
    for i, d in enumerate(dilations):
        x = ms_tcn_block(x, filters, d, dropout, name=f"mstcn{i+1}")

    # ── Pooling: concat avg + max để capture cả mean và peak ─────────
    gap = layers.GlobalAveragePooling1D()(x)
    gmp = layers.GlobalMaxPooling1D()(x)
    x_pool = layers.Concatenate()([gap, gmp])   # (B, filters*2)

    # ── Main classification head ──────────────────────────────────────
    x_cls = layers.Dense(64, activation="relu",
                         kernel_regularizer=regularizers.l2(1e-4))(x_pool)
    x_cls = layers.Dropout(0.3)(x_cls)
    x_cls = layers.Dense(32, activation="relu",
                         kernel_regularizer=regularizers.l2(1e-4))(x_cls)
    x_cls = layers.Dropout(0.2)(x_cls)
    main_out = layers.Dense(2, activation="softmax", name="output")(x_cls)

    # ── Auxiliary head: predict điểm ngã (per-frame binary) ───────────
    # Chỉ dùng trong training để guide temporal localization
    x_aux    = layers.Conv1D(32, 3, padding="same", activation="relu",
                             name="aux_conv")(x)
    x_aux    = layers.Dropout(0.3)(x_aux)
    aux_out  = layers.Conv1D(1, 1, activation="sigmoid",
                             name="aux_output")(x_aux)  # (B, T, 1)
    aux_out  = layers.Reshape((seq_len,), name="fall_moment")(aux_out)

    model = Model(inp, [main_out, aux_out],
                  name="FallDetector_MSTCN_CBAM")
    return model


def build_tcn_model(n_features, seq_len):
    """TCN đơn giản từ v7 — fallback."""
    try:
        import tensorflow as tf
        import keras
        from keras import layers, Model, regularizers
    except ImportError:
        err("pip install tensorflow"); sys.exit(1)

    def tcn_block(x, filters, kernel_size, dilation, dropout=0.3, name=""):
        h = layers.Conv1D(filters, kernel_size, padding="causal",
                          dilation_rate=dilation,
                          kernel_regularizer=regularizers.l2(1e-4),
                          name=f"{name}_conv1")(x)
        h = layers.BatchNormalization(name=f"{name}_bn1")(h)
        h = layers.Activation("relu")(h)
        h = layers.SpatialDropout1D(dropout)(h)
        h = layers.Conv1D(filters, kernel_size, padding="causal",
                          dilation_rate=dilation,
                          kernel_regularizer=regularizers.l2(1e-4),
                          name=f"{name}_conv2")(h)
        h = layers.BatchNormalization(name=f"{name}_bn2")(h)
        h = layers.Activation("relu")(h)
        h = layers.SpatialDropout1D(dropout)(h)
        if x.shape[-1] != filters:
            x = layers.Conv1D(filters, 1, name=f"{name}_res")(x)
        return layers.Add()([x, h])

    inp = layers.Input(shape=(seq_len, n_features), name="sequence")
    x   = layers.LayerNormalization()(inp)
    for i, d in enumerate([1, 2, 4, 8]):
        x = tcn_block(x, 96, 3, d, name=f"tcn{i+1}")
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(32, activation="relu",
                       kernel_regularizer=regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(2, activation="softmax", name="output")(x)
    return Model(inp, out, name="FallDetector_TCN")


def build_bilstm_model(n_features, seq_len):
    """BiLSTM+Attention từ v7 — legacy fallback."""
    try:
        import tensorflow as tf
        import keras
        from keras import layers, Model, regularizers
        from keras.utils import register_keras_serializable

        @register_keras_serializable(package="FallDet")
        class ReduceSumLayer(layers.Layer):
            def call(self, x):    return tf.reduce_sum(x, axis=1)
            def get_config(self): return super().get_config()
    except ImportError:
        err("pip install tensorflow"); sys.exit(1)

    inp = layers.Input(shape=(seq_len, n_features), name="sequence")
    x   = layers.LayerNormalization()(inp)
    x   = layers.Bidirectional(layers.LSTM(64, return_sequences=True,
                kernel_regularizer=regularizers.l2(1e-4)), name="bilstm1")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.35)(x)
    x   = layers.Bidirectional(layers.LSTM(32, return_sequences=True,
                kernel_regularizer=regularizers.l2(1e-4)), name="bilstm2")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.35)(x)
    attn = layers.Dense(1, activation="tanh")(x)
    attn = layers.Flatten()(attn)
    attn = layers.Activation("softmax", name="attention")(attn)
    lstm2_dim = 32 * 2
    attn = layers.RepeatVector(lstm2_dim)(attn)
    attn = layers.Permute([2, 1])(attn)
    context = ReduceSumLayer()(layers.Multiply()([x, attn]))
    x   = layers.Dense(32, activation="relu",
                       kernel_regularizer=regularizers.l2(1e-4))(context)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(2, activation="softmax", name="output")(x)
    return Model(inp, out, name="FallDetector_BiLSTM_Attention")


def _select_model_builder():
    arch = CONFIG.get("model_arch", "ms_tcn")
    if arch == "ms_tcn":
        return build_ms_tcn_model
    elif arch == "tcn":
        return build_tcn_model
    else:
        return build_bilstm_model


# ======================================================================
#  AUXILIARY LABEL GENERATOR
#  Tạo per-frame fall label từ sequence-level label
#  Logic: nếu sequence là fall, tạo window nhỏ ở cuối sequence
#  (khoảng ngã thường xảy ra ở 2/3 cuối sequence)
# ======================================================================
def _generate_aux_labels(y, seq_len, fall_window=10):
    """
    y: (N,) sequence labels
    Trả về: (N, seq_len) per-frame labels
    """
    N  = len(y)
    ya = np.zeros((N, seq_len), dtype=np.float32)

    for i in range(N):
        if y[i] == 1:  # fall sequence
            # Fall moment ước lượng ở 2/3 cuối sequence
            fall_center = int(seq_len * 0.70)
            start = max(0, fall_center - fall_window // 2)
            end   = min(seq_len, fall_center + fall_window // 2)
            ya[i, start:end] = 1.0
    return ya


# ======================================================================
#  CALLBACKS & TRAINING UTILS
# ======================================================================
def _load_keras_model(path):
    import tensorflow as tf
    # Focal loss phải được đăng ký để load model (Keras 3 yêu cầu)
    focal = _focal_loss(CONFIG.get("focal_alpha", 0.60),
                        CONFIG.get("focal_gamma", 2.5))
    custom_objects = {"focal_loss": focal}
    try:
        import tensorflow as tf
        import keras
        from keras.utils import get_custom_objects
        existing = get_custom_objects()
        if "ReduceSumLayer" in existing:
            custom_objects["ReduceSumLayer"] = existing["ReduceSumLayer"]
    except Exception:
        pass
    return tf.keras.models.load_model(
        path,
        custom_objects=custom_objects,
        compile=False)

def _make_callbacks(model_path, config, total_epochs=30, lr=0.001, phase="main"):
    try:
        import tensorflow as tf
        from keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                                ReduceLROnPlateau, LambdaCallback)
    except ImportError:
        err("pip install tensorflow"); sys.exit(1)

    patience_map = {
        "main":        config.get("early_stop_patience", 10),
        "finetune":    config.get("finetune_real_patience", 6),
        "incremental": config.get("incremental_patience", 4),
    }
    patience = patience_map.get(phase, 10)

    class RecallMetricCallback(tf.keras.callbacks.Callback):
        def __init__(self, val_data):
            super().__init__()
            self.X_val, self.y_val = val_data

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            out = self.model(self.X_val, training=False)
            if isinstance(out, (list, tuple)):
                y_prob = np.array(out[0])
            else:
                y_prob = np.array(out)
            y_pred = np.argmax(y_prob, axis=1)
            y_true = self.y_val if self.y_val.ndim == 1 \
                     else np.argmax(self.y_val, axis=1)
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            recall    = tp / (tp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            f1 = 2 * recall * precision / (recall + precision + 1e-8)
            logs["val_recall"]    = recall
            logs["val_precision"] = precision
            logs["val_f1"]        = f1

    class CosineWarmRestartCallback(tf.keras.callbacks.Callback):
        """Cosine annealing với warm restart (SGDR)."""
        def __init__(self, lr_max, lr_min=1e-7, T0=10, T_mult=2, warmup_epochs=3):
            super().__init__()
            self.lr_max       = lr_max
            self.lr_min       = lr_min
            self.T0           = T0
            self.T_mult       = T_mult
            self.warmup_epochs= warmup_epochs
            self._restart     = 0
            self._t_cur       = 0
            self._T_cur       = T0

        def on_epoch_begin(self, epoch, logs=None):
            if epoch < self.warmup_epochs:
                new_lr = self.lr_max * (epoch + 1) / (self.warmup_epochs + 1)
            else:
                t = self._t_cur
                T = self._T_cur
                cosine = 0.5 * (1 + math.cos(math.pi * t / T))
                new_lr = self.lr_min + (self.lr_max - self.lr_min) * cosine
                self._t_cur += 1
                if self._t_cur >= self._T_cur:
                    self._t_cur  = 0
                    self._T_cur  = int(self._T_cur * self.T_mult)
                    self._restart += 1
            try:
                self.model.optimizer.learning_rate.assign(new_lr)
            except Exception:
                self.model.optimizer.lr = new_lr

    class CosineAnnealingCallback(tf.keras.callbacks.Callback):
        def __init__(self, lr_max, lr_min=1e-7, total_epochs=30):
            super().__init__()
            self.lr_max = lr_max; self.lr_min = lr_min
            self.total_epochs = total_epochs

        def on_epoch_begin(self, epoch, logs=None):
            cosine = 0.5 * (1 + math.cos(math.pi * epoch / self.total_epochs))
            new_lr = self.lr_min + (self.lr_max - self.lr_min) * cosine
            try:
                self.model.optimizer.learning_rate.assign(new_lr)
            except Exception:
                self.model.optimizer.lr = new_lr

    lr_schedule = config.get("lr_schedule", "cosine_warm")
    if lr_schedule == "cosine_warm":
        lr_cb = CosineWarmRestartCallback(
            lr_max=lr, lr_min=1e-7, T0=8, T_mult=2,
            warmup_epochs=config.get("lr_warmup_epochs", 3))
    elif lr_schedule == "cosine":
        lr_cb = CosineAnnealingCallback(lr_max=lr, lr_min=1e-7, total_epochs=total_epochs)
    else:
        lr_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                  patience=max(3, patience // 2),
                                  min_lr=1e-7, verbose=1)

    cbs = [
        EarlyStopping(monitor="val_recall", patience=patience,
                      restore_best_weights=True, verbose=1, mode="max"),
        ModelCheckpoint(model_path, monitor="val_recall",
                        save_best_only=True, verbose=1, mode="max"),
        lr_cb,
    ]
    return cbs, RecallMetricCallback


# ======================================================================
#  AUTO TUNE INFERENCE PARAMS
# ======================================================================
def _auto_tune_inference_params(model, X_val, y_val):
    """
    Tìm optimal threshold + ensemble weights theo F2-score.
    F2 ưu tiên Recall gấp đôi Precision (phù hợp fall detection).
    """
    step("Auto tuning inference params (F2-score)...")
    try:
        import tensorflow as tf
        out = model(X_val, training=False)
        if isinstance(out, (list, tuple)):
            probs = np.array(out[0])
        else:
            probs = np.array(out)
        fall_probs = probs[:, 1]
    except Exception as e:
        warn(f"Auto tune: predict failed: {e}"); return

    y_true = y_val if y_val.ndim == 1 else np.argmax(y_val, axis=1)

    # Grid search threshold
    best_f2 = 0; best_thresh = CONFIG["confidence_threshold"]
    for thresh in np.arange(0.30, 0.80, 0.02):
        preds = (fall_probs >= thresh).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f2 = (5 * p * r) / (4 * p + r + 1e-8)
        if f2 > best_f2:
            best_f2 = f2; best_thresh = float(thresh)

    CONFIG["confidence_threshold"] = best_thresh
    ok(f"Auto threshold: {best_thresh:.2f}  (F2={best_f2:.3f})")


# ======================================================================
#  DETECT FALL WINDOW (tìm vùng ngã trong video)
# ======================================================================
def _detect_fall_window(valid_feats, pre=90, post=60):
    n = len(valid_feats)
    if n == 0: return 0, n

    arr     = np.array(valid_feats, dtype=np.float32)
    v_hip   = arr[:, 12]
    spine   = arr[:, 1]
    d_spine = np.abs(np.diff(spine, prepend=spine[0]))

    # v8: thêm impact_score nếu có
    impact_col = 37
    if arr.shape[1] > impact_col:
        impact_sig = arr[:, impact_col]
    else:
        impact_sig = np.zeros(n)

    score   = np.clip(v_hip, 0, 1) + d_spine * 2.0 + impact_sig * 1.5
    smooth  = np.convolve(score, np.ones(5) / 5.0, mode='same')
    peak    = int(np.argmax(smooth))

    if float(smooth[peak]) < 0.08 * 10 + 0.06 * 2.0:
        return 0, n

    return max(0, peak - pre), min(n, peak + post)


# ======================================================================
#  BUOC 1: EXTRACT
# ======================================================================
def run_extract(include_hard_neg=True):
    banner("BUOC 1/4 - EXTRACT KEYPOINTS (v8)")
    seq_len = CONFIG["sequence_length"]
    os.makedirs(CONFIG["processed_dir"], exist_ok=True)

    hard_neg_dir = CONFIG["hard_neg_dir"]
    if include_hard_neg and os.path.isdir(hard_neg_dir):
        hard_vids = [f for f in os.listdir(hard_neg_dir)
                     if f.lower().endswith((".mp4",".avi",".mov",".mkv"))]
        if hard_vids:
            step(f"Tim thay {len(hard_vids)} hard negative clips")
            not_fall_dir = os.path.join(CONFIG["dataset_dir"], "not_fall")
            os.makedirs(not_fall_dir, exist_ok=True)
            for vf in hard_vids:
                import shutil
                src = os.path.join(hard_neg_dir, vf)
                dst = os.path.join(not_fall_dir, "hardneg_" + vf)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    for cls in CONFIG["classes"]:
        d = os.path.join(CONFIG["dataset_dir"], cls)
        if not os.path.isdir(d):
            err(f"Missing: {d}"); sys.exit(1)

    all_seqs, all_labels = [], []
    stride  = CONFIG.get("extract_stride", 5)
    mp_size = 480

    detector      = _make_pose_detector("VIDEO")
    _global_ts_ms = 1

    for cls_idx, cls_name in enumerate(CONFIG["classes"]):
        cls_dir = os.path.join(CONFIG["dataset_dir"], cls_name)
        videos  = sorted([f for f in os.listdir(cls_dir)
                          if f.lower().endswith((".mp4",".avi",".mov",".mkv",".3gp",".flv",".m4v"))])
        if not videos:
            warn(f"No videos in {cls_dir}"); continue
        step(f"[{cls_name.upper()}] {len(videos)} videos  (stride={stride})")

        for vi, vfile in enumerate(videos):
            cap = cv2.VideoCapture(os.path.join(cls_dir, vfile))
            if not cap.isOpened():
                warn(f"Cannot open: {vfile}"); continue

            fps              = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval_ms= max(1, int(1000.0 / fps))
            total_frames     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 999)

            if total_frames < 120:   eff_stride = 1
            elif total_frames < 300: eff_stride = 2
            else:                    eff_stride = stride

            buf, prev_state, n, none_streak, frame_idx = [], None, 0, 0, 0
            feat_history = []
            all_feats_video = []
            _vid_ts_ms = _global_ts_ms

            while True:
                ret, frame = cap.read()
                if not ret: break
                _ts_ms       = _vid_ts_ms
                _vid_ts_ms  += frame_interval_ms
                frame_idx   += 1

                need_pose = (
                    len(buf) < seq_len
                    or frame_idx % eff_stride == 0
                    or none_streak > 0
                )
                if not need_pose:
                    all_feats_video.append(None)
                    continue

                fh_orig, fw_orig = frame.shape[:2]
                if max(fh_orig, fw_orig) > mp_size:
                    scale = mp_size / max(fh_orig, fw_orig)
                    frame = cv2.resize(frame,
                                       (int(fw_orig * scale), int(fh_orig * scale)),
                                       interpolation=cv2.INTER_LINEAR)

                # Enhance frame trước khi pose detection
                frame_enh = enhance_frame(frame)

                lm_list = _run_pose_on_frame(detector, frame_enh,
                                              timestamp_ms=_ts_ms, use_video_mode=True)
                if lm_list is None:
                    none_streak += 1
                    if none_streak > 10:
                        buf.clear(); prev_state = None; none_streak = 0
                    all_feats_video.append(None)
                    continue
                none_streak = 0

                # Dùng người đầu tiên (index 0)
                lm_pair = lm_list[0]
                feats, prev_state = extract_features(lm_pair, prev_state, feat_history)
                feat_history.append(feats)
                if len(feat_history) > 30: feat_history.pop(0)
                buf.append(feats)
                all_feats_video.append(feats)

            cap.release()
            _global_ts_ms = _vid_ts_ms

            valid_feats = [f for f in all_feats_video if f is not None]

            if cls_name == "fall" and len(valid_feats) >= seq_len:
                win_start, win_end = _detect_fall_window(
                    valid_feats,
                    pre  = CONFIG.get("fall_window_pre",  90),
                    post = CONFIG.get("fall_window_post", 60),
                )
                use_feats = valid_feats[win_start:win_end]
                if len(use_feats) < seq_len:
                    use_feats = valid_feats
                win_note = f"  win={win_start}-{win_end}/{len(valid_feats)}"
            else:
                use_feats = valid_feats
                win_note  = ""

            seq_buf = []
            for fi, feats in enumerate(use_feats):
                seq_buf.append(feats)
                if len(seq_buf) >= seq_len and (fi % eff_stride == 0):
                    all_seqs.append(np.array(seq_buf[-seq_len:], dtype=np.float32))
                    all_labels.append(cls_idx)
                    n += 1

            print(f"    [{vi+1:02d}/{len(videos):02d}] {vfile:<30} +{n} seqs{win_note}")
            if n < 3:
                zero_log = os.path.join(CONFIG["processed_dir"], "zero_seq_videos.txt")
                with open(zero_log, "a", encoding="utf-8") as zf:
                    zf.write(f"{cls_name}/{vfile}  ({n} seqs)\n")
                if n > 0:
                    del all_seqs[-n:]; del all_labels[-n:]

    detector.close()

    if not all_seqs:
        err("No sequences!"); sys.exit(1)

    X = np.array(all_seqs, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    step(f"Total: {len(X)} sequences  (fall={np.sum(y==1)}, not_fall={np.sum(y==0)})")

    # ── SMOTE-Temporal: cân bằng data ────────────────────────────────
    if CONFIG.get("smote_temporal", True) and np.sum(y == 1) > 0:
        step("SMOTE-Temporal: tạo fall sequences tổng hợp...")
        X_fall = X[y == 1]
        try:
            X_synthetic = _smote_temporal(
                X_fall,
                ratio=CONFIG.get("smote_ratio", 2.0),
                k_neighbors=CONFIG.get("smote_k_neighbors", 5))
            if len(X_synthetic) > 0:
                y_synthetic = np.ones(len(X_synthetic), dtype=np.int32)
                X = np.concatenate([X, X_synthetic])
                y = np.concatenate([y, y_synthetic])
                ok(f"Sau SMOTE: {len(X)} sequences  (fall={np.sum(y==1)}, not_fall={np.sum(y==0)})")
        except Exception as e:
            warn(f"SMOTE-Temporal failed: {e} — tiếp tục không có SMOTE")

    # ── Split TRƯỚC augment (tránh data leak) ─────────────────────────
    from sklearn.model_selection import train_test_split as _tts
    y_int = y.astype(np.int32)
    X_tr_raw, X_tmp, y_tr_raw, y_tmp = _tts(
        X, y_int, test_size=0.30, random_state=42, stratify=y_int)
    X_val_raw, X_te_raw, y_val_raw, y_te_raw = _tts(
        X_tmp, y_tmp.astype(np.int32), test_size=0.50,
        random_state=42, stratify=y_tmp.astype(np.int32))

    step(f"Split (raw): Train={len(X_tr_raw)} Val={len(X_val_raw)} Test={len(X_te_raw)}")
    step("Augmentation (chỉ trên train set)...")
    X_tr_aug, y_tr_aug = _augment(X_tr_raw, y_tr_raw)
    ok(f"Raw train:{len(X_tr_raw)} → Aug train:{len(X_tr_aug)}")

    p = CONFIG["processed_dir"]
    np.save(f"{p}/sequences_aug.npy", X_tr_aug)
    np.save(f"{p}/labels_aug.npy",    y_tr_aug)
    np.save(f"{p}/X_val.npy",         X_val_raw)
    np.save(f"{p}/Y_val.npy",         y_val_raw)
    np.save(f"{p}/X_test.npy",        X_te_raw)
    np.save(f"{p}/Y_test.npy",        y_te_raw)

    # Lưu cả raw để tham khảo
    np.save(f"{p}/sequences.npy", X)
    np.save(f"{p}/labels.npy",    y)

    ok(f"Saved to {p}/")
    step(f"fall ratio cuoi: {np.sum(y_tr_aug==1)/len(y_tr_aug)*100:.1f}% "
         f"({np.sum(y_tr_aug==1)}/{len(y_tr_aug)})")
    return X_tr_aug, y_tr_aug


# ======================================================================
#  BUOC 2: TRAIN
# ======================================================================
def run_train(X=None, y=None):
    banner("BUOC 2/4 - TRAIN Multi-Scale TCN + CBAM")
    try:
        import tensorflow as tf
        import keras
        from keras.utils import to_categorical
        from keras.optimizers import Adam
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError as e:
        err(f"{e}\npip install tensorflow scikit-learn"); sys.exit(1)

    if X is None:
        p  = CONFIG["processed_dir"]
        sp = (f"{p}/sequences_aug.npy" if os.path.exists(f"{p}/sequences_aug.npy")
              else f"{p}/sequences.npy")
        lp = (f"{p}/labels_aug.npy" if os.path.exists(f"{p}/labels_aug.npy")
              else f"{p}/labels.npy")
        if not os.path.exists(sp):
            err("Not extracted."); sys.exit(1)
        X = np.load(sp); y = np.load(lp)

    seq_len    = X.shape[1]
    n_features = X.shape[2]
    step(f"Data:{len(X)} seq_len={seq_len} features={n_features}")
    step(f"fall={np.sum(y==1)} not_fall={np.sum(y==0)}")

    Y = to_categorical(y, 2)

    # ── Load val/test sạch ────────────────────────────────────────────
    p = CONFIG["processed_dir"]
    val_path  = f"{p}/X_val.npy"
    test_path = f"{p}/X_test.npy"

    if os.path.exists(val_path) and os.path.exists(test_path):
        X_val     = np.load(val_path)
        y_val_raw = np.load(f"{p}/Y_val.npy")
        X_te      = np.load(test_path)
        Y_te_raw  = np.load(f"{p}/Y_test.npy")
        Y_val     = to_categorical(y_val_raw, 2)
        Y_te      = to_categorical(Y_te_raw, 2)
        X_tr = X; Y_tr = Y
        step(f"Train (aug):{len(X_tr)}  Val (sach):{len(X_val)}  Test (sach):{len(X_te)}")
    else:
        warn("Không tìm thấy val/test sạch. Chạy --retrain để extract lại.")
        from sklearn.model_selection import train_test_split
        X_tr, X_tmp, Y_tr, Y_tmp = train_test_split(X, Y, test_size=0.30,
                                                     random_state=42, stratify=y)
        y_tmp = np.argmax(Y_tmp, 1)
        X_val, X_te, Y_val, Y_te = train_test_split(
            X_tmp, Y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)
        y_val_raw = y_tmp[:len(X_val)]

    # ── Label smoothing ───────────────────────────────────────────────
    smoothing = CONFIG.get("label_smoothing", 0.08)
    if smoothing > 0:
        Y_tr_smooth = _apply_label_smoothing(Y_tr, smoothing)
    else:
        Y_tr_smooth = Y_tr

    # ── Mixup ─────────────────────────────────────────────────────────
    mixup_alpha = CONFIG.get("mixup_alpha", 0.15)
    if mixup_alpha and mixup_alpha > 0:
        rng_mix = np.random.default_rng(123)
        X_tr, Y_tr_smooth = _mixup_batch(X_tr, Y_tr_smooth, alpha=mixup_alpha, rng=rng_mix)
        step(f"Mixup alpha={mixup_alpha}: {len(X_tr)} sequences")

    # ── Auxiliary labels ──────────────────────────────────────────────
    y_orig_int = np.argmax(Y_tr, 1) if Y_tr.ndim == 2 else Y_tr.astype(int)
    aux_window = CONFIG.get("aux_fall_window", 10)
    y_aux_tr   = _generate_aux_labels(y_orig_int, seq_len, aux_window)
    y_aux_val  = _generate_aux_labels(y_val_raw.astype(int), seq_len, aux_window)

    # ── Class weights ─────────────────────────────────────────────────
    y_tr_int = np.argmax(Y_tr, 1) if Y_tr.ndim == 2 else Y_tr.astype(int)
    cw = {i: float(w) for i, w in enumerate(
        compute_class_weight("balanced",
                             classes=np.array([0, 1]),
                             y=y_tr_int))}
    # Boost fall class thêm nữa do imbalance 1:5
    cw[1] = cw[1] * 1.3
    step(f"Class weights: not_fall={cw[0]:.2f} fall={cw[1]:.2f}")

    # ── Build model ───────────────────────────────────────────────────
    builder    = _select_model_builder()
    model_arch = CONFIG.get("model_arch", "ms_tcn")
    model      = builder(n_features, seq_len, CONFIG) if model_arch == "ms_tcn" \
                 else builder(n_features, seq_len)

    model.summary(print_fn=lambda x: print(f"    {x}"))

    lr = CONFIG.get("learning_rate", 0.001)
    optimizer = Adam(learning_rate=lr, clipnorm=1.0)  # gradient clipping

    # ── Compile với auxiliary loss ────────────────────────────────────
    aux_weight = CONFIG.get("aux_loss_weight", 0.20)
    focal      = _focal_loss(CONFIG.get("focal_alpha", 0.60),
                             CONFIG.get("focal_gamma", 2.5))

    if model_arch == "ms_tcn":
        model.compile(
            optimizer=optimizer,
            loss={
                "output":     focal,
                "fall_moment": "binary_crossentropy",
            },
            loss_weights={
                "output":      1.0,
                "fall_moment": aux_weight,
            },
            metrics={"output": "accuracy"}
        )
    else:
        model.compile(optimizer=optimizer, loss=focal, metrics=["accuracy"])

    # ── Callbacks ─────────────────────────────────────────────────────
    model_dir = os.path.dirname(CONFIG["model_path"])
    os.makedirs(model_dir, exist_ok=True)

    cbs, RecallCB = _make_callbacks(
        CONFIG["model_path"], CONFIG,
        total_epochs=CONFIG["epochs"], lr=lr, phase="main")
    cbs.insert(0, RecallCB((X_val, Y_val)))

    # ── Fit ───────────────────────────────────────────────────────────
    if model_arch == "ms_tcn":
        # Keras 3 không hỗ trợ class_weight/sample_weight với multi-output dict
        # → class imbalance đã được xử lý bởi focal_alpha=0.60 + loss_weights
        # → Y_tr_smooth đã qua mixup/label_smoothing bù imbalance
        train_targets = {"output": Y_tr_smooth, "fall_moment": y_aux_tr}
        val_targets   = {"output": Y_val,       "fall_moment": y_aux_val}
        history = model.fit(
            X_tr, train_targets,
            validation_data=(X_val, val_targets),
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            callbacks=cbs, verbose=1)
    else:
        history = model.fit(
            X_tr, Y_tr_smooth,
            validation_data=(X_val, Y_val),
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            class_weight=cw,
            callbacks=cbs, verbose=1)

    ok("Training done.")

    # ── Export TFLite ─────────────────────────────────────────────────
    try:
        step("Exporting TFLite...")
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        tfl  = conv.convert()
        os.makedirs(os.path.dirname(CONFIG["tflite_path"]), exist_ok=True)
        with open(CONFIG["tflite_path"], "wb") as f:
            f.write(tfl)
        ok(f"TFLite: {CONFIG['tflite_path']}")
    except Exception as e:
        warn(f"TFLite export failed: {e}")

    return model


# ======================================================================
#  BUOC 3: EVALUATE
# ======================================================================
def run_evaluate(model=None, eval_real_test=False):
    banner("BUOC 3/4 - EVALUATE")
    try:
        import tensorflow as tf
        from sklearn.metrics import (classification_report, confusion_matrix,
                                     precision_recall_curve, roc_auc_score)
    except ImportError as e:
        err(f"{e}"); return None

    if model is None:
        if not os.path.exists(CONFIG["model_path"]):
            err("No model."); return None
        model = _load_keras_model(CONFIG["model_path"])

    p = CONFIG["processed_dir"]
    if eval_real_test:
        test_dir = CONFIG["real_test_dir"]
        step(f"Evaluate on real_test: {test_dir}")
        # TODO: extract real_test sequences rồi evaluate
        return model

    X_test_path = f"{p}/X_test.npy"
    if not os.path.exists(X_test_path):
        warn("No test set."); return model

    X_te = np.load(X_test_path)
    Y_te = np.load(f"{p}/Y_test.npy")

    out = model(X_te, training=False)
    if isinstance(out, (list, tuple)):
        probs = np.array(out[0])
    else:
        probs = np.array(out)

    y_pred = np.argmax(probs, axis=1)
    y_true = Y_te if Y_te.ndim == 1 else Y_te.astype(int)

    print(classification_report(y_true, y_pred,
                                 target_names=CONFIG["classes"]))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall    = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * recall * precision / (recall + precision + 1e-8)
    f2 = (5 * precision * recall) / (4 * precision + recall + 1e-8)
    fp_rate = fp / (fp + tn + 1e-8)

    try:
        auc = roc_auc_score(y_true, probs[:, 1])
    except Exception:
        auc = 0.0

    print(f"\n  Recall (Fall): {recall*100:.2f}%")
    print(f"  Precision    : {precision*100:.2f}%")
    print(f"  F1-score     : {f1*100:.2f}%")
    print(f"  F2-score     : {f2*100:.2f}%")
    print(f"  FP rate      : {fp_rate*100:.2f}%")
    print(f"  ROC-AUC      : {auc:.4f}")
    print(f"  Confusion matrix:\n    TN={tn} FP={fp}\n    FN={fn} TP={tp}")

    # Auto tune threshold
    if CONFIG.get("auto_threshold", True):
        _auto_tune_inference_params(model, X_te, y_true)

    return model


# ======================================================================
#  INFERENCE: Load model
# ======================================================================
def _load_model():
    use_tflite = CONFIG.get("use_tflite", False)
    if use_tflite:
        tflite_path = CONFIG["tflite_path"]
        if os.path.exists(tflite_path):
            try:
                import tensorflow as tf
                interp = tf.lite.Interpreter(model_path=tflite_path)
                interp.allocate_tensors()
                ok(f"Loaded TFLite: {tflite_path}")
                return interp, "tflite"
            except Exception as e:
                warn(f"TFLite load failed: {e}. Fallback Keras.")
        else:
            warn(f"TFLite not found: {tflite_path}. Fallback Keras.")

    model = _load_keras_model(CONFIG["model_path"])
    ok(f"Loaded Keras: {CONFIG['model_path']}")
    return model, "keras"

def _predict(model, mode, inp):
    """inp: (batch, seq_len, n_features) → returns (batch, 2)"""
    if mode == "tflite":
        results = []
        ind  = model.get_input_details()
        outd = model.get_output_details()
        for i in range(inp.shape[0]):
            model.set_tensor(ind[0]['index'], inp[i:i+1])
            model.invoke()
            results.append(model.get_tensor(outd[0]['index'])[0])
        return np.array(results, dtype=np.float32)

    import tensorflow as tf
    out = model(inp, training=False)
    if isinstance(out, (list, tuple)):
        return np.array(out[0])   # chỉ lấy main output, bỏ auxiliary
    return np.array(out)


# ======================================================================
#  BAYESIAN PROBABILITY SMOOTHER
# ======================================================================
class BayesianSmoother:
    """
    Làm mượt xác suất theo thời gian với exponential window.
    Phản ứng nhanh hơn khi prob tăng (ngã), chậm hơn khi giảm (hồi phục).
    """
    def __init__(self, alpha_rise=0.60, alpha_fall=0.35):
        self.alpha_rise = alpha_rise
        self.alpha_fall = alpha_fall
        self.smooth     = 0.0

    def update(self, raw_prob: float) -> float:
        if raw_prob > self.smooth:
            alpha = self.alpha_rise
        else:
            alpha = self.alpha_fall
        self.smooth = alpha * raw_prob + (1.0 - alpha) * self.smooth
        return self.smooth

    def reset(self):
        self.smooth = 0.0


# ======================================================================
#  SCENE CLASSIFIER (nhận diện điều kiện camera)
# ======================================================================
class SceneClassifier:
    """
    Nhận diện scene từ features để điều chỉnh threshold:
    - Top-down camera (CCTV trần nhà)
    - Low-light
    - Người quá gần camera
    """
    def __init__(self, window=30):
        self.window  = window
        self.history = []  # list of feature vectors

    def update(self, feat_vec):
        self.history.append(feat_vec.copy())
        if len(self.history) > self.window:
            self.history.pop(0)

    def get_scene_adjustment(self) -> float:
        """Trả về delta threshold (dương = tăng threshold, âm = giảm)."""
        if len(self.history) < 5:
            return 0.0

        arr = np.array(self.history[-10:], dtype=np.float32)

        # Top-down detection
        mean_bbox_ar   = float(np.mean(arr[:, 4]))  # bbox_ar
        mean_spine_h   = float(np.mean(arr[:, 1]))  # spine_horiz
        mean_head_hip  = float(np.mean(arr[:, 6]))  # head_hip_dy
        td_score = (
            float(np.clip(1.0 - abs(mean_bbox_ar - 1.0) * 2, 0, 1)) * 0.35
            + float(np.clip((mean_spine_h - 0.45) / 0.40, 0, 1)) * 0.40
            + float(np.clip(1.0 - mean_head_hip / 0.20, 0, 1)) * 0.25
        )
        is_topdown = td_score > 0.45

        # Low-light detection
        mean_vis = float(np.mean(arr[:, 17]))
        is_lowlight = mean_vis < 0.45

        delta = 0.0
        if is_topdown and CONFIG.get("scene_aware_threshold", True):
            delta += CONFIG.get("scene_topdown_boost", 0.08)
        if is_lowlight and CONFIG.get("scene_aware_threshold", True):
            delta -= CONFIG.get("scene_lowlight_reduce", 0.05)

        return float(np.clip(delta, -0.10, 0.15))

    def reset(self):
        self.history.clear()


# ======================================================================
#  STILLNESS VALIDATOR
# ======================================================================
class StillnessValidator:
    def __init__(self):
        self._buf = []

    def update(self, feat_vec):
        self._buf.append(feat_vec.copy())
        win = CONFIG.get("stillness_window", 45)
        if len(self._buf) > win:
            self._buf.pop(0)

    def is_real_fall(self):
        if len(self._buf) < CONFIG.get("stillness_min_frames", 10):
            return False
        arr  = np.array(self._buf, dtype=np.float32)
        v12  = arr[:, 12]; v13  = arr[:, 13]
        m12  = float(np.mean(np.abs(v12)))
        m13  = float(np.mean(np.abs(v13)))
        thr  = CONFIG.get("stillness_threshold", 0.04)
        return (m12 < thr and m13 < thr)

    def reset(self):
        self._buf.clear()


# ======================================================================
#  PERSON STATE
# ======================================================================
_PERSON_COLORS = [(0,200,255), (0,255,120), (255,120,0), (200,0,255)]

class PersonState:
    def __init__(self, pid):
        self.pid         = pid
        self.color       = _PERSON_COLORS[pid % len(_PERSON_COLORS)]
        self.buf         = []
        self.prev_state  = None
        self.feat_history= []
        self.smoother    = BayesianSmoother(
            alpha_rise=CONFIG.get("bayes_rise_alpha", 0.60),
            alpha_fall=CONFIG.get("bayes_alpha", 0.35))
        self.scene       = SceneClassifier()
        self.validator   = StillnessValidator()
        self.accum       = 0.0
        self.confirm_cnt = 0
        self.post_mode   = False
        self.post_frames = 0
        self.none_streak = 0
        self.last_prob   = 0.0
        self.skip_cnt    = 0
        self.vis_score   = 1.0
        self.fall_detected_time  = 0.0
        self.fall_warned = False
        self.fall_state  = "ok"
        self.recovery_nose_buf = []
        self._needs_predict = False

    def update_accum(self, prob, decay=0.92):
        self.accum = max(prob, self.accum * decay)
        return self.accum

    def reset_fall(self):
        self.post_mode   = False
        self.post_frames = 0
        self.confirm_cnt = 0
        self.accum       = 0.0
        self.smoother.reset()
        self.validator.reset()
        self.fall_warned = False
        self.fall_state  = "ok"

    def update_visibility(self, feat_vec):
        if len(feat_vec) > 17:
            self.vis_score = float(feat_vec[17])

    def check_recovery(self, feat_vec, fps):
        """Kiểm tra người có tự đứng dậy không."""
        self.recovery_nose_buf.append(float(feat_vec[13]))  # v_nose_y
        max_buf = int(fps * CONFIG.get("recovery_window_sec", 3.0))
        if len(self.recovery_nose_buf) > max_buf:
            self.recovery_nose_buf.pop(0)
        if len(self.recovery_nose_buf) < int(fps * 0.5):
            return False
        recent_v = self.recovery_nose_buf[-int(fps * 0.5):]
        # Nose đang di chuyển lên (v_nose âm = lên trong coordinate system)
        return float(np.mean(recent_v)) < -CONFIG.get("head_rise_threshold", 0.05)


# ======================================================================
#  DRAW HELPERS
# ======================================================================
def _draw_landmarks(frame, lm2d, color=(0, 200, 255)):
    try:
        import mediapipe as mp
        h, w = frame.shape[:2]
        connections = mp.solutions.pose.POSE_CONNECTIONS
        for lm in lm2d:
            if lm.visibility > 0.3:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, color, -1)
        if connections:
            for (i, j) in connections:
                if (i < len(lm2d) and j < len(lm2d)
                        and lm2d[i].visibility > 0.3
                        and lm2d[j].visibility > 0.3):
                    x1, y1 = int(lm2d[i].x * w), int(lm2d[i].y * h)
                    x2, y2 = int(lm2d[j].x * w), int(lm2d[j].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), color, 1)
    except Exception:
        pass

def _update_threshold(mean_vis):
    base = CONFIG.get("confidence_threshold", 0.60)
    if mean_vis < 0.35:
        return max(0.40, base - 0.08)
    elif mean_vis < 0.50:
        return max(0.45, base - 0.04)
    return base


# ======================================================================
#  DATASET RECORDER
# ======================================================================
RECORDER_LABELS = {
    ord('1'): ("fall",              "fall",      (0,   0,   220)),
    ord('2'): ("not_fall",          "not_fall",  (0,   200, 60)),
    ord('3'): ("action_like_fall",  "not_fall",  (0,   140, 255)),
}
RECORDER_LABEL_NAMES = {
    ord('1'): "[1] FALL",
    ord('2'): "[2] NOT FALL",
    ord('3'): "[3] ACTION LIKE FALL",
}

def _load_registry():
    reg = CONFIG.get("extract_registry", "")
    if not reg or not os.path.exists(reg):
        return []
    with open(reg) as f:
        return [l.strip() for l in f if l.strip()]

def _next_ud_index() -> int:
    import re
    max_idx = -1
    for cls in CONFIG["classes"]:
        d = os.path.join(CONFIG["dataset_dir"], cls)
        if not os.path.isdir(d): continue
        for f in os.listdir(d):
            m = re.match(r"U_D_(\d{4})\.", f)
            if m: max_idx = max(max_idx, int(m.group(1)))
    for entry in _load_registry():
        m = re.search(r"U_D_(\d{4})\.", entry)
        if m: max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1

def _save_recorded_clip(frames, label_key, fps, model_pred, ts_str):
    if label_key not in RECORDER_LABELS: return ""
    semantic_label, cls_dir_name, _ = RECORDER_LABELS[label_key]
    idx      = _next_ud_index()
    fname    = f"U_D_{idx:04d}.mp4"
    cls_dir  = os.path.join(CONFIG["dataset_dir"], cls_dir_name)
    os.makedirs(cls_dir, exist_ok=True)
    out_path = os.path.join(cls_dir, fname)
    if frames:
        h, w   = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw     = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for fr in frames: vw.write(fr)
        vw.release()
    rec_dir  = CONFIG.get("recorder_dir", "dataset_recorder")
    os.makedirs(rec_dir, exist_ok=True)
    log_path = os.path.join(rec_dir, "recorder_log.csv")
    write_hdr = not os.path.exists(log_path)
    with open(log_path, "a") as lf:
        if write_hdr:
            lf.write("timestamp,filename,semantic_label,dataset_class,"
                     "model_confidence,frames_saved\n")
        lf.write(f"{ts_str},{fname},{semantic_label},{cls_dir_name},"
                 f"{model_pred:.3f},{len(frames)}\n")
    ok(f"Saved: {out_path}  [{semantic_label}]  ({len(frames)} frames)")
    return out_path

class _FrameRingBuffer:
    def __init__(self, maxlen):
        self._buf = []; self._max = maxlen
    def push(self, frame):
        self._buf.append(frame.copy())
        if len(self._buf) > self._max: self._buf.pop(0)
    def get_all(self): return list(self._buf)
    def clear(self):   self._buf.clear()
    def __len__(self): return len(self._buf)

def _draw_recorder_menu(frame, state, countdown, label_key, last_pred):
    h, w = frame.shape[:2]
    if state == "idle": return
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h//2 - 90), (w, h//2 + 130), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cy = h // 2 - 65
    if state == "pre_record":
        cv2.putText(frame, f"RECORDING...  {countdown:.1f}s remaining",
            (w//2 - 160, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 80, 255), 2)
        cy += 36
        cv2.putText(frame, f"Model prediction: {last_pred*100:.0f}%",
            (w//2 - 130, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cy += 40
        cv2.putText(frame, "Label this clip:",
            (w//2 - 90, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cy += 34
        for key, txt in RECORDER_LABEL_NAMES.items():
            _, _, col = RECORDER_LABELS[key]
            cv2.putText(frame, txt, (w//2 - 110, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, col, 2)
            cy += 30
        cv2.putText(frame, "[ESC] Cancel",
            (w//2 - 55, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 120, 120), 1)
    elif state == "saved":
        _, _, col = RECORDER_LABELS.get(label_key, (None, None, (200, 200, 200)))
        lname = RECORDER_LABEL_NAMES.get(label_key, "?")
        cv2.putText(frame, f"SAVED as {lname}",
            (w//2 - 140, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    elif state == "cancelled":
        cv2.putText(frame, "Clip cancelled (not saved)",
            (w//2 - 160, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)


# ======================================================================
#  INCREMENTAL LEARNING (giữ nguyên logic từ v7, cập nhật path)
# ======================================================================
def _load_registry():
    reg = CONFIG.get("extract_registry", "")
    if not reg or not os.path.exists(reg): return []
    with open(reg) as f: return [l.strip() for l in f if l.strip()]

def _save_registry(entries):
    reg = CONFIG.get("extract_registry", "")
    if not reg: return
    os.makedirs(os.path.dirname(reg) or ".", exist_ok=True)
    with open(reg, "w") as f:
        for e in entries: f.write(e + "\n")

def run_extract_incremental():
    """Extract chỉ các file U_D_xxxx mới (chưa có trong registry)."""
    banner("EXTRACT INCREMENTAL (chi U_D_xxxx moi)")
    from sklearn.model_selection import train_test_split as _tts

    registry     = set(_load_registry())
    seq_len      = CONFIG["sequence_length"]
    new_seqs     = []
    new_labels   = []
    new_files    = []
    detector     = _make_pose_detector("VIDEO")
    _ts_ms       = 1
    stride       = CONFIG.get("extract_stride", 5)

    for cls_idx, cls_name in enumerate(CONFIG["classes"]):
        cls_dir = os.path.join(CONFIG["dataset_dir"], cls_name)
        if not os.path.isdir(cls_dir): continue
        videos = sorted([f for f in os.listdir(cls_dir)
                         if f.startswith("U_D_")
                         and f.lower().endswith((".mp4",".avi",".mov",".mkv"))])
        step(f"[{cls_name}] {len(videos)} U_D videos")
        for vfile in videos:
            full_key = f"{cls_name}/{vfile}"
            if full_key in registry:
                continue

            cap = cv2.VideoCapture(os.path.join(cls_dir, vfile))
            if not cap.isOpened(): continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_ms = max(1, int(1000.0 / fps))
            buf, prev, none_s, feat_hist = [], None, 0, []
            while True:
                ret, frame = cap.read()
                if not ret: break
                lm_list = _run_pose_on_frame(detector, enhance_frame(frame),
                                             timestamp_ms=_ts_ms,
                                             use_video_mode=True)
                _ts_ms += frame_ms
                if lm_list is None:
                    none_s += 1
                    if none_s > 10: buf.clear(); prev = None; none_s = 0
                    continue
                none_s = 0
                feats, prev = extract_features(lm_list[0], prev, feat_hist)
                feat_hist.append(feats)
                if len(feat_hist) > 30: feat_hist.pop(0)
                buf.append(feats)
                if len(buf) >= seq_len:
                    new_seqs.append(np.array(buf[-seq_len:], dtype=np.float32))
                    new_labels.append(cls_idx)
            cap.release()
            new_files.append(full_key)
            ok(f"  {full_key}: {sum(1 for l in new_labels if l == cls_idx)} seqs")

    detector.close()

    if not new_seqs:
        step("Không có file mới."); return None, None

    X_new = np.array(new_seqs, dtype=np.float32)
    y_new = np.array(new_labels, dtype=np.int32)
    ok(f"Mới: {len(X_new)} sequences  (fall={np.sum(y_new==1)}, not_fall={np.sum(y_new==0)})")

    p = CONFIG["processed_dir"]
    os.makedirs(p, exist_ok=True)
    np.save(f"{p}/sequences_new.npy", X_new)
    np.save(f"{p}/labels_new.npy",    y_new)

    new_registry = list(set(_load_registry()) | set(new_files))
    _save_registry(new_registry)
    ok(f"Registry updated: {len(new_registry)} entries")

    return X_new, y_new


def run_finetune_incremental(model=None):
    """Fine-tune nhanh với data U_D mới."""
    banner("INCREMENTAL LEARNING")
    try:
        import tensorflow as tf
        import keras
        from keras.utils import to_categorical
        from keras.optimizers import Adam
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError as e:
        err(f"{e}"); return

    p = CONFIG["processed_dir"]
    Xn_path = f"{p}/sequences_new.npy"
    if not os.path.exists(Xn_path):
        step("Chạy extract_incremental trước..."); return

    X_new = np.load(Xn_path)
    y_new = np.load(f"{p}/labels_new.npy")

    if model is None:
        model = _load_keras_model(CONFIG["model_path"])

    # Kết hợp với một phần data cũ nếu có (replay buffer)
    seqs_path = f"{p}/sequences_aug.npy"
    if os.path.exists(seqs_path):
        X_old = np.load(seqs_path); y_old = np.load(f"{p}/labels_aug.npy")
        n_replay = min(len(X_old), len(X_new) * 3)
        rng = np.random.default_rng(99)
        idx = rng.choice(len(X_old), size=n_replay, replace=False)
        X_combined = np.concatenate([X_new, X_old[idx]])
        y_combined = np.concatenate([y_new, y_old[idx]])
        ok(f"Replay buffer: {n_replay} old + {len(X_new)} new = {len(X_combined)}")
    else:
        X_combined = X_new; y_combined = y_new

    Y_combined = to_categorical(y_combined, 2)
    cw = {i: float(w) for i, w in enumerate(
        compute_class_weight("balanced", classes=np.array([0,1]), y=y_combined))}
    cw[1] *= 1.3

    lr = CONFIG.get("incremental_lr", 0.00008)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    focal = _focal_loss(CONFIG.get("focal_alpha", 0.60), CONFIG.get("focal_gamma", 2.5))

    arch = CONFIG.get("model_arch", "ms_tcn")
    if arch == "ms_tcn":
        model.compile(optimizer=optimizer,
                      loss={"output": focal, "fall_moment": "binary_crossentropy"},
                      loss_weights={"output": 1.0, "fall_moment": 0.20},
                      metrics={"output": "accuracy"})
    else:
        model.compile(optimizer=optimizer, loss=focal, metrics=["accuracy"])

    # Val
    if os.path.exists(f"{p}/X_val.npy"):
        X_val = np.load(f"{p}/X_val.npy")
        y_val = np.load(f"{p}/Y_val.npy")
        Y_val = to_categorical(y_val, 2)
    else:
        from sklearn.model_selection import train_test_split
        X_combined, X_val, Y_combined, Y_val = train_test_split(
            X_combined, Y_combined, test_size=0.20, random_state=42)

    cbs, RecallCB = _make_callbacks(CONFIG["model_path"], CONFIG,
                                    total_epochs=CONFIG["incremental_epochs"],
                                    lr=lr, phase="incremental")
    cbs.insert(0, RecallCB((X_val, Y_val)))

    seq_len = X_combined.shape[1]
    y_int   = np.argmax(Y_combined, 1) if Y_combined.ndim == 2 else y_combined
    y_aux   = _generate_aux_labels(y_int, seq_len)
    y_aux_v = _generate_aux_labels(y_val.astype(int) if y_val.ndim==1 else np.argmax(Y_val,1), seq_len)

    if arch == "ms_tcn":
        model.fit(X_combined, {"output": Y_combined, "fall_moment": y_aux},
                  validation_data=(X_val, {"output": Y_val, "fall_moment": y_aux_v}),
                  epochs=CONFIG["incremental_epochs"],
                  batch_size=CONFIG["batch_size"] // 2,
                  callbacks=cbs, verbose=1)
    else:
        model.fit(X_combined, Y_combined, validation_data=(X_val, Y_val),
                  epochs=CONFIG["incremental_epochs"],
                  batch_size=CONFIG["batch_size"] // 2,
                  class_weight=cw, callbacks=cbs, verbose=1)

    ok("Incremental learning done.")
    return model


# ======================================================================
#  FINE-TUNE REAL (với video real_test_v8)
# ======================================================================
def run_finetune_real(model=None):
    """Fine-tune 2-phase với video từ real_test_v8/."""
    banner("FINE-TUNE REAL (real_test_v8)")
    try:
        import tensorflow as tf
        import keras
        from keras.utils import to_categorical
        from keras.optimizers import Adam
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError as e:
        err(f"{e}"); return None

    real_dir = CONFIG["real_test_dir"]
    _VEXT    = (".mp4",".avi",".mov",".mkv",".wmv",".flv",".m4v",".3gp")

    def _get_videos(subdir):
        d = os.path.join(real_dir, subdir)
        if not os.path.isdir(d): return []
        return [f for f in os.listdir(d) if f.lower().endswith(_VEXT)]

    fall_vids     = _get_videos("fall")
    notfall_vids  = _get_videos("not_fall")
    action_vids   = _get_videos("action_like_fall")

    if not fall_vids and not notfall_vids and not action_vids:
        warn("Không có video trong real_test_v8/"); return model

    step(f"fall:{len(fall_vids)} not_fall:{len(notfall_vids)} action:{len(action_vids)}")

    if model is None:
        model = _load_keras_model(CONFIG["model_path"])

    model_path = CONFIG["model_path"]
    seq_len    = CONFIG["sequence_length"]

    # Extract sequences từ real_test videos
    detector   = _make_pose_detector("VIDEO")
    _ts_ms     = 1

    all_seqs = []; all_labels = []

    subdirs = [
        ("fall",             1, fall_vids),
        ("not_fall",         0, notfall_vids),
        ("action_like_fall", 0, action_vids),  # action_like_fall → label 0
    ]

    for subdir, label, videos in subdirs:
        if not videos: continue
        step(f"Extracting {subdir}: {len(videos)} videos")
        for vfile in videos:
            cap = cv2.VideoCapture(os.path.join(real_dir, subdir, vfile))
            if not cap.isOpened(): continue
            fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_ms = max(1, int(1000.0 / fps))
            buf, prev, none_s, feat_hist = [], None, 0, []
            n_seq = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_enh = enhance_frame(frame)
                lm_list   = _run_pose_on_frame(detector, frame_enh,
                                               timestamp_ms=_ts_ms,
                                               use_video_mode=True)
                _ts_ms += frame_ms
                if lm_list is None:
                    none_s += 1
                    if none_s > 10: buf.clear(); prev = None; none_s = 0
                    continue
                none_s = 0
                feats, prev = extract_features(lm_list[0], prev, feat_hist)
                feat_hist.append(feats)
                if len(feat_hist) > 30: feat_hist.pop(0)
                buf.append(feats)
                if len(buf) >= seq_len:
                    all_seqs.append(np.array(buf[-seq_len:], dtype=np.float32))
                    all_labels.append(label)
                    n_seq += 1
            cap.release()
            ok(f"  {vfile}: +{n_seq} seqs")

    detector.close()

    if not all_seqs:
        warn("Không extract được sequences."); return model

    X_real = np.array(all_seqs[:CONFIG.get("finetune_real_max_seqs", 6000)],
                      dtype=np.float32)
    y_real = np.array(all_labels[:len(X_real)], dtype=np.int32)
    ok(f"Real sequences: {len(X_real)}  (fall={np.sum(y_real==1)}, not_fall={np.sum(y_real==0)})")

    # Replay buffer từ data cũ
    p = CONFIG["processed_dir"]
    replay_ratio = CONFIG.get("finetune_real_replay_ratio", 0.40)
    if os.path.exists(f"{p}/sequences_aug.npy"):
        X_old = np.load(f"{p}/sequences_aug.npy")
        y_old = np.load(f"{p}/labels_aug.npy")
        n_replay = int(len(X_real) * replay_ratio / (1 - replay_ratio))
        n_replay = min(n_replay, len(X_old))
        rng = np.random.default_rng(77)
        idx = rng.choice(len(X_old), size=n_replay, replace=False)
        X_tr = np.concatenate([X_real, X_old[idx]])
        y_tr = np.concatenate([y_real, y_old[idx]])
        ok(f"Replay: {n_replay} old + {len(X_real)} real = {len(X_tr)}")
    else:
        X_tr = X_real; y_tr = y_real

    from sklearn.model_selection import train_test_split as _tts
    if len(np.unique(y_tr)) > 1:
        X_tr, X_val, y_tr, y_val = _tts(X_tr, y_tr, test_size=0.15,
                                          random_state=42, stratify=y_tr)
    else:
        X_val = X_tr[:max(10,len(X_tr)//5)]
        y_val = y_tr[:max(10,len(y_tr)//5)]

    Y_tr  = to_categorical(y_tr, 2)
    Y_val = to_categorical(y_val, 2)
    cw    = {i: float(w) for i, w in enumerate(
        compute_class_weight("balanced", classes=np.array([0,1]), y=y_tr))}
    cw[1] *= 1.3

    arch = CONFIG.get("model_arch", "ms_tcn")
    focal = _focal_loss(CONFIG.get("focal_alpha", 0.60), CONFIG.get("focal_gamma", 2.5))

    # ── Phase 1: freeze TCN layers, tune head ─────────────────────────
    step("Phase 1: freeze backbone, tune head...")
    for layer in model.layers:
        if "mstcn" in layer.name.lower() or "tcn" in layer.name.lower():
            layer.trainable = False

    lr = CONFIG.get("finetune_real_lr", 0.00004)
    model.compile(optimizer=Adam(lr * 2, clipnorm=1.0),
                  loss={"output": focal, "fall_moment": "binary_crossentropy"} if arch == "ms_tcn"
                       else focal,
                  loss_weights={"output": 1.0, "fall_moment": 0.20} if arch == "ms_tcn"
                               else None,
                  metrics={"output": "accuracy"} if arch == "ms_tcn" else ["accuracy"])

    import tempfile
    import tempfile as _tf
    _, ft_tmp1 = _tf.mkstemp(suffix=".keras"); _tf.os.close(_)
    _, ft_tmp2 = _tf.mkstemp(suffix=".keras"); _tf.os.close(_)

    epochs = CONFIG.get("finetune_real_epochs", 20)
    cbs1, RecallCB = _make_callbacks(ft_tmp1, CONFIG,
                                     total_epochs=epochs//2, lr=lr*2, phase="finetune")
    cbs1.insert(0, RecallCB((X_val, Y_val)))

    seq_len_ft = X_tr.shape[1]
    y_aux_tr_ft = _generate_aux_labels(y_tr, seq_len_ft)
    y_aux_val_ft= _generate_aux_labels(y_val, seq_len_ft)

    if arch == "ms_tcn":
        model.fit(X_tr, {"output": Y_tr, "fall_moment": y_aux_tr_ft},
                  validation_data=(X_val, {"output": Y_val, "fall_moment": y_aux_val_ft}),
                  epochs=epochs//2, batch_size=CONFIG["batch_size"]//2,
                  callbacks=cbs1, verbose=1)
    else:
        model.fit(X_tr, Y_tr, validation_data=(X_val, Y_val),
                  epochs=epochs//2, batch_size=CONFIG["batch_size"]//2,
                  class_weight=cw, callbacks=cbs1, verbose=1)

    # ── Phase 2: unfreeze all ─────────────────────────────────────────
    step("Phase 2: unfreeze all, LR/5...")
    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer=Adam(lr/5, clipnorm=1.0),
                  loss={"output": focal, "fall_moment": "binary_crossentropy"} if arch=="ms_tcn"
                       else focal,
                  loss_weights={"output": 1.0, "fall_moment": 0.20} if arch=="ms_tcn"
                               else None,
                  metrics={"output": "accuracy"} if arch=="ms_tcn" else ["accuracy"])

    cbs2, _ = _make_callbacks(ft_tmp2, CONFIG,
                               total_epochs=epochs//2, lr=lr/5, phase="finetune")
    cbs2.insert(0, RecallCB((X_val, Y_val)))

    if arch == "ms_tcn":
        model.fit(X_tr, {"output": Y_tr, "fall_moment": y_aux_tr_ft},
                  validation_data=(X_val, {"output": Y_val, "fall_moment": y_aux_val_ft}),
                  epochs=epochs//2, batch_size=CONFIG["batch_size"]//2,
                  callbacks=cbs2, verbose=1)
    else:
        model.fit(X_tr, Y_tr, validation_data=(X_val, Y_val),
                  epochs=epochs//2, batch_size=CONFIG["batch_size"]//2,
                  class_weight=cw, callbacks=cbs2, verbose=1)

    # Chọn model tốt hơn và lưu
    ft_model_path = CONFIG["finetune_real_model_path"]
    import shutil
    best_tmp = ft_tmp2 if os.path.exists(ft_tmp2) else ft_tmp1
    os.makedirs(os.path.dirname(ft_model_path), exist_ok=True)
    shutil.copy2(best_tmp, ft_model_path)
    shutil.copy2(best_tmp, model_path)
    ok(f"Fine-tune real done. Model: {model_path}")

    for tmp in [ft_tmp1, ft_tmp2]:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass

    return _load_keras_model(model_path)


# ======================================================================
#  HARD NEGATIVE MINING v2
# ======================================================================
def run_hard_negative_mining(model=None, source_dir=None, output_dir=None):
    banner("HARD NEGATIVE MINING v2")
    import re as _re, shutil as _sh

    if model is None:
        if not os.path.exists(CONFIG["model_path"]):
            err("No model."); return 0
        model = _load_keras_model(CONFIG["model_path"])

    nfall_dir    = source_dir or os.path.join(CONFIG["dataset_dir"], "not_fall")
    hard_neg_dir = output_dir or CONFIG["hard_neg_dir"]
    os.makedirs(hard_neg_dir, exist_ok=True)

    if not os.path.isdir(nfall_dir):
        warn(f"Not found: {nfall_dir}"); return 0

    seq_len   = CONFIG["sequence_length"]
    threshold = max(CONFIG["hard_neg_threshold"], 0.65)
    clip_sec  = CONFIG["hard_neg_clip_sec"]
    ew        = CONFIG["ensemble_weights"]

    videos = [f for f in os.listdir(nfall_dir)
              if f.lower().endswith((".mp4",".avi",".mov",".mkv"))]
    step(f"Mining {len(videos)} not_fall videos  threshold={threshold:.2f}")

    detector = _make_pose_detector("VIDEO")
    _ts_ms   = 1
    total_mined = 0

    for vfile in videos:
        vpath = os.path.join(nfall_dir, vfile)
        cap   = cv2.VideoCapture(vpath)
        if not cap.isOpened(): continue

        fps       = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_ms  = max(1, int(1000.0 / fps))
        min_gap   = int(fps * 1.5)

        buf, prev, none_s, feat_hist = [], None, 0, []
        frame_idx = 0; mined_times = []; last_mined = -9999
        all_frames = []

        while True:
            ret, frame = cap.read()
            if not ret: break
            all_frames.append(frame.copy())
            _ts = _ts_ms; _ts_ms += frame_ms
            lm_list = _run_pose_on_frame(detector, enhance_frame(frame),
                                          timestamp_ms=_ts, use_video_mode=True)
            if lm_list is None:
                none_s += 1
                if none_s > 10: buf.clear(); prev = None; none_s = 0
                frame_idx += 1; continue
            none_s = 0
            feats, prev = extract_features(lm_list[0], prev, feat_hist)
            feat_hist.append(feats)
            if len(feat_hist) > 30: feat_hist.pop(0)
            buf.append(feats)

            if len(buf) >= seq_len:
                inp      = np.expand_dims(np.array(buf[-seq_len:], dtype=np.float32), 0)
                lstm_p   = float(_predict(model, "keras", inp)[0][1])
                rule_p   = rule_based_score(buf[-seq_len:])
                ens_prob = ew[0] * lstm_p + ew[1] * rule_p

                if ens_prob >= threshold:
                    if frame_idx - last_mined < min_gap:
                        if mined_times and ens_prob > mined_times[-1][2]:
                            fs, fe, _ = mined_times[-1]
                            mined_times[-1] = (fs, fe, ens_prob)
                    else:
                        clip_frames = int(clip_sec * fps)
                        f_start = max(0, frame_idx - clip_frames)
                        f_end   = min(len(all_frames)-1, frame_idx + clip_frames//2)
                        mined_times.append((f_start, f_end, ens_prob))
                        last_mined = frame_idx

            frame_idx += 1
        cap.release()

        for idx, (fs, fe, prob) in enumerate(mined_times):
            out_name = f"hardneg_{os.path.splitext(vfile)[0]}_t{fs}_p{prob:.2f}.mp4"
            out_path = os.path.join(hard_neg_dir, out_name)
            if os.path.exists(out_path): continue
            h, w = all_frames[0].shape[:2]
            vw   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
            for fr in all_frames[fs:fe+1]: vw.write(fr)
            vw.release()
            ok(f"  Mined: {out_name}  (prob={prob:.2f})")
            total_mined += 1

    detector.close()
    ok(f"Total {total_mined} clips → {hard_neg_dir}")

    # Lọc prob >= 0.80 → copy sang action_like_fall
    _is_main = (output_dir is None)
    if total_mined > 0 and _is_main:
        _action_dir = os.path.join(CONFIG["real_test_dir"], "action_like_fall")
        _prob_thr   = CONFIG.get("hard_neg_filter_prob", 0.80)
        os.makedirs(_action_dir, exist_ok=True)
        _all = [f for f in os.listdir(hard_neg_dir) if f.lower().endswith(".mp4")]
        _copied = 0
        for _clip in _all:
            _m2 = _re.search(r"_p(\d+\.\d+)\.mp4$", _clip)
            if _m2 and float(_m2.group(1)) >= _prob_thr:
                _s = os.path.join(hard_neg_dir, _clip)
                _d = os.path.join(_action_dir, _clip)
                if not os.path.exists(_d):
                    _sh.copy2(_s, _d); _copied += 1
        ok(f"Copied {_copied} clips → action_like_fall/ (prob>={_prob_thr})")

        # Dọn hard_negatives gốc
        for _clip in _all:
            try: os.remove(os.path.join(hard_neg_dir, _clip))
            except Exception: pass

    return total_mined


# ======================================================================
#  BUOC 4: REALTIME
# ======================================================================
def run_realtime(model=None, model_mode="keras", video_source=None):
    banner("BUOC 4/4 - REALTIME FALL DETECTION v8")

    if model is None:
        model, model_mode = _load_model()

    seq_len  = CONFIG["sequence_length"]
    threshold = CONFIG.get("confidence_threshold", 0.60)
    confirm_need = CONFIG.get("confirm_frames", 2)
    ew = CONFIG.get("ensemble_weights", [0.72, 0.28])

    # Video source
    if video_source is not None:
        cap = cv2.VideoCapture(video_source)
        step(f"Video: {video_source}")
    else:
        cap = cv2.VideoCapture(0)
        # Set webcam resolution cao hơn để tránh vỡ hình khi display
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        step("Webcam (0)")

    if not cap.isOpened():
        err("Cannot open video source."); return

    cap_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fr_ms   = max(1, int(1000.0 / cap_fps))

    # frame_resize: dùng để xử lý pose (nhỏ hơn = nhanh hơn)
    # display_size: hiển thị ở độ phân giải gốc của webcam nếu có thể
    target_w, target_h = CONFIG.get("frame_resize", (640, 480))  # tăng từ 480x360 lên 640x480
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    display_w = actual_w if actual_w > 0 else CONFIG.get("display_size", (960, 720))[0]
    display_h = actual_h if actual_h > 0 else CONFIG.get("display_size", (960, 720))[1]

    detector  = _make_pose_detector("VIDEO")  # VIDEO mode cho cả webcam và file (synchronous, không cần callback)
    ts_ms     = 1
    persons   = {}
    any_fall  = False
    fall_frame_snapshot = None
    fall_pid  = -1
    fall_ps   = None

    # Recorder state
    rec_state    = "idle"
    rec_buffer   = _FrameRingBuffer(int(cap_fps * 10))
    rec_start_t  = 0.0
    rec_duration = 5.0
    rec_label_key = -1
    rec_last_pred = 0.0

    # Log file
    log_fp = open(CONFIG["log_path"], "a")
    if os.path.getsize(CONFIG["log_path"]) == 0:
        log_fp.write("timestamp,person_id,confidence,method,event,detail\n")
    log_fp.flush()

    last_alert_t = 0.0
    frame_count  = 0

    def _match_persons(current_lms, prev_persons, seq_len):
        """Khớp landmarks với person state theo IoU bounding box."""
        if not prev_persons:
            return {i: lm_pair for i, lm_pair in enumerate(current_lms)
                    if i < CONFIG.get("max_persons", 3)}
        matched = {}
        used_pids = set()
        for i, lm_pair in enumerate(current_lms[:CONFIG.get("max_persons", 3)]):
            lm2d = lm_pair[0]
            xs   = [l.x for l in lm2d if l.visibility > 0.3]
            ys   = [l.y for l in lm2d if l.visibility > 0.3]
            if not xs:
                matched[i] = lm_pair; continue
            cx = np.mean(xs); cy = np.mean(ys)
            best_pid = i; best_dist = 1e9
            for pid, ps in prev_persons.items():
                if pid in used_pids: continue
                if ps.buf:
                    # Dùng vị trí hip trước đó
                    prev_cx = 0.5; prev_cy = 0.5
                    dist = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                    if dist < best_dist:
                        best_dist = dist; best_pid = pid
            matched[best_pid] = lm_pair
            used_pids.add(best_pid)
        return matched

    step("Press [Q] to quit | [R] to record clip | [S] to screenshot | [H] for help")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_orig = frame.copy()  # giữ frame gốc để display sắc nét
        frame = cv2.resize(frame, (target_w, target_h))
        frame_enh = enhance_frame(frame)

        # Recorder ring buffer
        rec_buffer.push(frame)
        rec_last_pred_display = 0.0

        _ts = ts_ms
        ts_ms += fr_ms
        frame_count += 1

        # Pose detection
        lm_list = _run_pose_on_frame(
            detector, frame_enh,
            timestamp_ms=_ts,
            use_video_mode=True)  # VIDEO mode cho cả webcam và file

        h, w = frame.shape[:2]
        any_fall = False

        if lm_list:
            matched = _match_persons(lm_list, persons, seq_len)

            for pid, lm_pair in matched.items():
                if pid not in persons:
                    persons[pid] = PersonState(pid)
                ps = persons[pid]
                ps.none_streak = 0

                lm2d = lm_pair[0]
                _draw_landmarks(frame, lm2d, color=ps.color)

                feats, ps.prev_state = extract_features(lm_pair, ps.prev_state, ps.feat_history)
                ps.buf.append(feats)
                if len(ps.buf) > seq_len: ps.buf.pop(0)
                ps.feat_history.append(feats)
                if len(ps.feat_history) > 30: ps.feat_history.pop(0)

                # Scene-aware threshold
                ps.scene.update(feats)
                scene_delta = ps.scene.get_scene_adjustment()
                dyn_threshold = _update_threshold(float(feats[17])) + scene_delta

                # ── Post-fall mode ─────────────────────────────────────
                if ps.post_mode:
                    ps.validator.update(feats)
                    ps.post_frames += 1
                    _fps_rt  = cap_fps
                    recovered = ps.check_recovery(feats, _fps_rt)
                    _elapsed  = time.time() - ps.fall_detected_time
                    _rec_win  = CONFIG.get("recovery_window_sec", 3.0)

                    if recovered and _elapsed <= _rec_win:
                        if not ps.fall_warned:
                            ps.fall_warned = True; ps.fall_state = "warned"
                            ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
                            log_fp.write(f"{ts_str},{pid},{ps.last_prob*100:.1f}%,"
                                         f"ensemble,warn,self_recovered\n")
                            log_fp.flush()
                        cv2.putText(frame, f"P{pid} FELL but recovered",
                            (10, 80 + pid*30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 200, 255), 2)
                        if _elapsed > _rec_win: ps.reset_fall()
                        continue

                    if ps.post_frames >= CONFIG["stillness_min_frames"]:
                        if ps.validator.is_real_fall():
                            any_fall = True
                            fall_frame_snapshot = frame.copy()
                            fall_pid = pid; fall_ps = ps
                            cv2.putText(frame,
                                f"P{pid} FALL CONFIRMED {ps.last_prob*100:.0f}%",
                                (10, 80+pid*30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0,0,255), 2)
                            ps.reset_fall()
                        else:
                            cv2.putText(frame, f"P{pid} False alarm",
                                (10, 80+pid*30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0,200,100), 2)
                            ps.reset_fall()
                    else:
                        _remain = int(_rec_win - _elapsed)
                        cv2.putText(frame,
                            f"P{pid} Verify {ps.post_frames}/{CONFIG['stillness_min_frames']}"
                            f" (rec? {_remain}s)",
                            (10, 80+pid*30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,165,255), 2)
                    continue

                # Near-camera warning
                nc = float(feats[24]) if len(feats) > 24 else 0.0
                if nc > 0.55:
                    nc_col = (0,100,255) if nc > 0.75 else (0,165,255)
                    cv2.putText(frame, f"P{pid} TOO CLOSE ({nc*100:.0f}%)",
                        (10, h-80-pid*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, nc_col, 1)

                if len(ps.buf) < seq_len: continue

                # Frame skip
                if ps.skip_cnt > 0 and ps.confirm_cnt == 0:
                    ps.skip_cnt -= 1
                    bar_w = int(120 * ps.last_prob)
                    bx, by = w-145, 60+pid*28
                    cv2.rectangle(frame, (bx,by), (bx+120,by+14), (50,50,50), -1)
                    cv2.rectangle(frame, (bx,by), (bx+bar_w,by+14),
                                  (0,0,200) if ps.last_prob >= dyn_threshold else (40,160,40), -1)
                    cv2.putText(frame, f"P{pid} {ps.last_prob*100:.0f}%",
                        (bx-5,by+12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ps.color, 1)
                    continue

                ps._needs_predict = True

            # ── Batch predict ─────────────────────────────────────────
            batch_pids = [pid for pid, ps in persons.items()
                          if getattr(ps, "_needs_predict", False)
                          and len(ps.buf) >= seq_len]

            if batch_pids:
                batch_inp = np.array(
                    [persons[pid].buf[-seq_len:] for pid in batch_pids],
                    dtype=np.float32)
                batch_proba = _predict(model, model_mode, batch_inp)

                for i, pid in enumerate(batch_pids):
                    ps = persons[pid]
                    ps._needs_predict = False
                    lstm_prob = float(batch_proba[i][1])

                    # Rule-based
                    rule_prob = rule_based_score(ps.buf)

                    # Ensemble
                    raw_prob  = ew[0] * lstm_prob + ew[1] * rule_prob

                    # Bayesian smoother
                    smooth_prob = ps.smoother.update(raw_prob)

                    # Scene-aware threshold
                    ps.scene.update(ps.buf[-1])
                    scene_delta   = ps.scene.get_scene_adjustment()
                    dyn_threshold = _update_threshold(float(ps.buf[-1][17])) + scene_delta

                    rec_last_pred_display = max(rec_last_pred_display, smooth_prob)

                    # Visibility-based occlusion suppress
                    ps.update_visibility(ps.buf[-1])
                    vis = ps.vis_score
                    _mean_vis = float(ps.buf[-1][17]) if len(ps.buf[-1]) > 17 else vis
                    _min_vis  = float(ps.buf[-1][18]) if len(ps.buf[-1]) > 18 else vis
                    _vis_gap  = _mean_vis - _min_vis
                    if _vis_gap > 0.20 and vis < 0.6:
                        suppress  = 0.65 + vis * 0.35
                        smooth_prob = smooth_prob * suppress

                    ps.last_prob = smooth_prob

                    if smooth_prob < 0.20 and ps.confirm_cnt == 0:
                        ps.skip_cnt = 2

                    # Decay accumulator
                    accum = ps.update_accum(smooth_prob)

                    # HUD
                    bar_w = int(120 * accum)
                    bx, by = w-145, 60+pid*28
                    bar_c  = (0,0,200) if accum >= dyn_threshold else (40,160,40)
                    cv2.rectangle(frame, (bx,by), (bx+120,by+14), (50,50,50), -1)
                    cv2.rectangle(frame, (bx,by), (bx+bar_w,by+14), bar_c, -1)
                    vis_tag = f" V{vis*100:.0f}%" if vis < 0.7 else ""
                    thr_tag = f" T{dyn_threshold*100:.0f}%"
                    cv2.putText(frame,
                        f"P{pid} {accum*100:.0f}%(A){vis_tag}{thr_tag}",
                        (bx-5,by+12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ps.color, 1)

                    # Near-camera boost threshold
                    _nc = float(ps.buf[-1][24]) if len(ps.buf) > 0 else 0.0
                    _nc_boost = 0.15 if _nc > 0.6 else (0.08 if _nc > 0.4 else 0.0)
                    accum_thr = CONFIG.get("accum_threshold", 0.68) + _nc_boost + scene_delta

                    if accum >= accum_thr:
                        ps.confirm_cnt += 1
                        ps.validator.update(ps.buf[-1])
                    else:
                        ps.confirm_cnt = 0
                        if accum < accum_thr * 0.5:
                            ps.validator.reset()

                    if ps.confirm_cnt >= confirm_need:
                        ps.post_mode          = True
                        ps.post_frames        = 0
                        ps.fall_detected_time = time.time()
                        ps.recovery_nose_buf  = []
                        ps.smoother.reset()  # reset smoother sau khi vào post_mode

        # ── Remove stale persons ───────────────────────────────────────
        for pid in list(persons.keys()):
            if lm_list and pid < len(lm_list): continue
            persons[pid].none_streak += 1
            if persons[pid].none_streak > 30:
                del persons[pid]

        # ── Alert khi fall confirmed ───────────────────────────────────
        if any_fall:
            now = time.time()
            debounce = CONFIG.get("alert_debounce_sec", 1.0)
            if now - last_alert_t >= debounce:
                last_alert_t = now
                ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n  *** FALL DETECTED P{fall_pid} [{ts_str}] ***\n")
                log_fp.write(f"{ts_str},{fall_pid},"
                             f"{fall_ps.last_prob*100 if fall_ps else 0:.1f}%,"
                             f"ensemble,FALL,confirmed\n")
                log_fp.flush()

                # Telegram
                msg = (f"<b>⚠ FALL DETECTED</b>\n"
                       f"Person {fall_pid} | Conf: {fall_ps.last_prob*100:.0f}%\n"
                       f"Time: {ts_str}")
                _telegram_send(msg, image_bgr=fall_frame_snapshot)

                # Visual alert
                cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 6)
                cv2.putText(frame, "!!! FALL DETECTED !!!",
                    (w//2-160, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        # ── Recorder UI ────────────────────────────────────────────────
        if rec_state == "pre_record":
            elapsed   = time.time() - rec_start_t
            countdown = max(0, rec_duration - elapsed)
            _draw_recorder_menu(frame, rec_state, countdown, rec_label_key,
                                rec_last_pred_display)
            if elapsed >= rec_duration:
                rec_state = "waiting_label"

        elif rec_state == "waiting_label":
            _draw_recorder_menu(frame, "pre_record", 0.0, rec_label_key,
                                rec_last_pred_display)

        elif rec_state in ("saved", "cancelled"):
            _draw_recorder_menu(frame, rec_state, 0, rec_label_key, rec_last_pred_display)
            if time.time() - rec_start_t > 1.5:
                rec_state = "idle"

        # ── Status bar ────────────────────────────────────────────────
        cv2.putText(frame, f"v8  F:{frame_count}  P:{len(persons)}",
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

        # ── Display ───────────────────────────────────────────────────
        # Resize frame (đã có overlay) lên display_size bằng INTER_LINEAR để sắc nét hơn
        if (display_w, display_h) != (frame.shape[1], frame.shape[0]):
            disp = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
        else:
            disp = frame
        cv2.imshow("Fall Detection v8", disp)

        # ── Keyboard ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r') and rec_state == "idle":
            rec_state   = "pre_record"
            rec_start_t = time.time()
            rec_buffer.clear()
            ok("Recording started...")

        elif key == 27:  # ESC
            if rec_state in ("pre_record", "waiting_label"):
                rec_state = "cancelled"; rec_start_t = time.time()

        elif key in RECORDER_LABELS and rec_state == "waiting_label":
            rec_label_key = key
            ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
            _save_recorded_clip(rec_buffer.get_all(), key, cap_fps,
                                rec_last_pred_display, ts_str)
            rec_state = "saved"; rec_start_t = time.time()

        elif key == ord('s'):
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            fname  = f"screenshot_{ts_str}.jpg"
            cv2.imwrite(fname, frame)
            ok(f"Screenshot: {fname}")

        elif key == ord('h'):
            print("\n  Keys: Q=quit, R=record, S=screenshot, ESC=cancel, 1/2/3=label")

    log_fp.close()
    cap.release()
    cv2.destroyAllWindows()
    if hasattr(detector, 'close'): detector.close()
    ok("Realtime stopped.")


# ======================================================================
#  MAIN
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Fall Detection v8")
    parser.add_argument("--extract-only",   action="store_true")
    parser.add_argument("--train-only",     action="store_true")
    parser.add_argument("--eval-only",      action="store_true")
    parser.add_argument("--eval-real",      action="store_true")
    parser.add_argument("--retrain",        action="store_true")
    parser.add_argument("--update-data",    action="store_true",
                        help="Incremental learning với U_D_xxxx mới")
    parser.add_argument("--extract-new",    action="store_true",
                        help="Chỉ extract U_D_xxxx mới, không fine-tune")
    parser.add_argument("--mine",           action="store_true",
                        help="Hard negative mining")
    parser.add_argument("--finetune-real",  action="store_true",
                        help="Fine-tune với video real_test_v8")
    parser.add_argument("--video",          type=str, default=None,
                        help="Video file path (None = webcam)")
    parser.add_argument("--arch",           type=str,
                        choices=["ms_tcn","tcn","bilstm"], default=None,
                        help="Override model architecture")
    args = parser.parse_args()

    if args.arch:
        CONFIG["model_arch"] = args.arch
        step(f"Model arch override: {args.arch}")

    model      = None
    model_mode = "keras"

    if args.update_data:
        X_new, y_new = run_extract_incremental()
        if X_new is not None:
            model, model_mode = _load_model()
            model = run_finetune_incremental(model)
            p = CONFIG["processed_dir"]
            if os.path.exists(f"{p}/X_test.npy"):
                run_evaluate(model=model)
        else:
            step("Không có data mới. Model giữ nguyên.")
        return

    if args.extract_new:
        run_extract_incremental(); return

    if args.mine:
        run_hard_negative_mining(); return

    if args.finetune_real:
        model = run_finetune_real()
        if model is None or CONFIG["use_tflite"]:
            model, model_mode = _load_model()
        run_realtime(model=model, model_mode=model_mode, video_source=args.video)
        return

    # Kiểm tra model hiện có
    model_exists  = os.path.exists(CONFIG["model_path"])
    features_ok   = False

    if model_exists and not args.retrain:
        step(f"Found model: {CONFIG['model_path']}")
        try:
            import tensorflow as tf
            _m   = _load_keras_model(CONFIG["model_path"])
            # Model có thể có 1 hoặc 2 outputs (ms_tcn vs tcn)
            inp_shape = _m.input_shape
            exp_f = inp_shape[-1]; exp_s = inp_shape[1]
            if exp_f != N_FEATURES or exp_s != CONFIG["sequence_length"]:
                warn(f"Model shape ({exp_s},{exp_f}) != expected "
                     f"({CONFIG['sequence_length']},{N_FEATURES}). Train lại.")
                del _m
            else:
                model = _m; features_ok = True
                step("Model tương thích.")
        except Exception as e:
            warn(f"Load failed: {e}. Train lại.")

    need_train = args.retrain or not model_exists or not features_ok

    if args.eval_only or args.eval_real:
        run_evaluate(model=model, eval_real_test=args.eval_real); return

    if args.extract_only:
        run_extract(); ok("Extract done."); return

    if args.train_only:
        p = CONFIG["processed_dir"]
        data_exists = (os.path.exists(f"{p}/sequences_aug.npy")
                       or os.path.exists(f"{p}/sequences.npy"))
        if not data_exists: run_extract()
        model = run_train(); run_evaluate(model=model); return

    if need_train:
        step("Full pipeline: extract → train → evaluate → realtime")
        p = CONFIG["processed_dir"]
        data_exists = (os.path.exists(f"{p}/sequences_aug.npy")
                       or os.path.exists(f"{p}/sequences.npy"))
        if not data_exists or args.retrain:
            run_extract()
        else:
            step("Data có sẵn, bỏ qua extract.")
        model = run_train()

    p = CONFIG["processed_dir"]
    if os.path.exists(f"{p}/X_test.npy"):
        model = run_evaluate(model=model)
    else:
        warn("Chưa có test set. Bỏ qua evaluate.")

    # Auto tune sau train
    if model is not None and os.path.exists(f"{p}/X_val.npy"):
        try:
            _Xv = np.load(f"{p}/X_val.npy")
            _yv = np.load(f"{p}/Y_val.npy")
            _auto_tune_inference_params(model, _Xv, _yv)
        except Exception as _ate:
            warn(f"Auto tune failed: {_ate}")

    # Auto fine-tune nếu có video trong real_test_v8
    _real_dir = CONFIG["real_test_dir"]
    _VEXT_RT  = (".mp4",".avi",".mov",".mkv",".wmv",".flv",".m4v",".3gp")

    def _has_videos(folder):
        if not os.path.isdir(folder): return False
        return any(f.lower().endswith(_VEXT_RT) for f in os.listdir(folder))

    _has_fall    = _has_videos(os.path.join(_real_dir, "fall"))
    _has_notfall = _has_videos(os.path.join(_real_dir, "not_fall"))
    _has_action  = _has_videos(os.path.join(_real_dir, "action_like_fall"))
    _has_any     = _has_fall or _has_notfall or _has_action

    if _has_any and not need_train and os.path.exists(CONFIG["model_path"]):
        step(f"Phát hiện video trong {_real_dir}/ → tự động fine-tune...")
        ft_model = run_finetune_real()
        if ft_model is not None:
            model = ft_model
        ok("Auto fine-tune hoàn tất.")

        import shutil as _ft_sh, re as _ft_re
        _backup_dir = os.path.join(_real_dir, "used_backup")
        _action_dir = os.path.join(_real_dir, "action_like_fall")
        _real_nfall = os.path.join(_real_dir, "not_fall")

        def _move_folder(sub):
            _s = os.path.join(_real_dir, sub)
            _d = os.path.join(_backup_dir, sub)
            if not os.path.isdir(_s): return 0
            _vs = [f for f in os.listdir(_s) if f.lower().endswith(_VEXT_RT)]
            if not _vs: return 0
            os.makedirs(_d, exist_ok=True)
            for _v in _vs:
                _ft_sh.move(os.path.join(_s, _v), os.path.join(_d, _v))
            ok(f"Moved {len(_vs)} videos: {sub}/ → used_backup/{sub}/")
            return len(_vs)

        _move_folder("fall")
        _move_folder("action_like_fall")

        _has_nfall_v = (os.path.isdir(_real_nfall)
                        and any(f.lower().endswith(_VEXT_RT)
                                for f in os.listdir(_real_nfall)))
        if _has_nfall_v and model is not None:
            step("Mine real_test/not_fall/ → action_like_fall/ mới...")
            try:
                _n_new = run_hard_negative_mining(
                    model=model, source_dir=_real_nfall, output_dir=_action_dir)
                if _n_new > 0:
                    ok(f"Mined {_n_new} clips → action_like_fall/")
            except Exception as _me:
                warn(f"Mining lỗi: {_me}")

        _move_folder("not_fall")
        step("action_like_fall/ sẵn sàng cho lần fine-tune tiếp theo.")

    elif _has_any and need_train:
        warn(f"Có video trong {_real_dir}/ nhưng model vừa train lại từ đầu.")
        warn("Fine-tune real sẽ chạy ở lần sau khi model ổn định.")

    if model is None or CONFIG["use_tflite"]:
        model, model_mode = _load_model()

    run_realtime(model=model, model_mode=model_mode, video_source=args.video)


if __name__ == "__main__":
    main()
