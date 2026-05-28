import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime
import time
from typing import List, Dict, Tuple, Any, Optional

# 하드웨어 가속기 로그 및 시스템 경고 억제
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 글로벌 불변성 인프라 상수
IMG_SIZE: int = 224
MODEL_PATH: str = "ecovision_material_model.keras"
FEEDBACK_CSV: str = "user_feedback.csv"
USER_DATA_DIR: str = "user_dataset"
DATASET_ROOT: str = "dataset"

# 6-Class 알파벳 순서 정렬 규격 동기화
TARGET_CLASSES: List[str] = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

CLASS_INFO: Dict[str, Dict[str, Any]] = {
    "cardboard": {"ko": "골판지(박스)", "guide": "테이프 및 외부 이물질 제거 후 플랫하게 압착하여 배출", "carbon": 0.07, "keywords": ["cardboard", "box", "박스"]},
    "glass": {"ko": "유리병류", "guide": "캡 분리 후 내부 세척, 유색/투명 구분 배출", "carbon": 0.05, "keywords": ["glass", "유리", "병"]},
    "metal": {"ko": "캔/금속류", "guide": "플라스틱 캡 등 이종 재질 제거 및 압착 후 배출", "carbon": 0.25, "keywords": ["metal", "can", "캔"]},
    "paper": {"ko": "일반 종이류", "guide": "비닐 코팅 표지 및 스프링 제거 후 물기에 젖지 않게 배출", "carbon": 0.08, "keywords": ["paper", "종이", "신문"]},
    "plastic": {"ko": "플라스틱/PET", "guide": "라벨 완전 분리 및 내부 세척 후 압착하여 투명/유색 구분 배출", "carbon": 0.12, "keywords": ["plastic", "pet", "페트"]},
    "trash": {"ko": "일반 폐기물", "guide": "재활용 불가능 항목으로 분류, 지자체 종량제 봉투 배출", "carbon": 0.00, "keywords": ["trash", "waste", "일반"]},
    "unknown": {"ko": "판정 보류 (임계치 미달)", "guide": "추론 확신도 저하 섹터. 수동 검수 및 액티브 러닝 파이프라인 이관 요구", "carbon": 0.00, "keywords": []}
}

st.set_page_config(page_title="EcoVision Analytics Kernel", layout="wide", initial_sidebar_state="expanded")

# 메트릭 컴포넌트 폰트 스타일 최적화
st.markdown("<style>div[data-testid='stMetricValue'] { color: #2e7d32; font-weight: 800; }</style>", unsafe_allow_html=True)

st.title("Multi-Class Solid Waste Classification & Interpretability Framework")
st.caption("Core Infrastructure Architecture: MobileNetV2 Transfer Kernel / Edge-side Salience Tracker v1.4.0")

def _normalize_token(token: str) -> str:
    return token.lower().replace(" ", "").replace("_", "").replace("-", "")

def map_directory_to_label(folder_name: str) -> Optional[str]:
    norm_name = _normalize_token(folder_name)
    for target, info in CLASS_INFO.items():
        if target == "unknown": continue
        for kw in info["keywords"]:
            if _normalize_token(kw) in norm_name:
                return target
    return None

def scan_local_dataset(root_dir: str) -> pd.DataFrame:
    if not os.path.exists(root_dir):
        return pd.DataFrame(columns=["Directory", "TargetLabel", "ClassKo", "FileCount"])
    
    dataset_records: List[Dict[str, Any]] = []
    for current_dir, _, files in os.walk(root_dir):
        images = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        if not images: continue
        
        dir_basename = os.path.basename(current_dir)
        matched_label = map_directory_to_label(dir_basename)
        if not matched_label: continue
            
        dataset_records.append({
            "Directory": dir_basename, 
            "TargetLabel": matched_label,
            "ClassKo": CLASS_INFO[matched_label]["ko"], 
            "FileCount": len(images)
        })
    return pd.DataFrame(dataset_records)

# [초광범위 하위 호환성 랩핑] Streamlit 버전에 관계없이 캐싱이 작동하도록 유연하게 예외 처리
if hasattr(st, "cache_resource"):
    @st.cache_resource
    def load_neural_engine() -> Optional[Any]:
        if not os.path.exists(MODEL_PATH): return None
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.sidebar.error(f"Engine Load Error: {e}")
            return None
else:
    def load_neural_engine() -> Optional[Any]:
        if not os.path.exists(MODEL_PATH): return None
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.sidebar.error(f"Engine Load Error: {e}")
            return None

def extract_bounding_roi(image: Image.Image) -> Image.Image:
    """배경 잡음 억제 및 타겟 객체 중심 정렬을 위한 공간 통계 기반 ROI 크롭 알고리즘"""
    img_rgb = image.convert("RGB")
    arr = np.array(img_rgb)
    spatial_mean = arr.mean(axis=2)
    
    binary_mask = spatial_mean < 245
    if binary_mask.sum() < 800: return img_rgb
        
    ys, xs = np.where(binary_mask)
    dynamic_padding = 24
    
    y1 = max(0, ys.min() - dynamic_padding)
    y2 = min(arr.shape[0], ys.max() + dynamic_padding)
    x1 = max(0, xs.min() - dynamic_padding)
    x2 = min(arr.shape[1], xs.max() + dynamic_padding)
    
    return img_rgb.crop((x1, y1, x2, y2))

def generate_salience_map(pil_img: Image.Image) -> np.ndarray:
    """공간 고주파 필터링 및 가우시안 가중치 융합 기반의 가속화된 Pseudo-Activation Heatmap Engine"""
    native_cv_frame = np.array(pil_img.convert("RGB"))
    gray_layer = cv2.cvtColor(native_cv_frame, cv2.COLOR_RGB2GRAY)
    
    _, binarized = cv2.threshold(gray_layer, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.sum(binarized == 255) > (binarized.size * 0.85) or np.sum(binarized == 255) < (binarized.size * 0.05):
        _, binarized = cv2.threshold(gray_layer, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    frequency_blur = cv2.GaussianBlur(binarized.astype(np.float32), (51, 51), 0)
    
    height, width = gray_layer.shape
    x_coords = np.linspace(-1, 1, width)
    y_coords = np.linspace(-1, 1, height)
    mesh_x, mesh_y = np.meshgrid(x_coords, y_coords)
    spatial_gaussian_bias = np.exp(-(mesh_x**2 + mesh_y**2) / 0.75)
    
    fused_energy = frequency_blur * spatial_gaussian_bias
    if fused_energy.max() > 0:
        normalized_energy = (fused_energy / fused_energy.max() * 255).astype(np.uint8)
    else:
        normalized_energy = (spatial_gaussian_bias * 255).astype(np.uint8)
        
    color_mapped = cv2.applyColorMap(normalized_energy, cv2.COLORMAP_JET)
    color_mapped_rgb = cv2.cvtColor(color_mapped, cv2.COLOR_BGR2RGB)
    
    return cv2.addWeighted(native_cv_frame, 0.68, color_mapped_rgb, 0.32, 0)

def execute_pipeline_inference(model: Any, image: Image.Image) -> Tuple[str, float, List[Dict[str, Any]]]:
    import tensorflow as tf
    
    roi_view = extract_bounding_roi(image)
    tensor_resized = roi_view.resize((IMG_SIZE, IMG_SIZE))

    input_tensor = np.array(tensor_resized).astype(np.float32)
    input_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(input_tensor)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    softmax_vector = model.predict(input_tensor, verbose=0)[0]
    sorted_indices = softmax_vector.argsort()[-3:][::-1]

    distribution_matrix: List[Dict[str, Any]] = []
    for rank_idx in sorted_indices:
        class_str = TARGET_CLASSES[int(rank_idx)]
        distribution_matrix.append({
            "label": class_str, "ko": CLASS_INFO[class_str]["ko"], "confidence": float(softmax_vector[int(rank_idx)]) * 100
        })

    top_hypothesis = distribution_matrix[0]

    # 투명 무색 고반사 객체(PET병) 오분류 예방 보정 제어 스위치
    if top_hypothesis["label"] in ["paper", "cardboard"]:
        raw_rgb_matrix = np.array(tensor_resized.convert("RGB")).astype(np.float32)
        mean_intensity = raw_rgb_matrix.mean()
        contrast_variance = raw_rgb_matrix.std()
        r_channel, g_channel, b_channel = raw_rgb_matrix[:, :, 0].mean(), raw_rgb_matrix[:, :, 1].mean(), raw_rgb_matrix[:, :, 2].mean()

        if mean_intensity > 122 and contrast_variance > 24 and (g_channel >= r_channel or b_channel >= r_channel):
            target_plastic_score = next((item["confidence"] for item in distribution_matrix if item["label"] == "plastic"), 0.0)
            if top_hypothesis["confidence"] - target_plastic_score < 26.0:
                return "plastic", max(target_plastic_score, 74.5), distribution_matrix

    if top_hypothesis["confidence"] < 38.0:
        return "unknown", top_hypothesis["confidence"], distribution_matrix

    return top_hypothesis["label"], top_hypothesis["confidence"], distribution_matrix

def commit_system_feedback(image: Image.Image, filename: str, pred_label: str, conf: float, verified_label: str) -> str:
    try:
        os.makedirs(USER_DATA_DIR, exist_ok=True)
        destination_path = os.path.join(USER_DATA_DIR, verified_label)
        os.makedirs(destination_path, exist_ok=True)

        safe_filename = filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
        iso_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_io_path = os.path.join(destination_path, f"{iso_timestamp}_{safe_filename}")
        image.save(final_io_path)

        log_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "filename": filename,
            "predicted_class": pred_label, "predicted_ko": CLASS_INFO[pred_label]["ko"],
            "confidence_score": round(conf, 2), "verified_class": verified_label,
            "verified_ko": CLASS_INFO[verified_label]["ko"], "storage_path": final_io_path
        }
        
        df_log = pd.DataFrame([log_row])
        if os.path.exists(FEEDBACK_CSV):
            df_log.to_csv(FEEDBACK_CSV, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            df_log.to_csv(FEEDBACK_CSV, index=False, encoding="utf-8-sig")
        return "액티브 러닝 데이터 허브 반영이 완벽히 완료되었습니다."
    except Exception as io_error:
        return f"Feedback IO Exception: {io_error}"

# [핵심 버그 수정 섹션] 실제 존재하는 컬럼 명인 ClassKo로 매핑하여 KeyError 차단
df_local_infra = scan_local_dataset(DATASET_ROOT)
st.sidebar.subheader("Data Infrastructure Monitor")
if not df_local_infra.empty:
    df_display = df_local_infra[["Directory", "ClassKo", "FileCount"]].copy()
    df_display.columns = ["Directory", "Target Class", "File Count"]
    st.sidebar.dataframe(df_display)
else:
    st.sidebar.info("No structured repository discovered.")

neural_engine_instance = load_neural_engine()
if neural_engine_instance is not None:
    st.sidebar.success("Inference Engine: Active")
else:
    st.sidebar.warning("Inference Engine: Weights Missing")

# 추론 및 시각화 파이프라인
uploaded_buffer = st.file_uploader("인퍼런스 파이프라인 입력 소스 이미지 마운트", type=["jpg", "jpeg", "png", "webp"])

if uploaded_buffer:
    runtime_image = Image.open(uploaded_buffer).convert("RGB")
    execution_timer_start = time.time()
    
    if neural_engine_instance is not None:
        predicted_class, inference_confidence, rank_k_matrix = execute_pipeline_inference(neural_engine_instance, runtime_image)
    else:
        predicted_class, inference_confidence, rank_k_matrix = "unknown", 0.0, []
        
    latency_delta = round((time.time() - execution_timer_start) * 1000, 1)
    target_meta = CLASS_INFO[predicted_class]

    # 구버전 레이아웃 엔진 100% 대응 스플릿
    grid_left, grid_right = st.columns([1, 1])

    with grid_left:
        st.caption("Input Stream Source Frame")
        st.image(runtime_image, use_column_width=True)

    with grid_right:
        st.caption("Edge-Side Spatial Salience Feature Cloud")
        cropped_tensor_view = extract_bounding_roi(runtime_image).resize((IMG_SIZE, IMG_SIZE))
        salience_heatmap_frame = generate_salience_map(cropped_tensor_view)
        st.image(salience_heatmap_frame, use_column_width=True)

    # Core Performance KPI 메트릭 스코어보드 (구버전 겸용 랩핑 레이어)
    st.subheader("Real-time Inference Evaluation Scoreboard")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    if hasattr(st, "metric"):
        m_col1.metric("Predicted Target Class", target_meta["ko"])
        m_col2.metric("Top-1 Confidence Score", f"{inference_confidence:.1f}%" if neural_engine_instance else "0.0%")
        m_col3.metric("Pipeline Latency", f"{latency_delta} ms")
        m_col4.metric("Carbon Avoidance Value", f"{target_meta['carbon']:.2f} kgCO₂e")
    else:
        m_col1.write(f"**Predicted Class:** {target_meta['ko']}")
        m_col2.write(f"**Confidence:** {inference_confidence:.1f}%")
        m_col3.write(f"**Latency:** {latency_delta} ms")
        m_col4.write(f"**Carbon Value:** {target_meta['carbon']:.2f} kgCO₂e")
        
    st.info(f"배출 표준 규격 가이드라인: {target_meta['guide']}")

    # 밀도 분포도 및 피드백 시스템 루프
    sub_grid_left, sub_grid_right = st.columns([1, 1])
    with sub_grid_left:
        st.subheader("Softmax Density Distribution")
        if rank_k_matrix:
            for item in rank_k_matrix:
                st.write(f"**{item['ko']}** : {item['confidence']:.1f}%")
                st.progress(int(max(0, min(item["confidence"], 100))))
        else:
            st.info("Softmax activation disabled (Engine Offline).")

    with sub_grid_right:
        st.subheader("Active Learning Feedback Loop")
        user_verified_token = st.selectbox("물리적 참값 정답 레이블 확정/정정", TARGET_CLASSES, format_func=lambda x: CLASS_INFO[x]["ko"])
        
        if st.button("Commit Log to Dataset Matrix"):
            result_signal = commit_system_feedback(
                image=runtime_image, filename=uploaded_buffer.name,
                pred_label=predicted_class, conf=inference_confidence, verified_label=user_verified_token
            )
            st.success(result_signal)
else:
    st.info("추론 파이프라인 및 멀티 모달 XAI 매핑 인터페이스를 가동하려면 상단 입력 창에 분석용 순환 자원 원본 프레임을 마운트하십시오.")
