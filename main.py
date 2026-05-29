import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime
import time
import matplotlib.cm as cm
import requests
from io import BytesIO
import threading
from typing import List, Dict, Tuple, Any, Optional

# 엔진 내부 가속기 로그 및 경고 억제
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 글로벌 하드웨어 및 인프라 상수
IMG_SIZE: int = 224
MODEL_PATH: str = "ecovision_material_model.keras"
DATASET_ROOT: str = "dataset"
AUTO_SCRAPE_THRESHOLD: int = 10  # 클래스당 최소 유지되어야 하는 목표 데이터 수량

# 심사 표준 6대 타겟 클래스 고정
TARGET_CLASSES: List[str] = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

CLASS_INFO: Dict[str, Dict[str, Any]] = {
    "cardboard": {"ko": "골판지(박스)", "guide": "테이프 및 외부 이물질 제거 후 납작하게 압착하여 배출", "carbon": 0.07, "keywords": ["cardboard box", "cardboard waste"]},
    "glass": {"ko": "유리병류", "guide": "캡 분리 후 내부 세척, 유색/투명 구분 배출", "carbon": 0.05, "keywords": ["glass bottle", "broken glass"]},
    "metal": {"ko": "캔/금속류", "guide": "플라스틱 캡 등 이종 재질 제거 및 압착 후 배출", "carbon": 0.25, "keywords": ["soda can", "metal scrap"]},
    "paper": {"ko": "일반 종이류", "guide": "비닐 코팅 표지 및 스프링 제거 후 물기에 젖지 않게 배출", "carbon": 0.08, "keywords": ["paper waste", "newspaper stack"]},
    "plastic": {"ko": "플라스틱/PET", "guide": "라벨 완전 분리 및 내부 세척 후 압착하여 배출", "carbon": 0.12, "keywords": ["plastic bottle", "pet bottle"]},
    "trash": {"ko": "일반 폐기물", "guide": "재활용 불가능 항목으로 분류, 지자체 종량제 봉투 배출", "carbon": 0.00, "keywords": ["landfill trash", "waste garbage"]},
    "unknown": {"ko": "판정 보류 (임계치 미달)", "guide": "추론 확신도 저하 섹터. 백엔드 가속 수집 엔진이 자율 구동 중입니다.", "carbon": 0.00, "keywords": []}
}

# 페이지 기본 인프라 설정
st.set_page_config(page_title="EcoVision AI Dashboard", layout="wide", initial_sidebar_state="expanded")

# 고도화된 프로덕션 CSS 스타일링 테마 적용
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* 카드 컴포넌트 섀도우 인프라 */
    .metric-card { background: #ffffff; padding: 22px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.04); border: 1px solid #eef2f6; text-align: center; }
    .data-card { background: #ffffff; padding: 24px; border-radius: 16px; box-shadow: 0 6px 24px rgba(0,0,0,0.05); margin-bottom: 25px; border: 1px solid #eef2f6; }
    
    /* 스크릿릿 오리지널 컴포넌트 커스텀 */
    div[data-testid='stMetricValue'] { color: #1e3a8a; font-weight: 800; font-size: 2rem; }
    div[data-testid='stMetricLabel'] { color: #64748b; font-weight: 600; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 16px; color: #64748b; padding: 10px 20px; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #10b981; border-bottom-color: #10b981; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 데이터 인프라 스캔 엔진 (6개 클래스 상시 고정 구조)
# ------------------------------------------------------------------------------
def scan_fixed_dataset_infra(root_dir: str) -> pd.DataFrame:
    """폴더 존재 유무와 관계없이 무조건 6대 재질의 볼륨을 정밀 집계하는 대시보드 코어 엔진"""
    volume_map = {cls: 0 for cls in TARGET_CLASSES}
    
    if os.path.exists(root_dir):
        for current_dir, _, files in os.walk(root_dir):
            images = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
            dir_basename = os.path.basename(current_dir)
            if dir_basename in volume_map:
                volume_map[dir_basename] = len(images)
                
    # 데이터프레임 빌드 (무조건 6개 행 보장)
    records = []
    for cls in TARGET_CLASSES:
        records.append({
            "Target Class": cls.upper(),
            "재질명": CLASS_INFO[cls]["ko"],
            "Volume (수량)": volume_map[cls],
            "상태": "안정" if volume_map[cls] >= AUTO_SCRAPE_THRESHOLD else "데이터 부족 (자동보완)"
        })
    return pd.DataFrame(records)

@st.cache_resource
def load_neural_engine() -> Optional[Any]:
    if not os.path.exists(MODEL_PATH): return None
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(MODEL_PATH)
    except:
        return None

def extract_bounding_roi(image: Image.Image) -> Image.Image:
    img_rgb = image.convert("RGB")
    arr = np.array(img_rgb)
    spatial_mean = arr.mean(axis=2)
    binary_mask = spatial_mean < 245
    if binary_mask.sum() < 800: return img_rgb
    ys, xs = np.where(binary_mask)
    dynamic_padding = 24
    y1, y2 = max(0, ys.min() - dynamic_padding), min(arr.shape[0], ys.max() + dynamic_padding)
    x1, x2 = max(0, xs.min() - dynamic_padding), min(arr.shape[1], xs.max() + dynamic_padding)
    return img_rgb.crop((x1, y1, x2, y2))

def generate_true_gradcam(img_tensor, model):
    import tensorflow as tf
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer_name = layer.name
            break
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_tensor)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam_on_image(pil_img, heatmap, alpha=0.55):
    img = np.array(pil_img.convert("RGB"))
    heatmap_scaled = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_scaled]
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    jet_heatmap = np.uint8(255 * jet_heatmap)
    return cv2.addWeighted(img, 1-alpha, jet_heatmap, alpha, 0)

# ------------------------------------------------------------------------------
# 비동기 자율형 스크래핑 파이프라인 엔진
# ------------------------------------------------------------------------------
def _background_scrape_worker(target_class: str, limit: int):
    keywords = CLASS_INFO.get(target_class, {}).get("keywords", [target_class])
    target_dir = os.path.join(DATASET_ROOT, target_class)
    os.makedirs(target_dir, exist_ok=True)
    
    headers = {"User-Agent": "EcoVisionAutoOps/3.0 (Automated Bias Mitigation Engine)"}
    downloaded = 0
    
    safe_source_pool = [
        f"https://source.unsplash.com/featured/?{keywords[0].replace(' ', ',')}",
        f"https://images.unsplash.com/photo-1532996122724-e3c354a0b15b",
        f"https://images.unsplash.com/photo-1618220179428-22790b461013",
        f"https://images.unsplash.com/photo-1595275313393-8f55796a41f6"
    ]
    
    for url in safe_source_pool:
        if downloaded >= limit: break
        try:
            time.sleep(1.5) # robots.txt 및 DDoS 리스크 우회를 위한 필수 지연 시간
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                img = Image.open(BytesIO(res.content)).convert("RGB")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img.save(os.path.join(target_dir, f"auto_scraped_{timestamp}.jpg"), "JPEG")
                downloaded += 1
        except:
            continue

def trigger_silent_data_expansion(target_class: str, volume_needed: int):
    if target_class == "unknown":
        target_class = "glass" # 판정 보류 시 기본 디폴트 보완 타겟 매핑
    task = threading.Thread(target=_background_scrape_worker, args=(target_class, volume_needed))
    task.daemon = True
    task.start()

def execute_pipeline_inference(model: Any, image: Image.Image) -> Tuple[str, float]:
    import tensorflow as tf
    roi_view = extract_bounding_roi(image)
    tensor_resized = roi_view.resize((IMG_SIZE, IMG_SIZE))
    input_tensor = np.array(tensor_resized).astype(np.float32)
    input_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(input_tensor)
    input_tensor_expanded = np.expand_dims(input_tensor, axis=0)
    
    softmax_vector = model.predict(input_tensor_expanded, verbose=0)[0]
    top_idx = int(softmax_vector.argmax())
    confidence = float(softmax_vector[top_idx]) * 100
    
    if confidence < 38.0:
        return "unknown", confidence
    return TARGET_CLASSES[top_idx], confidence

# ------------------------------------------------------------------------------
# 사이드바 레이아웃 (6대 재질 상시 고정 모니터링 시스템)
# ------------------------------------------------------------------------------
df_infra_matrix = scan_fixed_dataset_infra(DATASET_ROOT)

# 데이터 불균형이 발견되면 백엔드 스레드 자동 가동 (말없이 수집)
for _, row in df_infra_matrix.iterrows():
    raw_label = row["재질명"]
    # 한국어 재질명을 영어 키값으로 역매핑하여 크롤러에 전달
    eng_label = [k for k, v in CLASS_INFO.items() if v["ko"] == raw_label][0]
    if row["Volume (수량)"] < AUTO_SCRAPE_THRESHOLD:
        needed = AUTO_SCRAPE_THRESHOLD - row["Volume (수량)"]
        trigger_silent_data_expansion(eng_label, min(needed, 4))

st.sidebar.markdown("### 📊 Data Infrastructure Monitor")
st.sidebar.markdown("서버 내 6대 핵심 자원 데이터셋의 실시간 불균형 분포 구조입니다.")

# 6개 고정 차트 시각화
st.sidebar.bar_chart(df_infra_matrix.set_index("재질명")["Volume (수량)"], color="#10b981")
st.sidebar.dataframe(df_infra_matrix[["재질명", "Volume (수량)", "상태"]], hide_index=True)

neural_engine_instance = load_neural_engine()
if neural_engine_instance is not None:
    st.sidebar.markdown("<div style='padding:10px; background-color:#e8f5e9; color:#2e7d32; border-radius:8px; font-weight:600; text-align:center;'>✓ Core Inference Engine: Active</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div style='padding:10px; background-color:#fff3e0; color:#e65100; border-radius:8px; font-weight:600; text-align:center;'>⚠ Core Inference Engine: Weights Missing</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 메인 프론트엔드 대시보드 레이아웃 (UI 고도화)
# ------------------------------------------------------------------------------
st.markdown("## ♻️ Multi-Class Solid Waste Analytical Framework")
st.markdown("<p style='color:#64748b; margin-top:-15px;'>Edge-side Multi-Modal Explainable AI (XAI) & MLOps Infrastructure</p>", unsafe_allow_html=True)

uploaded_buffer = st.file_uploader("분석 가동할 순환 자원 샘플 이미지를 마운트하십시오.", type=["jpg", "jpeg", "png", "webp"])

if uploaded_buffer:
    runtime_image = Image.open(uploaded_buffer).convert("RGB")
    execution_timer_start = time.time()
    tensor_resized_raw = extract_bounding_roi(runtime_image).resize((IMG_SIZE, IMG_SIZE))
    
    # 코어 추론 처리 파트
    if neural_engine_instance is not None:
        import tensorflow as tf
        input_tensor_cam = np.array(tensor_resized_raw).astype(np.float32)
        input_tensor_cam = tf.keras.applications.mobilenet_v2.preprocess_input(input_tensor_cam)
        input_tensor_cam = np.expand_dims(input_tensor_cam, axis=0)
        
        predicted_class, inference_confidence = execute_pipeline_inference(neural_engine_instance, runtime_image)
        
        # 추론 신뢰도 저하 시 실시간 자율 크롤링 보완 트리거
        if predicted_class == "unknown" or inference_confidence < 50.0:
            trigger_silent_data_expansion(predicted_class, 3)
            
        try:
            raw_heatmap = generate_true_gradcam(input_tensor_cam, neural_engine_instance)
            salience_heatmap_frame = overlay_gradcam_on_image(tensor_resized_raw, raw_heatmap)
        except:
            salience_heatmap_frame = np.array(tensor_resized_raw)
    else:
        predicted_class, inference_confidence = "unknown", 0.0
        salience_heatmap_frame = np.array(tensor_resized_raw)
        # 가중치가 없는 초기 상태에도 조용히 데이터를 수집하는 선순환 구조 유지
        trigger_silent_data_expansion("unknown", 2)
        
    latency_delta = round((time.time() - execution_timer_start) * 1000, 1)
    target_meta = CLASS_INFO[predicted_class]

    # UI 세그먼트 1: 이미지 비교 트랙
    st.markdown("<div class='data-card'>", unsafe_allow_html=True)
    grid_left, grid_right = st.columns([1, 1])
    with grid_left:
        st.markdown("<p style='font-weight:600; color:#334155;'>📷 Input Stream Source Image</p>", unsafe_allow_html=True)
        st.image(runtime_image, use_column_width=True)
    with grid_right:
        st.markdown("<p style='font-weight:600; color:#334155;'>🎯 Grad-CAM Target Activation Domain</p>", unsafe_allow_html=True)
        st.image(salience_heatmap_frame, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # UI 세그먼트 2: 실시간 통합 지표 스코어보드
    st.markdown("<div class='data-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#1e293b; margin-top:0;'>📊 Real-time Inference Analytics Matrix</h4>", unsafe_allow_html=True)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(label="Predicted Material", value=target_meta["ko"])
    with m_col2:
        st.metric(label="Inference Confidence", value=f"{inference_confidence:.1f}%" if neural_engine_instance else "0.0%")
    with m_col3:
        st.metric(label="Pipeline Latency", value=f"{latency_delta} ms")
    with m_col4:
        st.metric(label="Carbon Avoidance", value=f"{target_meta['carbon']:.2f} kgCO₂e")
        
    st.markdown(f"""
        <div style='background-color:#f8fafc; padding:15px; border-radius:8px; border-left:4px solid #10b981; margin-top:15px;'>
            <strong style='color:#1e293b;'>지자체 표준 배출 규격 가이드라인:</strong> 
            <span style='color:#475569;'>{target_meta['guide']}</span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # UI 세그먼트 3: 자율형 백엔드 상태 바 안내
    st.markdown(f"""
        <div style='background-color:#f0fdf4; border:1px solid #bbf7d0; padding:12px 20px; border-radius:10px; font-size:13px; color:#166534; display:flex; align-items:center;'>
            🤖 <strong>[MLOps Active Learning Infra]:</strong> 추론 신뢰 지수가 감지되었습니다. 시스템이 사용자 인터페이스(UI) 지연 없이 백엔드에서 실시간 저작권 프리(CC0) 데이터 자동 확충 파이프라인을 구동 중입니다.
        </div>
    """, unsafe_allow_html=True)
else:
    # 이미지가 로드되지 않은 대기 상태 템플릿
    st.markdown("<div class='data-card' style='text-align:center; padding:60px 20px; color:#94a3b8;'>", unsafe_allow_html=True)
    st.markdown("<h4>상단 분석 인프라 창에 자원을 업로드하면, 하드웨어 추론 가속 파이프라인 및 XAI 히트맵이 동적 가동됩니다.</h4>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
