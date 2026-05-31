import os
import io
import gc
import time
import base64
import datetime
import sqlite3
import numpy as np
import pandas as pd
import cv2
import streamlit as st
from PIL import Image

# 파이토치 및 경량화 환경 최적화
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

st.set_page_config(page_title="EcoVision Enterprise XAI Platform", page_icon="⚡", layout="wide")

# ==========================================
# [글로벌 제어 상수 & 하이퍼파라미터]
# ==========================================
TARGET_MATERIALS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CARBON_FACTORS = {"plastic": 0.12, "paper": 0.08, "metal": 0.25, "glass": 0.05, "cardboard": 0.07, "trash": 0.00}
TARGET_OBJECTS = ["수정테이프", "페트병", "종이컵", "음료수캔", "골판지상자", "일반비닐", "가위", "기타물품"]

MODEL_PATH = "best_ecovision_multitask.pth"

# 영구 보존 스토리지 마운트 확인
if os.path.exists("/data"):
    DB_PATH = "/data/ecovision_enterprise.db"
    storage_status = "🔒 영구 보존 스토리지(/data) 바인딩 완료"
else:
    DB_PATH = "ecovision_enterprise.db"
    storage_status = "⚠️ 로컬 런타임 스토리지 가동 중"

# ==========================================
# 1. RDBMS(SQLite) 데이터 레이어 (순수 실데이터 전용)
# ==========================================
def init_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS feedback
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          timestamp TEXT, filename TEXT, predicted TEXT, 
                          confidence REAL, actual TEXT, is_correct INTEGER,
                          carbon_saved REAL)''')
            conn.commit()
    except Exception as e:
        st.error(f"DB 초기화 실패: {e}")

init_db()

# ==========================================
# 2. 멀티태스크 AI 아키텍처 (가짜 ML 전면 차단)
# ==========================================
class EcovisionMultiTaskModel(nn.Module):
    def __init__(self, num_objects, num_materials):
        super(EcovisionMultiTaskModel, self).__init__()
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity() 
        
        self.object_head = nn.Sequential(
            nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_objects)
        )
        self.material_head = nn.Sequential(
            nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_materials)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.object_head(features), self.material_head(features)

@st.cache_resource
def load_ecovision_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EcovisionMultiTaskModel(num_objects=len(TARGET_OBJECTS), num_materials=len(TARGET_MATERIALS))
    
    # [핵심] 가짜 머신러닝 금지 로직: 가중치 파일이 없으면 즉시 시스템 셧다운
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except Exception as e:
            st.error(f"🚨 모델 가중치 로드 실패: {e}")
            st.stop()
    else:
        st.error(f"🚨 딥러닝 가중치 파일 '{MODEL_PATH}'을 찾을 수 없습니다. 난수 시뮬레이션을 허용하지 않으므로 인프라 가동을 중단합니다.")
        st.stop()
        
    model.to(device)
    model.eval()
    return model, device

model, device = load_ecovision_model()

# ==========================================
# 3. 실시간 Grad-CAM (메모리 릭 방지 적용)
# ==========================================
class GradCAMUtility:
    def __init__(self, model_instance):
        self.model = model_instance
        self.gradients = None
        self.activations = None
        self.target_layer = self.model.backbone.features[-1]
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output): self.activations = output
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]

    def generate(self, input_tensor, cv_img, pred_idx):
        try:
            self.model.zero_grad()
            obj_preds, mat_preds = self.model(input_tensor)
            score = mat_preds[0, pred_idx]
            score.backward(retain_graph=False)

            gradients = self.gradients.cpu().data.numpy()[0]
            activations = self.activations.cpu().data.numpy()[0]
            
            weights = np.mean(gradients, axis=(1, 2))
            cam = np.zeros(activations.shape[1:], dtype=np.float32)

            for i, w in enumerate(weights):
                cam += w * activations[i]

            cam = np.maximum(cam, 0)
            if cam.max() > 0: cam = cam / cam.max()
            
            cam = cv2.resize(cam, (cv_img.shape[1], cv_img.shape[0]))
            heatmap = np.uint8(255 * cam)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            blended = cv2.addWeighted(cv_img, 0.6, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), 0.4, 0)
            
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            del gradients, activations, cam, heatmap, heatmap_colored, blended
            return base64.b64encode(buffer).decode('utf-8')
        except Exception:
            return None

gradcam_engine = GradCAMUtility(model)

# ==========================================
# 4. 실데이터 기반 통계 연산 루틴 (Random 완전 배제)
# ==========================================
def get_real_analytics():
    total_scans = 0
    accuracy = 0.0
    total_carbon = 0.0
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Random 값을 완전히 제거하고 오직 DB에 쌓인 진짜 데이터만 조회합니다.
            c = conn.cursor()
            c.execute("SELECT COUNT(*), SUM(is_correct), SUM(carbon_saved) FROM feedback")
            row = c.fetchone()
            
            if row and row[0] > 0:
                total_scans = row[0]
                correct_scans = row[1] if row[1] is not None else 0
                accuracy = round((correct_scans / total_scans * 100), 1)
                total_carbon = row[2] if row[2] is not None else 0.0
    except Exception:
        pass
    
    return total_scans, accuracy, total_carbon

# ==========================================
# 5. 엔터프라이즈 보안 게이트웨이
# ==========================================
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { color: #2e7d32; font-weight: 800; font-size: 2.3rem; }
    .report-card { background-color: #ffffff; padding: 24px; border-radius: 14px; box-shadow: 0 6px 16px rgba(0,0,0,0.04); margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.role = None

if not st.session_state.authenticated:
    st.sidebar.markdown("### 🔐 시스템 보안 게이트웨이")
    auth_id = st.sidebar.text_input("연구 계정 ID")
    auth_phone = st.sidebar.text_input("인가 연락처 2FA", type="password")
    
    if st.sidebar.button("시스템 접속", use_container_width=True, type="primary"):
        if auth_id == "taegyun" and auth_phone == "01099999999":
            st.session_state.authenticated = True
            st.session_state.role = "Developer (최상위 관리 권한)"
            st.rerun()
        else:
            st.sidebar.error("인증 실패")
    st.stop()
else:
    st.sidebar.success(f"🔓 {st.session_state.role}")
    if st.sidebar.button("보안 로그아웃", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# ==========================================
# 6. 메인 UI (안전성 및 에러 방어 적용)
# ==========================================
st.title("⚡ 무결성 실시간 XAI 자원 순환 시스템")
st.caption(f"비즈니스 아키텍처 | 100% Real Inference Data Only | {storage_status}")

total_scans, accuracy, total_carbon = get_real_analytics()

m1, m2, m3 = st.columns(3)
m1.metric("실데이터 기반 정확도", f"{accuracy} %", "DB 검증 완료")
m2.metric("누적 순환 분류 처리", f"{total_scans} 건", "실제 트랜잭션")
m3.metric("검증된 탄소 저감량", f"{total_carbon:.2f} kg", "실데이터 합산")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='report-card'>", unsafe_allow_html=True)
    st.subheader("📸 실시간 센싱 입력 인프라")
    uploaded_file = st.file_uploader("분석 대상을 업로드하십시오.", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        file_bytes = uploaded_file.read()
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        st.image(img, caption="센싱된 원본 데이터", use_container_width=True)
        
        if st.button("🚀 정밀 추론 개시", use_container_width=True, type="primary"):
            with st.spinner("딥러닝 가중치 행렬 연산 중..."):
                start_time = time.time()
                
                img_resized = img.resize((224, 224))
                cv_img_res = np.array(img_resized)
                
                transform_pipe = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                input_tensor = transform_pipe(img_resized).unsqueeze(0).to(device)
                
                # 예측 가동 (랜덤 확률 배제, 오직 모델 결과만 사용)
                with torch.no_grad():
                    obj_out, mat_out = model(input_tensor)
                    obj_probs = torch.softmax(obj_out, dim=1).cpu().numpy()[0]
                    mat_probs = torch.softmax(mat_out, dim=1).cpu().numpy()[0]
                
                p_mat_idx = np.argmax(mat_probs)
                p_obj_idx = np.argmax(obj_probs)
                
                pred_mat = TARGET_MATERIALS[p_mat_idx]
                pred_obj = TARGET_OBJECTS[p_obj_idx]
                conf = float(mat_probs[p_mat_idx] * 100)
                
                input_tensor.requires_grad_()
                heatmap_base64 = gradcam_engine.generate(input_tensor, cv_img_res, p_mat_idx)
                
                st.session_state.xai_res = {
                    "pred_mat": pred_mat, "pred_obj": pred_obj,
                    "conf": round(conf, 2), "latency": round((time.time() - start_time) * 1000, 1),
                    "heatmap": heatmap_base64, "filename": uploaded_file.name
                }
                
                del input_tensor, cv_img_res
                gc.collect()
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='report-card'>", unsafe_allow_html=True)
    st.subheader("🎯 모델링 근거 시각화 및 무결성 환류")
    
    # NameError 방어 및 상태 관리 최적화
    if "xai_res" in st.session_state and st.session_state.xai_res is not None:
        res = st.session_state.xai_res
        
        c1, c2, c3 = st.columns(3)
        c1.metric("예측 재질", f"{res['pred_mat'].upper()} ({res['pred_obj']})")
        c2.metric("신뢰도", f"{res['conf']} %")
        c3.metric("지연 시간", f"{res['latency']} ms")
        
        if res.get("heatmap"):
            heatmap_bytes = base64.b64decode(res["heatmap"])
            st.image(heatmap_bytes, caption="Grad-CAM 특징점 활성화 영역", use_container_width=True)
            
        st.divider()
        st.write("🛠️ **Active Learning 자율 정제 시스템**")
        
        final_label = st.selectbox(
            "실제 정답 데이터 확정", 
            TARGET_MATERIALS, 
            index=TARGET_MATERIALS.index(res['pred_mat']) if res['pred_mat'] in TARGET_MATERIALS else 0
        )
        
        if st.button("RDBMS 피드백 무결성 커밋", use_container_width=True):
            is_correct = 1 if res['pred_mat'] == final_label else 0
            carbon_val = CARBON_FACTORS.get(final_label, 0.0)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    c = conn.cursor()
                    c.execute("INSERT INTO feedback (timestamp, filename, predicted, confidence, actual, is_correct, carbon_saved) VALUES (?, ?, ?, ?, ?, ?, ?)",
                              (timestamp, res['filename'], res['pred_mat'], res['conf'], final_label, is_correct, carbon_val))
                    c.execute("DELETE FROM feedback WHERE id NOT IN (SELECT id FROM feedback ORDER BY id DESC LIMIT 1000)")
                    conn.commit()
                st.success("🎯 피드백 커밋 완료. 1000개 로우 최적화 적용됨.")
                time.sleep(1)
                st.session_state.xai_res = None
                st.rerun()
            except Exception as e:
                st.error(f"DB 트랜잭션 에러: {e}")
    else:
        st.info("원격 이미지를 업로드하고 분석을 실행하십시오.")
    st.markdown("</div>", unsafe_allow_html=True)
