# frontend.py
import streamlit as st
import requests
import io
import base64
import pandas as pd
from PIL import Image

st.set_page_config(page_title="EcoVision Enterprise Platform", page_icon="⚡", layout="wide")
BACKEND_URL = "http://localhost:8000"

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { color: #2e7d32; font-weight: 800; font-size: 2.3rem; }
    .report-card { background-color: #ffffff; padding: 24px; border-radius: 14px; box-shadow: 0 6px 16px rgba(0,0,0,0.04); margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 1. 엔터프라이즈 보안 인가 모듈
# -----------------------------
def enterprise_login_system():
    st.sidebar.markdown("### 🔐 시스템 보안 접근")
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.role = None

    if not st.session_state.authenticated:
        st.sidebar.info("관리자 및 개발자 인가가 필요합니다.")
        auth_id = st.sidebar.text_input("사번 또는 계정 ID")
        auth_phone = st.sidebar.text_input("인가된 연락처 (2FA 대체용)", type="password")
        
        if st.sidebar.button("시스템 접속", use_container_width=True, type="primary"):
            # 개발자 고속 우회(Bypass) 로직
            if auth_id == "taegyun" and auth_phone == "01099999999":
                st.session_state.authenticated = True
                st.session_state.role = "Developer(최상위)"
                st.rerun()
            else:
                st.sidebar.error("인가되지 않은 사용자이거나 정보가 일치하지 않습니다.")
        st.stop()
    else:
        st.sidebar.success(f"접속 권한: {st.session_state.role}")
        if st.sidebar.button("보안 로그아웃", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

enterprise_login_system()

# -----------------------------
# 2. 메인 대시보드 애플리케이션
# -----------------------------
st.title("⚡ 고성능 자원 순환 자동화 및 XAI 모니터링 시스템")
st.caption("비즈니스 아키텍처 / Explainable AI (Linear Transformation) & Edge Performance")

if "xai_res" not in st.session_state: st.session_state.xai_res = None
if "uploaded_filename" not in st.session_state: st.session_state.uploaded_filename = None

try:
    analytics_res = requests.get(f"{BACKEND_URL}/analytics").json()
except:
    analytics_res = {"total_scans": 0, "system_accuracy": 0.0, "chart_labels": [], "carbon_trends": [], "edge_load_pct": 0.0}
    st.warning("데이터베이스 백엔드 서버에 연결할 수 없습니다. FastAPI 서버를 가동해 주십시오.")

st.subheader("🌐 인프라 가동 모니터링 및 누적 ESG 실적")
m1, m2, m3, m4 = st.columns(4)
m1.metric("종합 판단 무결성", f"{analytics_res['system_accuracy']} %", "RDBMS 검증")
m2.metric("누적 인프라 순환 분류", f"{analytics_res['total_scans']} 건", "▲ 자동 수집 중")
m3.metric("Edge CPU 가동률", f"{analytics_res['edge_load_pct']:.1f} %", "🟢 최적화 완료")
m4.metric("당일 누적 탄소 저감", f"{sum(analytics_res['carbon_trends']):.1f} kg", "ESG 종합 기여")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='report-card'>", unsafe_allow_html=True)
    st.subheader("📸 고해상도 자원 샘플 입력")
    uploaded_file = st.file_uploader("검증을 진행할 순환 자원 이미지 데이터를 업로드하십시오.", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="입력 원본 데이터 세트", use_container_width=True)
        
        if st.button("🚀 정밀 고속 추론 프로세스 가동", use_container_width=True, type="primary"):
            with st.spinner("FastAPI 백엔드 행렬 연산 및 특징점 추출 중..."):
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format if img.format else "JPEG")
                files = {"file": (uploaded_file.name, img_byte_arr.getvalue(), "image/jpeg")}
                try:
                    res = requests.post(f"{BACKEND_URL}/predict", files=files)
                    if res.status_code == 200:
                        st.session_state.xai_res = res.json()
                        st.session_state.uploaded_filename = uploaded_file.name
                    else: st.error("백엔드 딥러닝 컴파일 엔진 장애가 식별되었습니다.")
                except Exception as e: st.error(f"원격 서버 커넥션 장애: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='report-card'>", unsafe_allow_html=True)
    st.subheader("🎯 모델링 근거 시각화 분석")
    
    if st.session_state.xai_res:
        res = st.session_state.xai_res
        p_label, conf, latency = res["prediction"], res["confidence"], res["latency_ms"]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("AI 최종 예측", p_label.upper())
        c2.metric("산출 신뢰도", f"{conf} %")
        c3.metric("Edge 지연 처리", f"{latency} ms")
        
        st.write("🔍 **선형 변환 특징점 맵 분석 (XAI 추출 결과)**")
        heatmap_bytes = base64.b64decode(res["heatmap_data"])
        st.image(heatmap_bytes, caption="Grad-CAM 레이어 매핑 (선형 행렬 변환 활성화 영역)", use_container_width=True)
        
        st.divider()
        st.write("🛠️ **Active Learning 자율형 데이터 정제 시스템**")
        classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        final_label = st.selectbox("정답 재질 정정 레이블 지정 (DB 연동)", classes, index=classes.index(p_label) if p_label in classes else 0)
        
        if st.button("RDBMS 피드백 무결성 커밋", use_container_width=True):
            payload = {"filename": st.session_state.uploaded_filename, "predicted_label": p_label, "confidence": conf, "final_label": final_label}
            fb_res = requests.post(f"{BACKEND_URL}/feedback", json=payload)
            if fb_res.status_code == 200:
                st.success("🎯 피드백 데이터가 SQLite 엔터프라이즈 DB에 트랜잭션 기록되었습니다.")
    else:
        st.info("좌측 입력 영역에 순환 자원 샘플을 바인딩하면 시스템이 가동됩니다.")
    st.markdown("</div>", unsafe_allow_html=True)
