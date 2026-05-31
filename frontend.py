import streamlit as st
import requests
import io
import base64
import pandas as pd
from PIL import Image

st.set_page_config(page_title="신소재/자원 순환 XAI 플랫폼", page_icon="⚡", layout="wide")
BACKEND_URL = "http://localhost:8000"

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { color: #2e7d32; font-weight: 800; font-size: 2.3rem; }
    .report-card { background-color: #ffffff; padding: 24px; border-radius: 14px; box-shadow: 0 6px 16px rgba(0,0,0,0.04); margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# 1. 시스템 보안 인가 모듈
def enterprise_login_system():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.role = None

    if not st.session_state.authenticated:
        st.sidebar.markdown("### 🔐 시스템 보안 접근")
        auth_id = st.sidebar.text_input("연구 계정 ID")
        auth_phone = st.sidebar.text_input("인가 연락처 2FA 보안키", type="password")
        
        c1, c2 = st.sidebar.columns(2)
        with c1:
            if st.sidebar.button("연구원 인증", use_container_width=True, type="primary"):
                if auth_id == "taegyun" and auth_phone == "01099999999":
                    st.session_state.authenticated = True
                    st.session_state.role = "Developer (최상위 관리 권한)"
                    st.rerun()
                else:
                    st.sidebar.error("인증 자격 유효성 실패")
        with c2:
            if st.sidebar.button("게스트 입장", use_container_width=True):
                st.session_state.authenticated = True
                st.session_state.role = "Guest (분석 전용 권한)"
                st.rerun()
        st.stop()
    else:
        st.sidebar.success(f"🔓 권한: {st.session_state.role}")
        if st.sidebar.button("보안 로그아웃", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.role = None
            st.rerun()

enterprise_login_system()

# 2. 메인 대시보드
st.title("⚡ 신소재 및 자원 순환 자동화 분석 시스템")
st.caption("비즈니스/대학 학술 아키텍처 | 분산형 엣지 컴파일 구조")

if "xai_res" not in st.session_state: st.session_state.xai_res = None
if "uploaded_filename" not in st.session_state: st.session_state.uploaded_filename = None

try:
    analytics_res = requests.get(f"{BACKEND_URL}/analytics").json()
except Exception:
    analytics_res = {"total_scans": 248, "system_accuracy": 96.4, "chart_labels": [], "carbon_trends": [], "edge_load_pct": 18.4}
    st.warning("⚠️ 백엔드 데이터 코어 서버와 오프라인 상태입니다. 추론 파이프라인 가동을 위해 FastAPI 서버를 확인하십시오.")

st.subheader("🌐 인프라 가동 모니터링 및 누적 성과 지표")
m1, m2, m3, m4 = st.columns(4)
m1.metric("종합 판단 무결성", f"{analytics_res['system_accuracy']} %", "RDBMS 검증")
m2.metric("누적 인프라 스캔", f"{analytics_res['total_scans']} 회", "실시간 누적 데이터")
m3.metric("탄소 지표 추세성", f"{round(analytics_res['total_scans'] * 0.12, 1)} kg", "CO2 Reduction")
m4.metric("분산 엣지 서버 로드율", f"{analytics_res['edge_load_pct']} %", "안정 수준")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='report-card'>", unsafe_allow_html=True)
    st.subheader("📸 초정밀 원격 객체 이미지 센싱")
    uploaded_file = st.file_uploader("분석 대상 샘플 이미지를 드롭하십시오.", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="센싱된 로우 데이터 (Raw Image Matrix)", use_container_width=True)
        
        if st.button("🚀 백엔드 분산 엔진으로 고속 추론 및 특징 변환 요청", use_container_width=True):
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            
            try:
                files = {"file": (uploaded_file.name, img_bytes, "image/jpeg")}
                res = requests.post(f"{BACKEND_URL}/predict", files=files)
                if res.status_code == 200:
                    st.session_state.xai_res = res.json()
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.rerun()
                else:
                    st.error("백엔드 딥러닝 연산 인프라 장애 식별")
            except Exception as e:
                st.error(f"원격 마이크로서비스 커넥션 장애: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='report-card'>", unsafe_allow_html=True)
    st.subheader("🎯 모델링 근거 시각화 분석")
    
    if st.session_state.xai_res:
        res = st.session_state.xai_res
        p_label, conf, latency = res["prediction"], res["confidence"], res["latency_ms"]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("AI 최종 예측 재질", p_label.upper())
        c2.metric("산출 신뢰도", f"{round(conf, 1)} %")
        c3.metric("Edge 지연 처리", f"{latency} ms")
        
        st.write("🔍 **선형 변환 특징점 맵 분석 (배경 소거 완료)**")
        heatmap_bytes = base64.b64decode(res["heatmap_data"])
        st.image(heatmap_bytes, caption="Grad-CAM 레이어 매핑 (객체 표면 질감 활성화 영역)", use_container_width=True)
        
        if "Developer" in st.session_state.role:
            st.divider()
            st.write("🛠️ **Active Learning 자율형 데이터 정제 시스템**")
            classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
            final_label = st.selectbox("정답 재질 정정 레이블 지정 (DB 연동)", classes, index=classes.index(p_label) if p_label in classes else 0)
            
            if st.button("RDBMS 피드백 무결성 커밋", use_container_width=True):
                payload = {
                    "filename": st.session_state.uploaded_filename,
                    "predicted_label": p_label,
                    "confidence": conf,
                    "final_label": final_label
                }
                f_res = requests.post(f"{BACKEND_URL}/feedback", json=payload)
                if f_res.status_code == 200:
                    st.success(f"🎯 [{final_label}] 피드백이 실시간 데이터 백엔드에 안전하게 영구 커밋되었습니다.")
                    time.sleep(1)
                    st.session_state.xai_res = None
                    st.rerun()
                else:
                    st.error("피드백 데이터 기여 컴파일 실패")
        else:
            st.info("🔒 게스트 권한은 모니터링 및 시각화 조회 전용입니다. 피드백 쓰기는 제한됩니다.")
    else:
        st.info("프론트엔드 센싱 레이어에 데이터가 수신되지 않았습니다.")
    st.markdown("</div>", unsafe_allow_html=True)
