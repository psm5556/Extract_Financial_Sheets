import streamlit as st
import yfinance as yf
import pandas as pd
import gspread
from datetime import datetime
import time
from urllib.parse import quote

# --- 재무 지표 가져오기 함수 ---
def get_financial_ratios(ticker_symbol):
    try:
        symbol = ticker_symbol.upper().strip()
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # 데이터 추출
        dte = info.get("debtToEquity")
        cr = info.get("currentRatio")
        op_margin = info.get("operatingMargins")  # ✅ 영업이익률 추가
        roe = info.get("returnOnEquity")
        total_cash = info.get("totalCash")
        free_cf = info.get("freeCashflow")
        operating_cf = info.get("operatingCashflow")
        net_income = info.get("netIncomeToCommon")
        pbr = info.get("priceToBook")
        bps = info.get("bookValue")

        # 단위 변환 (%)
        cr = round(cr * 100, 2) if cr else None
        op_margin = round(op_margin * 100, 2) if op_margin else None  # ✅ % 변환
        roe = round(roe * 100, 2) if roe else None
        
        def to_million(val):
            return round(val / 1_000_000, 2) if val else None

        # Runway 계산
        runway_years = None
        if total_cash and free_cf:
            if free_cf < 0:
                runway_years = round(total_cash / abs(free_cf), 2)
            else:
                runway_years = float('inf')

        # 리스트 순서에 op_margin 삽입 (유동비율 다음)
        return [
            dte, cr, op_margin, roe, runway_years, 
            to_million(total_cash), to_million(free_cf), 
            to_million(operating_cf), to_million(net_income),
            round(pbr, 2) if pbr else None, 
            round(bps, 2) if bps else None
        ]
    except Exception:
        return [None] * 11  # 컬럼이 하나 늘었으므로 11개 반환

# --- UI 구성 ---
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("🚀 주식 분석 자동화 도구")

# --- 사이드바: 입력 방식 선택 ---
st.sidebar.header("📥 티커 입력 설정")
input_method = st.sidebar.radio(
    "입력 방식을 선택하세요",
    ("텍스트 붙여넣기", "구글 스프레드시트", "CSV 파일 업로드")
)

tickers = []

if input_method == "텍스트 붙여넣기":
    raw_input = st.sidebar.text_area("티커를 입력하세요 (한 줄에 하나씩)", height=200, placeholder="AAPL\nTSLA\nNVDA")
    if raw_input:
        tickers = [t.strip() for t in raw_input.split('\n') if t.strip()]

elif input_method == "구글 스프레드시트":
    try:
        sheet_id = st.secrets["GOOGLE_SHEET_ID"]
        sheet_name = st.secrets["GOOGLE_SHEET_NAME"]
        
        # 1. 시트 이름에 한글이 있을 경우를 대비해 URL 인코딩 처리
        encoded_sheet_name = quote(sheet_name)
        
        # 2. 구글 시트 CSV 내보내기 URL 구성
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={encoded_sheet_name}"
        
        # 3. 데이터 읽기
        gs_df = pd.read_csv(url)
        
        st.sidebar.success(f"✅ 시트 연결 성공: {sheet_name}")
        ticker_col = st.sidebar.selectbox("티커가 포함된 열(Column) 선택", gs_df.columns)
        tickers = gs_df[ticker_col].dropna().astype(str).tolist()
        
    except Exception as e:
        st.sidebar.error(f"구글 시트 로드 실패: {e}")

elif input_method == "CSV 파일 업로드":
    uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        ticker_col = st.sidebar.selectbox("티커 열 선택", df.columns)
        tickers = df[ticker_col].dropna().astype(str).tolist()

# --- 메인 실행 화면 ---
if tickers:
    st.write(f"🔍 분석 대상 티커 개수: **{len(tickers)}개**")
    
    if st.button("데이터 분석 시작"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_list = []

        for idx, symbol in enumerate(tickers):
            status_text.text(f"진행 중: {symbol} ({idx+1}/{len(tickers)})")
            data = get_financial_ratios(symbol)
            results_list.append([symbol] + data + [datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            progress_bar.progress((idx + 1) / len(tickers))
            time.sleep(0.4) # API 호출 제한 방지

        # 컬럼 이름 리스트
        columns = [
            'ticker', 'debtToEquity(%)', 'currentRatio(%)', 'OperatingMargin(%)', 'ROE(%)', 
            'Runway(Years)', 'TotalCash(M$)', 'FreeCashflow(M$)', 
            'OperatingCashflow(M$)', 'NetIncome(M$)', 'PBR', 'BPS($)', 'lastUpdated'
        ]
        res_df = pd.DataFrame(results_list, columns=columns)
        
        st.success("✅ 분석 완료!")
        st.dataframe(res_df, use_container_width=True)

        # 결과 다운로드
        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="결과 CSV 다운로드",
            data=csv,
            file_name=f"stock_report_{datetime.now().strftime('%m%d_%H%M')}.csv",
            mime='text/csv'
        )
else:
    st.info("왼쪽 사이드바에서 티커 목록을 제공해 주세요.")
