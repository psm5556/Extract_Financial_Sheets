import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
from urllib.parse import quote

# --- [함수] 재무 데이터 추출 로직 ---
def get_extended_financials(ticker_symbol):
    try:
        symbol = ticker_symbol.upper().strip()
        ticker = yf.Ticker(symbol)
        
        # 데이터 로드
        info = ticker.info
        fin = ticker.financials
        bs = ticker.balance_sheet
        cf = ticker.cashflow

        def get_val(df, label, idx):
            try: return df.loc[label].iloc[idx]
            except: return None

        # 1. TTM (최근 12개월) 데이터 추출
        ttm_dte = info.get("debtToEquity")
        ttm_cr = (info.get("currentRatio") * 100) if info.get("currentRatio") else None
        ttm_opm = (info.get("operatingMargins") * 100) if info.get("operatingMargins") else None
        ttm_ocf = (info.get("operatingCashflow") / 1_000_000) if info.get("operatingCashflow") else None
        ttm_pbr = info.get("priceToBook")
        ttm_per = info.get("trailingPE")
        ttm_eps = info.get("trailingEps")

        # Runway 계산 및 inf 처리
        total_cash = info.get("totalCash")
        free_cf = info.get("freeCashflow")
        if total_cash and free_cf:
            if free_cf < 0:
                runway = round(total_cash / abs(free_cf), 2)
            else:
                runway = "Infinite (Profit)"
        else:
            runway = None

        base_results = [
            round(ttm_dte, 2) if ttm_dte is not None else None,
            round(ttm_cr, 2) if ttm_cr is not None else None,
            round(ttm_opm, 2) if ttm_opm is not None else None,
            runway,
            round(ttm_ocf, 2) if ttm_ocf is not None else None,
            round(ttm_pbr, 2) if ttm_pbr is not None else None,
            round(ttm_per, 2) if ttm_per is not None else None,
            round(ttm_eps, 2) if ttm_eps is not None else None
        ]

        # 2. 항목별 추이 데이터 (과거순 배치: Y4 -> Y3 -> Y2 -> Y1 -> TTM)
        history = { "DTE": [], "CR": [], "OPM": [], "OCF": [], "EPS": [] }
        num_years = min(len(fin.columns), 4) if not fin.empty else 0

        # Y4부터 Y1까지 역순으로 수집하기 위해 리스트를 미리 만들고 채움
        temp_history = { "DTE": [None]*4, "CR": [None]*4, "OPM": [None]*4, "OCF": [None]*4, "EPS": [None]*4 }

        for i in range(num_years):
            # i=0이 최신(Y1), i=3이 가장 과거(Y4)
            idx = 3 - i # 리스트 내 위치 (Y4가 index 0, Y1이 index 3이 되도록)
            
            liab = get_val(bs, 'Total Liabilities Net Minority Interest', i)
            equity = get_val(bs, 'Total Equity Gross Minority Interest', i)
            temp_history["DTE"][idx] = round((liab/equity*100), 2) if liab and equity else None
            
            ca = get_val(bs, 'Current Assets', i)
            cl = get_val(bs, 'Current Liabilities', i)
            temp_history["CR"][idx] = round((ca/cl*100), 2) if ca and cl else None
            
            op_inc = get_val(fin, 'Operating Income', i)
            rev = get_val(fin, 'Total Revenue', i)
            temp_history["OPM"][idx] = round((op_inc/rev*100), 2) if op_inc and rev else None
            
            ocf = get_val(cf, 'Operating Cash Flow', i)
            temp_history["OCF"][idx] = round(ocf/1_000_000, 2) if ocf else None
            
            eps = get_val(fin, 'Basic EPS', i)
            temp_history["EPS"][idx] = round(eps, 2) if eps else None

        # 최종 조합: [Y4, Y3, Y2, Y1, TTM]
        flattened_history = []
        for key in ["DTE", "CR", "OPM", "OCF", "EPS"]:
            # 과거 데이터(Y4~Y1) + TTM 데이터
            # TTM 데이터 위치 찾기: DTE는 base_results[0], CR은 [1], OPM은 [2], OCF는 [4], EPS는 [7]
            ttm_idx_map = {"DTE": 0, "CR": 1, "OPM": 2, "OCF": 4, "EPS": 7}
            combined = temp_history[key] + [base_results[ttm_idx_map[key]]]
            flattened_history.extend(combined)

        return base_results + flattened_history
    except Exception:
        return [None] * (8 + 25)

# --- [UI] Streamlit 설정 ---
st.set_page_config(page_title="Stock Trend Analyzer", layout="wide")
st.title("📊 재무 추이 분석기 (Y4 → TTM)")

# --- [사이드바] 입력 ---
st.sidebar.header("📥 설정")
method = st.sidebar.radio("입력 방식", ("텍스트 붙여넣기", "구글 스프레드시트", "CSV 파일 업로드"))

tickers = []
if method == "텍스트 붙여넣기":
    raw = st.sidebar.text_area("티커 입력 (한 줄에 하나)")
    if raw: tickers = [t.strip().upper() for t in raw.split('\n') if t.strip()]
elif method == "구글 스프레드시트":
    try:
        sid, sname = st.secrets["GOOGLE_SHEET_ID"], st.secrets["GOOGLE_SHEET_NAME"]
        url = f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={quote(sname)}"
        gs_df = pd.read_csv(url)
        t_col = st.sidebar.selectbox("티커 열 선택", gs_df.columns)
        tickers = gs_df[t_col].dropna().astype(str).tolist()
    except Exception as e: st.sidebar.error(f"로드 실패: {e}")
elif method == "CSV 파일 업로드":
    up = st.sidebar.file_uploader("CSV 업로드", type=["csv"])
    if up:
        df = pd.read_csv(up)
        t_col = st.sidebar.selectbox("티커 열 선택", df.columns)
        tickers = df[t_col].dropna().astype(str).tolist()

# --- [메인] 실행 ---
if tickers:
    st.write(f"📝 대상 종목: **{len(tickers)}개**")
    if st.button("분석 실행"):
        progress = st.progress(0)
        results = []
        
        # 칼럼 헤더 정의 (과거 -> 현재 순서)
        base_cols = ['ticker', 'DTE(%)', 'CR(%)', 'OPM(%)', 'Runway(Y)', 'OCF(M$)', 'PBR', 'PER', 'EPS', 'Updated']
        metrics = ["DTE", "CR", "OPM", "OCF", "EPS"]
        years = ["Y4", "Y3", "Y2", "Y1", "TTM"]
        history_cols = [f"{m}_{y}" for m in metrics for y in years]
        final_cols = base_cols + history_cols

        for idx, symbol in enumerate(tickers):
            st.write(f"🔍 {symbol} 처리 중...")
            data = get_extended_financials(symbol)
            row = [symbol] + data[:8] + [datetime.now().strftime('%H:%M:%S')] + data[8:]
            results.append(row)
            progress.progress((idx + 1) / len(tickers))
            time.sleep(0.5)

        res_df = pd.DataFrame(results, columns=final_cols).fillna("-")
        st.success("✅ 완료!")
        st.dataframe(res_df, use_container_width=True)
        st.download_button("CSV 다운로드", res_df.to_csv(index=False).encode('utf-8'), "report.csv", "text/csv")
else:
    st.info("👈 사이드바에서 데이터를 입력하세요.")
