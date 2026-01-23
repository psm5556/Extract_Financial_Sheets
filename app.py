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
        ttm_roe = (info.get("returnOnEquity") * 100) if info.get("returnOnEquity") else None # ROE 추가
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
            round(ttm_roe, 2) if ttm_roe is not None else None, # ROE 위치
            runway,
            round(ttm_ocf, 2) if ttm_ocf is not None else None,
            round(ttm_pbr, 2) if ttm_pbr is not None else None,
            round(ttm_per, 2) if ttm_per is not None else None,
            round(ttm_eps, 2) if ttm_eps is not None else None
        ]

        # 2. 항목별 추이 데이터 (과거순 배치: Y4 -> Y3 -> Y2 -> Y1 -> TTM)
        history = { "DTE": [], "CR": [], "OPM": [], "ROE": [], "OCF": [], "EPS": [] }
        num_years = min(len(fin.columns), 4) if not fin.empty else 0
        temp_history = { "DTE": [None]*4, "CR": [None]*4, "OPM": [None]*4, "ROE": [None]*4, "OCF": [None]*4, "EPS": [None]*4 }

        for i in range(num_years):
            idx = 3 - i # Y4(0) ~ Y1(3) 위치 잡기
            
            # DTE (부채비율)
            liab = get_val(bs, 'Total Liabilities Net Minority Interest', i)
            equity = get_val(bs, 'Total Equity Gross Minority Interest', i)
            temp_history["DTE"][idx] = round((liab/equity*100), 2) if liab and equity else None
            
            # CR (유동비율)
            ca = get_val(bs, 'Current Assets', i)
            cl = get_val(bs, 'Current Liabilities', i)
            temp_history["CR"][idx] = round((ca/cl*100), 2) if ca and cl else None
            
            # OPM (영업이익률)
            op_inc = get_val(fin, 'Operating Income', i)
            rev = get_val(fin, 'Total Revenue', i)
            temp_history["OPM"][idx] = round((op_inc/rev*100), 2) if op_inc and rev else None

            # ROE (Net Income / Total Equity) - 추가
            net_inc = get_val(fin, 'Net Income', i)
            temp_history["ROE"][idx] = round((net_inc/equity*100), 2) if net_inc and equity else None
            
            # OCF (영업현금흐름)
            ocf = get_val(cf, 'Operating Cash Flow', i)
            temp_history["OCF"][idx] = round(ocf/1_000_000, 2) if ocf else None
            
            # EPS
            eps = get_val(fin, 'Basic EPS', i)
            temp_history["EPS"][idx] = round(eps, 2) if eps else None

        # 최종 조합: [Y4, Y3, Y2, Y1, TTM]
        flattened_history = []
        metrics_order = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS"]
        ttm_idx_map = {"DTE": 0, "CR": 1, "OPM": 2, "ROE": 3, "OCF": 5, "EPS": 8}
        
        for key in metrics_order:
            combined = temp_history[key] + [base_results[ttm_idx_map[key]]]
            flattened_history.extend(combined)

        return base_results + flattened_history
    except Exception:
        return [None] * (9 + 30) # 기본 9개 + (6개 항목 * 5개 시점)

# --- [UI] Streamlit 설정 ---
st.set_page_config(page_title="Stock Deep Analysis", layout="wide")
st.title("📊 재무 추이 정밀 분석기 (Y4 → TTM)")

# --- [사이드바] 입력 ---
st.sidebar.header("📥 데이터 입력")
method = st.sidebar.radio("방식 선택", ("텍스트 붙여넣기", "구글 스프레드시트", "CSV 파일 업로드"))

tickers = []
if method == "텍스트 붙여넣기":
    raw = st.sidebar.text_area("티커 입력 (예: AAPL)")
    if raw: tickers = [t.strip().upper() for t in raw.split('\n') if t.strip()]
elif method == "구글 스프레드시트":
    try:
        sid, sname = st.secrets["GOOGLE_SHEET_ID"], st.secrets["GOOGLE_SHEET_NAME"]
        url = f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={quote(sname)}"
        gs_df = pd.read_csv(url)
        t_col = st.sidebar.selectbox("티커 컬럼 선택", gs_df.columns)
        tickers = gs_df[t_col].dropna().astype(str).tolist()
    except Exception as e: st.sidebar.error(f"연결 실패: {e}")
elif method == "CSV 파일 업로드":
    up = st.sidebar.file_uploader("파일 선택", type=["csv"])
    if up:
        df = pd.read_csv(up)
        t_col = st.sidebar.selectbox("티커 컬럼 선택", df.columns)
        tickers = df[t_col].dropna().astype(str).tolist()

# --- [메인] 실행 ---
if tickers:
    st.write(f"📝 대상 종목: **{len(tickers)}개**")
    if st.button("전수 분석 시작"):
        progress = st.progress(0)
        results = []
        
        # 칼럼 헤더 정의
        base_cols = ['ticker', 'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Runway(Y)', 'OCF(M$)', 'PBR', 'PER', 'EPS', 'Updated']
        metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS"]
        years = ["Y4", "Y3", "Y2", "Y1", "TTM"]
        history_cols = [f"{m}_{y}" for m in metrics for y in years]
        final_cols = base_cols + history_cols

        for idx, symbol in enumerate(tickers):
            st.write(f"🔍 {symbol} 분석 중...")
            data = get_extended_financials(symbol)
            # data structure: [9개 기본 데이터] + [30개 히스토리 데이터]
            row = [symbol] + data[:9] + [datetime.now().strftime('%H:%M:%S')] + data[9:]
            results.append(row)
            progress.progress((idx + 1) / len(tickers))
            time.sleep(0.5)

        res_df = pd.DataFrame(results, columns=final_cols).fillna("-")
        st.success("✅ 분석 완료!")
        st.dataframe(res_df, use_container_width=True)
        st.download_button("결과 CSV 다운로드", res_df.to_csv(index=False).encode('utf-8'), f"report_{datetime.now().strftime('%m%d')}.csv", "text/csv")
else:
    st.info("👈 분석할 티커를 입력해주세요.")
