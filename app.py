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

        # 1. 기존 기본 지표 (TTM 기반)
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
                runway = "inf" # inf 대신 문자열 처리
        else:
            runway = None

        base_results = [
            round(ttm_dte, 2) if ttm_dte is not None else None,
            round(ttm_cr, 2) if ttm_cr is not None else None,
            round(ttm_opm, 2) if ttm_opm is not None else None,
            runway, # Runway 위치 (기존 로직 유지)
            round(ttm_ocf, 2) if ttm_ocf is not None else None,
            round(ttm_pbr, 2) if ttm_pbr is not None else None,
            round(ttm_per, 2) if ttm_per is not None else None,
            round(ttm_eps, 2) if ttm_eps is not None else None
        ]

        # 2. 항목별 추이 데이터 (TTM 포함 5개 시점: TTM, Y1, Y2, Y3, Y4)
        history = {
            "DTE": [base_results[0]], 
            "CR": [base_results[1]], 
            "OPM": [base_results[2]], 
            "OCF": [base_results[4]], 
            "EPS": [base_results[7]]
        }
        
        num_years = min(len(fin.columns), 4) if not fin.empty else 0

        for i in range(4):
            if i < num_years:
                liab = get_val(bs, 'Total Liabilities Net Minority Interest', i)
                equity = get_val(bs, 'Total Equity Gross Minority Interest', i)
                history["DTE"].append(round((liab/equity*100), 2) if liab and equity else None)
                
                ca = get_val(bs, 'Current Assets', i)
                cl = get_val(bs, 'Current Liabilities', i)
                history["CR"].append(round((ca/cl*100), 2) if ca and cl else None)
                
                op_inc = get_val(fin, 'Operating Income', i)
                rev = get_val(fin, 'Total Revenue', i)
                history["OPM"].append(round((op_inc/rev*100), 2) if op_inc and rev else None)
                
                ocf = get_val(cf, 'Operating Cash Flow', i)
                history["OCF"].append(round(ocf/1_000_000, 2) if ocf else None)
                
                eps = get_val(fin, 'Basic EPS', i)
                history["EPS"].append(round(eps, 2) if eps else None)
            else:
                for key in history:
                    if len(history[key]) < 5: history[key].append(None)

        flattened_history = []
        for key in ["DTE", "CR", "OPM", "OCF", "EPS"]:
            flattened_history.extend(history[key])

        return base_results + flattened_history
    except Exception:
        return [None] * (8 + 25)

# --- [UI] Streamlit 앱 설정 ---
st.set_page_config(page_title="Stock Analysis Pro", layout="wide")
st.title("📊 재무 분석 대시보드 (TTM & 4Y)")

# --- [사이드바] 입력 설정 ---
st.sidebar.header("📥 데이터 소스 설정")
input_method = st.sidebar.radio("입력 방식", ("텍스트 붙여넣기", "구글 스프레드시트", "CSV 파일 업로드"))

tickers = []
if input_method == "텍스트 붙여넣기":
    raw_input = st.sidebar.text_area("티커 입력 (한 줄에 하나)")
    if raw_input: tickers = [t.strip().upper() for t in raw_input.split('\n') if t.strip()]
elif input_method == "구글 스프레드시트":
    try:
        sid, sname = st.secrets["GOOGLE_SHEET_ID"], st.secrets["GOOGLE_SHEET_NAME"]
        url = f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={quote(sname)}"
        gs_df = pd.read_csv(url)
        t_col = st.sidebar.selectbox("티커 열 선택", gs_df.columns)
        tickers = gs_df[t_col].dropna().astype(str).tolist()
    except Exception as e: st.sidebar.error(f"❌ 로드 실패: {e}")
elif input_method == "CSV 파일 업로드":
    up_file = st.sidebar.file_uploader("CSV 업로드", type=["csv"])
    if up_file:
        df = pd.read_csv(up_file)
        t_col = st.sidebar.selectbox("티커 열 선택", df.columns)
        tickers = df[t_col].dropna().astype(str).tolist()

# --- [메인] 분석 실행 ---
if tickers:
    st.write(f"📝 분석 대상: **{len(tickers)}개 종목**")
    if st.button("분석 시작"):
        progress_bar = st.progress(0)
        results_list = []
        
        # 칼럼 헤더 정의
        base_cols = ['ticker', 'debtToEquity(%)', 'currentRatio(%)', 'OperatingMargin(%)', 
                     'Runway(Years)', 'OperatingCashflow(M$)', 'PBR', 'PER', 'EPS', 'lastUpdated']
        
        metrics_step = ["DTE", "CR", "OPM", "OCF", "EPS"]
        years_step = ["TTM", "Y1", "Y2", "Y3", "Y4"]
        history_cols = [f"{m}_{y}" for m in metrics_step for y in years_step]
        final_cols = base_cols + history_cols

        for idx, symbol in enumerate(tickers):
            st.write(f"⏳ {symbol} 진행 중...")
            raw_data = get_extended_financials(symbol)
            
            # 행 데이터 재조합
            row = [symbol] + raw_data[:8] + [datetime.now().strftime('%H:%M:%S')] + raw_data[8:]
            results_list.append(row)
            progress_bar.progress((idx + 1) / len(tickers))
            time.sleep(0.5)

        # 데이터프레임 생성 및 결측치 처리
        res_df = pd.DataFrame(results_list, columns=final_cols)
        
        # ✅ 데이터가 없는(None) 부분을 "-"로 교체
        res_df = res_df.fillna("-")
        
        st.success("✅ 분석 완료!")
        st.dataframe(res_df, use_container_width=True)

        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button("CSV 다운로드", csv, "analysis.csv", "text/csv")
else:
    st.info("👈 사이드바에서 티커를 입력해주세요.")
