import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from urllib.parse import quote

# --- [함수] 재무 데이터 추출 로직 ---
def get_extended_financials(ticker_symbol):
    try:
        symbol = ticker_symbol.upper().strip()
        ticker = yf.Ticker(symbol)
        
        info = ticker.info
        fin = ticker.financials
        bs = ticker.balance_sheet
        cf = ticker.cashflow

        def get_val(df, label, idx):
            try: return df.loc[label].iloc[idx]
            except: return None

        # 1. TTM (최근 12개월) 기본 데이터
        ttm_dte = info.get("debtToEquity")
        ttm_cr = (info.get("currentRatio") * 100) if info.get("currentRatio") else None
        ttm_opm = (info.get("operatingMargins") * 100) if info.get("operatingMargins") else None
        ttm_roe = (info.get("returnOnEquity") * 100) if info.get("returnOnEquity") else None
        ttm_ocf = info.get("operatingCashflow")
        ttm_fcf = info.get("freeCashflow")
        ttm_net_inc = info.get("netIncomeToCommon")
        
        # Runway 계산
        total_cash = info.get("totalCash")
        if total_cash and ttm_fcf:
            runway = round(total_cash / abs(ttm_fcf), 2) if ttm_fcf < 0 else "Infinite (Profit)"
        else:
            runway = None

        # base_cols용 데이터 (CFQ 제외)
        base_results = [
            round(ttm_dte, 2) if ttm_dte is not None else None,
            round(ttm_cr, 2) if ttm_cr is not None else None,
            round(ttm_opm, 2) if ttm_opm is not None else None,
            round(ttm_roe, 2) if ttm_roe is not None else None,
            runway,
            round(ttm_ocf / 1_000_000, 2) if ttm_ocf else None,
            round(info.get("priceToBook"), 2) if info.get("priceToBook") else None,
            round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
            round(info.get("trailingEps"), 2) if info.get("trailingEps") else None
        ]

        # 2. 항목별 추이 데이터 (Y4 -> Y3 -> Y2 -> Y1 -> TTM)
        metrics_order = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history = {m: [None]*4 for m in metrics_order}
        num_years = min(len(fin.columns), 4) if not fin.empty else 0

        for i in range(num_years):
            idx = 3 - i 
            net_inc = get_val(fin, 'Net Income', i)
            ocf = get_val(cf, 'Operating Cash Flow', i)
            # FCF 계산 (OCF + CapEx)
            cap_ex = get_val(cf, 'Capital Expenditure', i)
            fcf_val = (ocf + cap_ex) if ocf is not None and cap_ex is not None else None
            
            history["DTE"][idx] = round((get_val(bs, 'Total Liabilities Net Minority Interest', i)/get_val(bs, 'Total Equity Gross Minority Interest', i)*100), 2) if get_val(bs, 'Total Liabilities Net Minority Interest', i) and get_val(bs, 'Total Equity Gross Minority Interest', i) else None
            history["CR"][idx] = round((get_val(bs, 'Current Assets', i)/get_val(bs, 'Current Liabilities', i)*100), 2) if get_val(bs, 'Current Assets', i) and get_val(bs, 'Current Liabilities', i) else None
            history["OPM"][idx] = round((get_val(fin, 'Operating Income', i)/get_val(fin, 'Total Revenue', i)*100), 2) if get_val(fin, 'Operating Income', i) and get_val(fin, 'Total Revenue', i) else None
            history["ROE"][idx] = round((net_inc/get_val(bs, 'Total Equity Gross Minority Interest', i)*100), 2) if net_inc and get_val(bs, 'Total Equity Gross Minority Interest', i) else None
            history["OCF"][idx] = round(ocf/1_000_000, 2) if ocf else None
            history["EPS"][idx] = round(get_val(fin, 'Basic EPS', i), 2) if get_val(fin, 'Basic EPS', i) else None
            history["CFQ"][idx] = round(ocf/net_inc, 2) if ocf and net_inc and net_inc != 0 else None
            history["FCF"][idx] = round(fcf_val/1_000_000, 2) if fcf_val else None

        # TTM 데이터 맵핑
        ttm_vals = {
            "DTE": base_results[0], "CR": base_results[1], "OPM": base_results[2], "ROE": base_results[3],
            "OCF": base_results[5], "EPS": base_results[8],
            "CFQ": round(ttm_ocf/ttm_net_inc, 2) if ttm_ocf and ttm_net_inc and ttm_net_inc != 0 else None,
            "FCF": round(ttm_fcf/1_000_000, 2) if ttm_fcf else None
        }

        flattened_history = []
        for key in metrics_order:
            combined = history[key] + [ttm_vals[key]]
            flattened_history.extend(combined)

        # 3. FCF Stability Index 계산 (플러스 횟수 비율)
        # 대상: Y4, Y3, Y2, Y1, TTM (총 5개 시점)
        fcf_series = history["FCF"] + [ttm_vals["FCF"]]
        plus_count = sum(1 for v in fcf_series if v is not None and v > 0)
        
        # 데이터가 하나라도 있는 경우에만 계산 (분모는 5로 고정)
        if any(v is not None for v in fcf_series):
            stability = (plus_count / 5) * 100
        else:
            stability = None

        return base_results + [stability] + flattened_history
    except Exception:
        return [None] * (10 + 40)

# --- [UI] Streamlit 설정 ---
st.set_page_config(page_title="Stock Professional Analyzer", layout="wide")
st.title("📊 재무 정밀 분석 리포트 (Y4 → TTM)")

# --- [사이드바] ---
st.sidebar.header("📥 데이터 소스")
method = st.sidebar.radio("방식", ("텍스트 붙여넣기", "구글 스프레드시트", "CSV 파일 업로드"))
tickers = []
if method == "텍스트 붙여넣기":
    raw = st.sidebar.text_area("티커 입력 (한 줄에 하나)")
    if raw: tickers = [t.strip().upper() for t in raw.split('\n') if t.strip()]
elif method == "구글 스프레드시트":
    try:
        sid, sname = st.secrets["GOOGLE_SHEET_ID"], st.secrets["GOOGLE_SHEET_NAME"]
        url = f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={quote(sname)}"
        gs_df = pd.read_csv(url)
        t_col = st.sidebar.selectbox("티커 컬럼", gs_df.columns)
        tickers = gs_df[t_col].dropna().astype(str).tolist()
    except Exception as e: st.sidebar.error(f"연결 실패: {e}")
elif method == "CSV 파일 업로드":
    up = st.sidebar.file_uploader("CSV 업로드", type=["csv"])
    if up:
        df = pd.read_csv(up); t_col = st.sidebar.selectbox("티커 컬럼", df.columns)
        tickers = df[t_col].dropna().astype(str).tolist()

# --- [메인] 실행 ---
if tickers:
    total = len(tickers)
    if st.button("🚀 전수 분석 시작"):
        prog = st.progress(0); status = st.empty(); results = []
        
        # 헤더 정의
        base_cols = ['ticker', 'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Runway(Y)', 'OCF(M$)', 'PBR', 'PER', 'EPS', 'FCF_Stability(%)', 'Updated']
        metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history_cols = [f"{m}_{y}" for m in metrics for y in ["Y4", "Y3", "Y2", "Y1", "TTM"]]
        final_cols = base_cols + history_cols

        for idx, symbol in enumerate(tickers):
            status.markdown(f"### ⏳ 분석 중: **{symbol}** ({idx+1}/{total})")
            data = get_extended_financials(symbol)
            # data: base(9) + stability(1) + history(40)
            row = [symbol] + data[:10] + [datetime.now().strftime('%H:%M:%S')] + data[10:]
            results.append(row)
            prog.progress((idx+1)/total); time.sleep(0.5)

        status.success(f"✅ {total}개 종목 분석 완료!")
        res_df = pd.DataFrame(results, columns=final_cols).fillna("-")
        st.dataframe(res_df, use_container_width=True)
        st.download_button("📥 CSV 다운로드", res_df.to_csv(index=False).encode('utf-8'), "financial_report.csv", "text/csv")
