import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from urllib.parse import quote

# --- [1. 재무 데이터 추출 로직] ---
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

        # TTM 및 기본 데이터
        ttm_dte = info.get("debtToEquity")
        ttm_cr = (info.get("currentRatio") * 100) if info.get("currentRatio") else None
        ttm_opm = (info.get("operatingMargins") * 100) if info.get("operatingMargins") else None
        ttm_roe = (info.get("returnOnEquity") * 100) if info.get("returnOnEquity") else None
        ttm_ocf = info.get("operatingCashflow")
        ttm_fcf = info.get("freeCashflow")
        ttm_net_inc = info.get("netIncomeToCommon")
        total_cash = info.get("totalCash")
        
        # Runway 계산
        if total_cash and ttm_fcf:
            runway = round(total_cash / abs(ttm_fcf), 2) if ttm_fcf < 0 else "Infinite"
        else: runway = None

        # 항목별 추이 (Y4 -> TTM)
        metrics_order = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history = {m: [None]*4 for m in metrics_order}
        num_years = min(len(fin.columns), 4) if not fin.empty else 0

        for i in range(num_years):
            idx = 3 - i 
            net_inc = get_val(fin, 'Net Income', i)
            equity = get_val(bs, 'Total Equity Gross Minority Interest', i)
            ocf_val = get_val(cf, 'Operating Cash Flow', i)
            cap_ex = get_val(cf, 'Capital Expenditure', i)
            fcf_val = (ocf_val + cap_ex) if ocf_val is not None and cap_ex is not None else None
            
            history["DTE"][idx] = round((get_val(bs, 'Total Liabilities Net Minority Interest', i)/equity*100), 2) if get_val(bs, 'Total Liabilities Net Minority Interest', i) and equity else None
            history["CR"][idx] = round((get_val(bs, 'Current Assets', i)/get_val(bs, 'Current Liabilities', i)*100), 2) if get_val(bs, 'Current Assets', i) and get_val(bs, 'Current Liabilities', i) else None
            history["OPM"][idx] = round((get_val(fin, 'Operating Income', i)/get_val(fin, 'Total Revenue', i)*100), 2) if get_val(fin, 'Operating Income', i) and get_val(fin, 'Total Revenue', i) else None
            history["ROE"][idx] = round((net_inc/equity*100), 2) if net_inc and equity else None
            history["OCF"][idx] = round(ocf_val/1_000_000, 2) if ocf_val else None
            history["EPS"][idx] = round(get_val(fin, 'Basic EPS', i), 2) if get_val(fin, 'Basic EPS', i) else None
            history["CFQ"][idx] = round(ocf_val/net_inc, 2) if ocf_val and net_inc and net_inc != 0 else None
            history["FCF"][idx] = round(fcf_val/1_000_000, 2) if fcf_val else None

        ttm_fcf_m = round(ttm_fcf/1_000_000, 2) if ttm_fcf else None
        
        # 기본 결과 리스트 (13개 항목)
        base_results = [
            round(ttm_dte, 2) if ttm_dte is not None else None,
            round(ttm_cr, 2) if ttm_cr is not None else None,
            round(ttm_opm, 2) if ttm_opm is not None else None,
            round(ttm_roe, 2) if ttm_roe is not None else None,
            runway,
            round(total_cash / 1_000_000, 2) if total_cash else None,
            ttm_fcf_m,
            None, # Stability (추후 계산)
            round(ttm_ocf / 1_000_000, 2) if ttm_ocf else None,
            round(info.get("priceToBook"), 2) if info.get("priceToBook") else None,
            round(info.get("bookValue"), 2) if info.get("bookValue") else None,
            round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
            round(info.get("trailingEps"), 2) if info.get("trailingEps") else None
        ]

        ttm_vals_map = {
            "DTE": base_results[0], "CR": base_results[1], "OPM": base_results[2], 
            "ROE": base_results[3], "OCF": base_results[8], "EPS": base_results[12],
            "CFQ": round(ttm_ocf/ttm_net_inc, 2) if ttm_ocf and ttm_net_inc and ttm_net_inc != 0 else None,
            "FCF": ttm_fcf_m
        }
        
        flattened_history = []
        for key in metrics_order:
            combined = history[key] + [ttm_vals_map[key]]
            flattened_history.extend(combined)

        return base_results + flattened_history
    except Exception:
        return [None] * (13 + 40)

# --- [2. 투자 등급 평가 로직] ---
def evaluate_stock(row):
    score = 0
    reasons = []
    
    # 1. EPS 성장성 (Y3 -> TTM)
    try:
        eps_y3 = float(row.get('EPS_Y3', 0))
        eps_ttm = float(row.get('EPS_TTM', 0))
        if eps_ttm > eps_y3 and eps_y3 > 0:
            score += 30
            reasons.append("EPS 성장")
    except: pass

    # 2. 현금흐름 질 (CFQ)
    try:
        cfq = float(row.get('CFQ_TTM', 0))
        if cfq >= 1.0:
            score += 30
            reasons.append("현금질 우수")
    except: pass

    # 3. ROE (15% 기준)
    try:
        roe = float(row.get('ROE(%)', 0))
        if roe >= 15:
            score += 20
            reasons.append("고수익성(ROE)")
    except: pass

    # 4. 부채비율 (100% 기준)
    try:
        dte = float(row.get('DTE(%)', 1000))
        if dte <= 100:
            score += 20
            reasons.append("재무 건전")
    except: pass

    if score >= 80: grade = "S (강력 매수)"
    elif score >= 60: grade = "A (우량주)"
    elif score >= 40: grade = "B (보통)"
    else: grade = "C (투자 유의)"
    
    return grade, " | ".join(reasons)

# --- [3. UI 설정] ---
st.set_page_config(page_title="Stock Master Pro", layout="wide")
st.title("📊 주식 재무 시계열 분석 및 투자 평가")

# 사이드바 데이터 소스
st.sidebar.header("📥 데이터 소스")
method = st.sidebar.radio("방식", ("텍스트 붙여넣기", "CSV 파일 업로드"))
tickers = []

if method == "텍스트 붙여넣기":
    raw = st.sidebar.text_area("티커 입력 (예: AAPL, TSLA)")
    if raw: tickers = [t.strip().upper() for t in raw.replace(',', '\n').split('\n') if t.strip()]
else:
    up = st.sidebar.file_uploader("CSV", type=["csv"])
    if up:
        df = pd.read_csv(up); t_col = st.sidebar.selectbox("티커 컬럼", df.columns)
        tickers = df[t_col].dropna().astype(str).tolist()

# 메인 분석 실행
if tickers:
    if st.button("🚀 전수 분석 및 등급 평가 시작"):
        prog = st.progress(0); status = st.empty(); results = []
        
        base_cols = ['ticker', 'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Runway(Y)', 
                     'TotalCash(M$)', 'FCF(M$)', 'FCF_Stability(%)', 'OCF(M$)', 
                     'PBR', 'BPS', 'PER', 'EPS', 'Updated']
        metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history_cols = [f"{m}_{y}" for m in metrics for y in ["Y4", "Y3", "Y2", "Y1", "TTM"]]
        final_cols = base_cols + history_cols

        for idx, symbol in enumerate(tickers):
            status.markdown(f"### ⏳ 분석 중: **{symbol}** ({idx+1}/{len(tickers)})")
            data = get_extended_financials(symbol)
            row_data = [symbol] + data[:13] + [datetime.now().strftime('%H:%M:%S')] + data[13:]
            results.append(row_data)
            prog.progress((idx+1)/len(tickers))
            time.sleep(0.5)

        res_df = pd.DataFrame(results, columns=final_cols)
        
        # 투자 평가 적용
        res_df['투자 등급'], res_df['평가 근거'] = zip(*res_df.apply(evaluate_stock, axis=1))
        
        # 결과 화면 출력
        st.success("✅ 분석 완료!")
        
        # 핵심 요약 테이블 (등급 중심)
        st.subheader("🎯 종합 투자 등급 리포트")
        summary_cols = ['ticker', '투자 등급', '평가 근거', 'ROE(%)', 'EPS_TTM', 'DTE(%)', 'PER']
        st.dataframe(res_df[summary_cols].sort_values(by='투자 등급'), use_container_width=True)

        # 전체 상세 데이터
        st.subheader("📈 상세 시계열 데이터 (Y4 → TTM)")
        st.dataframe(res_df.fillna("-"), use_container_width=True)
        
        st.download_button("📥 결과 CSV 다운로드", res_df.to_csv(index=False).encode('utf-8'), "stock_analysis_report.csv")
