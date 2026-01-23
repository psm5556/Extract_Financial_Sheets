import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from urllib.parse import quote

# --- [함수] 투자 등급 평가 로직 (Rule-based) ---
def evaluate_investment(row):
    score = 0
    reasons = []
    
    # 1. EPS 성장성 (최근 3년 추세: Y3 vs TTM)
    # 데이터가 '-' 인 경우를 대비해 예외처리
    try:
        eps_y3 = float(row.get('EPS_Y3', 0))
        eps_ttm = float(row.get('EPS_TTM', 0))
        if eps_ttm > eps_y3 and eps_y3 != 0:
            score += 30
            reasons.append("✅ EPS 성장세")
    except: pass
    
    # 2. 현금흐름 질 (CFQ)
    try:
        cfq = float(row.get('CFQ_TTM', 0))
        if cfq >= 1.0:
            score += 30
            reasons.append("✅ 현금창출력(CFQ>1)")
    except: pass
    
    # 3. 수익성 (ROE)
    try:
        roe = float(row.get('ROE(%)', 0))
        if roe >= 15:
            score += 20
            reasons.append("✅ 고ROE(15%↑)")
        elif roe < 0:
            score -= 10
            reasons.append("⚠️ 적자지속")
    except: pass

    # 4. 재무 건전성 (DTE)
    try:
        dte = float(row.get('DTE(%)', 1000))
        if dte <= 100:
            score += 20
            reasons.append("✅ 재무안정")
        elif dte > 200:
            score -= 10
            reasons.append("🚨 고부채")
    except: pass

    # 등급 결정
    if score >= 90: grade = "S (강력 매수 후보)"
    elif score >= 70: grade = "A (우량 투자 대상)"
    elif score >= 50: grade = "B (보유 및 관망)"
    else: grade = "C (투자 유의/제외)"
    
    return grade, ", ".join(reasons)

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

        # 기본 데이터 수집
        ttm_dte = info.get("debtToEquity")
        ttm_cr = (info.get("currentRatio") * 100) if info.get("currentRatio") else None
        ttm_opm = (info.get("operatingMargins") * 100) if info.get("operatingMargins") else None
        ttm_roe = (info.get("returnOnEquity") * 100) if info.get("returnOnEquity") else None
        ttm_ocf = info.get("operatingCashflow")
        ttm_fcf = info.get("freeCashflow")
        ttm_net_inc = info.get("netIncomeToCommon")
        total_cash = info.get("totalCash")
        
        runway = round(total_cash / abs(ttm_fcf), 2) if total_cash and ttm_fcf and ttm_fcf < 0 else "Infinite"

        # 시계열 데이터 수집 (Y4 ~ Y1)
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
        fcf_series = history["FCF"] + [ttm_fcf_m]
        stability = (sum(1 for v in fcf_series if v and v > 0) / 5) * 100 if any(v is not None for v in fcf_series) else 0

        # 요약 결과 리턴
        base_results = [
            round(ttm_dte, 2) if ttm_dte else None,
            round(ttm_cr, 2) if ttm_cr else None,
            round(ttm_opm, 2) if ttm_opm else None,
            round(ttm_roe, 2) if ttm_roe else None,
            runway, round(total_cash/1_000_000, 2) if total_cash else None,
            ttm_fcf_m, stability, round(ttm_ocf/1_000_000, 2) if ttm_ocf else None,
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
            flattened_history.extend(history[key] + [ttm_vals_map.get(key)])

        return base_results + flattened_history
    except:
        return [None] * (13 + 40)

# --- [UI] ---
st.set_page_config(page_title="Stock Master Eval", layout="wide")
st.title("📊 재무 기반 투자 등급 자동 평가 시스템")

raw = st.sidebar.text_area("티커 입력")
tickers = [t.strip().upper() for t in raw.split('\n') if t.strip()]

if tickers and st.sidebar.button("분석 시작"):
    results = []
    
    # 1. 헤더 설정
    base_cols = [
        'ticker', 'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Runway(Y)', 
        'TotalCash(M$)', 'FCF(M$)', 'FCF_Stability(%)', 'OCF(M$)', 
        'PBR', 'BPS', 'PER', 'EPS', 'Updated'
    ]
    metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
    history_cols = [f"{m}_{y}" for m in metrics for y in ["Y4", "Y3", "Y2", "Y1", "TTM"]]
    final_cols = base_cols + history_cols

    for t in tickers:
        data = get_extended_financials(t)
        row = [t] + data[:13] + [datetime.now().strftime('%H:%M:%S')] + data[13:]
        results.append(row)
        time.sleep(0.5)

    res_df = pd.DataFrame(results, columns=final_cols)

    # 2. 투자 등급 평가 칼럼 추가 (작성해주신 로직 적용)
    eval_list = []
    for _, row in res_df.iterrows():
        grade, reason = evaluate_investment(row)
        eval_list.append({'최종 등급': grade, '평가 근거': reason})
    
    eval_df = pd.DataFrame(eval_list)
    # 데이터프레임 맨 앞에 티커와 평가 결과 배치
    display_df = pd.concat([res_df[['ticker']], eval_df, res_df.drop(columns=['ticker'])], axis=1).fillna("-")

    # 3. 결과 출력
    st.subheader("🎯 종목별 종합 투자 평가")
    st.dataframe(display_df, use_container_width=True)
