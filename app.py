import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from urllib.parse import quote

# --- [함수] 투자 스타일 및 등급 평가 (기존 로직 유지) ---
def evaluate_investment_by_style(row):
    try:
        def to_f(val):
            try: return float(val) if val not in [None, '-', 'Infinite'] else 0.0
            except: return 0.0

        per, pbr, roe = to_f(row.get('PER')), to_f(row.get('PBR')), to_f(row.get('ROE(%)'))
        dte, cfq = to_f(row.get('DTE(%)')), to_f(row.get('CFQ_TTM'))
        eps_y3, eps_ttm = to_f(row.get('EPS_Y3')), to_f(row.get('EPS_TTM'))
        eps_g = ((eps_ttm - eps_y3) / abs(eps_y3) * 100) if eps_y3 != 0 else 0

        # 스타일 분류
        if per > 20 or eps_g > 15 or pbr > 3.0:
            style, is_growth = "성장주(Growth)", True
        else:
            style, is_growth = "가치주(Value)", False

        score = 0
        reasons = []

        if is_growth:
            if eps_g >= 20: score += 30; reasons.append("📈 고성장")
            if roe >= 15: score += 30; reasons.append("🚀 고효율")
            if cfq >= 1.0: score += 20; reasons.append("✅ 현금양호")
            if dte <= 100: score += 20; reasons.append("🛡️ 재무안전")
        else:
            if pbr <= 1.2: score += 30; reasons.append("💎 저평가")
            if cfq >= 1.2: score += 30; reasons.append("💰 현금왕")
            if roe >= 10: score += 20; reasons.append("✅ 꾸준함")
            if dte <= 100: score += 20; reasons.append("🛡️ 재무안정")

        if dte > 250: score -= 20; reasons.append("🚨 고부채")
        if roe < 0: score -= 30; reasons.append("⚠️ 적자")

        grade = "S" if score >= 85 else "A" if score >= 65 else "B" if score >= 45 else "C"
        return style, grade, ", ".join(reasons)
    except:
        return "미분류", "등급외", "데이터 부족"

# --- [함수] 재무 데이터 추출 (지연 시간 적용) ---
def get_extended_financials(ticker_symbol):
    try:
        symbol = ticker_symbol.upper().strip()
        ticker = yf.Ticker(symbol)
        
        # 1. info 데이터 가져오기 (시간 지연 유도 효과)
        info = ticker.info
        if not info: return [None] * 53

        # 2. 재무제표 가져오기 (에러 방지용)
        fin = ticker.financials
        bs = ticker.balance_sheet
        cf = ticker.cashflow

        # 데이터가 하나라도 비어있으면 중단
        if fin.empty or bs.empty or cf.empty:
            return [None] * 53

        def get_val(df, label, idx):
            try: return df.loc[label].iloc[idx]
            except: return None

        # 기본 지표 추출
        ttm_dte = info.get("debtToEquity")
        ttm_cr = (info.get("currentRatio") * 100) if info.get("currentRatio") else None
        ttm_opm = (info.get("operatingMargins") * 100) if info.get("operatingMargins") else None
        ttm_roe = (info.get("returnOnEquity") * 100) if info.get("returnOnEquity") else None
        ttm_ocf = info.get("operatingCashflow")
        ttm_fcf = info.get("freeCashflow")
        ttm_net_inc = info.get("netIncomeToCommon")
        total_cash = info.get("totalCash")
        
        runway = round(total_cash / abs(ttm_fcf), 2) if total_cash and ttm_fcf and ttm_fcf < 0 else "Infinite"

        # 시계열 추이 (Y4 -> TTM)
        metrics_order = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history = {m: [None]*4 for m in metrics_order}
        num_years = min(len(fin.columns), 4)

        for i in range(num_years):
            idx = 3 - i 
            net_inc = get_val(fin, 'Net Income', i)
            equity = get_val(bs, 'Total Equity Gross Minority Interest', i)
            ocf_val = get_val(cf, 'Operating Cash Flow', i)
            cap_ex = get_val(cf, 'Capital Expenditure', i)
            fcf_val = (ocf_val + cap_ex) if ocf_val is not None and cap_ex is not None else None
            
            history["DTE"][idx] = round((get_val(bs, 'Total Liabilities Net Minority Interest', i)/equity*100), 2) if equity else None
            history["CR"][idx] = round((get_val(bs, 'Current Assets', i)/get_val(bs, 'Current Liabilities', i)*100), 2) if get_val(bs, 'Current Assets', i) else None
            history["OPM"][idx] = round((get_val(fin, 'Operating Income', i)/get_val(fin, 'Total Revenue', i)*100), 2) if get_val(fin, 'Operating Income', i) else None
            history["ROE"][idx] = round((net_inc/equity*100), 2) if equity else None
            history["OCF"][idx] = round(ocf_val/1_000_000, 2) if ocf_val else None
            history["EPS"][idx] = round(get_val(fin, 'Basic EPS', i), 2)
            history["CFQ"][idx] = round(ocf_val/net_inc, 2) if net_inc and net_inc != 0 else None
            history["FCF"][idx] = round(fcf_val/1_000_000, 2) if fcf_val else None

        ttm_fcf_m = round(ttm_fcf/1_000_000, 2) if ttm_fcf else None
        stability = (sum(1 for v in history["FCF"] + [ttm_fcf_m] if v and v > 0) / 5) * 100

        base_results = [
            round(ttm_dte, 2) if ttm_dte else None, round(ttm_cr, 2) if ttm_cr else None,
            round(ttm_opm, 2) if ttm_opm else None, round(ttm_roe, 2) if ttm_roe else None,
            runway, round(total_cash / 1_000_000, 2) if total_cash else None,
            ttm_fcf_m, stability, round(ttm_ocf / 1_000_000, 2) if ttm_ocf else None,
            round(info.get("priceToBook"), 2), round(info.get("bookValue"), 2),
            round(info.get("trailingPE"), 2), round(info.get("trailingEps"), 2)
        ]

        ttm_cfq = round(ttm_ocf/ttm_net_inc, 2) if ttm_ocf and ttm_net_inc and ttm_net_inc != 0 else None
        ttm_vals_map = {"DTE": base_results[0], "CR": base_results[1], "OPM": base_results[2], "ROE": base_results[3], "OCF": base_results[8], "EPS": base_results[12], "CFQ": ttm_cfq, "FCF": ttm_fcf_m}
        
        flattened_history = []
        for key in metrics_order:
            flattened_history.extend(history[key] + [ttm_vals_map.get(key)])

        return base_results + flattened_history
    except:
        return [None] * 53

# --- [UI] Streamlit ---
st.set_page_config(page_title="Stable Stock Analyzer", layout="wide")
st.title("🛡️ 안전 재무 분석 시스템 (데이터 누락 방지)")

# 사이드바 구조 유지
st.sidebar.header("📥 데이터 소스")
method = st.sidebar.radio("방식", ("텍스트 붙여넣기", "구글 스프레드시트", "CSV 파일 업로드"))
tickers = []

if method == "텍스트 붙여넣기":
    raw = st.sidebar.text_area("티커 입력 (줄바꿈 구분)")
    if raw: tickers = [t.strip().upper() for t in raw.split('\n') if t.strip()]
elif method == "구글 스프레드시트":
    try:
        sid, sname = st.secrets["GOOGLE_SHEET_ID"], st.secrets["GOOGLE_SHEET_NAME"]
        url = f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={quote(sname)}"
        gs_df = pd.read_csv(url); t_col = st.sidebar.selectbox("티커 컬럼", gs_df.columns)
        tickers = gs_df[t_col].dropna().astype(str).tolist()
    except: st.sidebar.error("시트 연결 오류")
elif method == "CSV 파일 업로드":
    up = st.sidebar.file_uploader("CSV", type=["csv"])
    if up:
        df = pd.read_csv(up); t_col = st.sidebar.selectbox("티커 컬럼", df.columns)
        tickers = df[t_col].dropna().astype(str).tolist()

if tickers:
    if st.button("🚀 안정적 전수 분석 시작"):
        prog = st.progress(0); status = st.empty(); results = []
        
        base_cols = ['ticker', 'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Runway(Y)', 'TotalCash(M$)', 'FCF(M$)', 'FCF_Stability(%)', 'OCF(M$)', 'PBR', 'BPS', 'PER', 'EPS', 'Updated']
        metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history_cols = [f"{m}_{y}" for m in metrics for y in ["Y4", "Y3", "Y2", "Y1", "TTM"]]
        final_cols = base_cols + history_cols

        for idx, symbol in enumerate(tickers):
            status.info(f"데이터 요청 중: {symbol} ({idx+1}/{len(tickers)})")
            
            # --- [핵심] 지연 시간 부여 ---
            # 야후 서버 부하를 줄이기 위해 루프당 1.2초 대기
            time.sleep(1.2) 
            
            data = get_extended_financials(symbol)
            row = [symbol] + data[:13] + [datetime.now().strftime('%H:%M:%S')] + data[13:]
            results.append(row)
            prog.progress((idx+1)/len(tickers))

        res_df = pd.DataFrame(results, columns=final_cols)

        # 스타일 평가 적용
        eval_list = []
        for _, row in res_df.iterrows():
            style, grade, reason = evaluate_investment_by_style(row)
            eval_list.append({'스타일': style, '등급': grade, '포인트': reason})
        
        eval_df = pd.DataFrame(eval_list)
        final_display_df = pd.concat([res_df[['ticker']], eval_df, res_df.drop(columns=['ticker'])], axis=1).fillna("-")

        status.success("✅ 분석 완료!")
        st.dataframe(final_display_df, use_container_width=True)
        st.download_button("📥 결과 저장", final_display_df.to_csv(index=False).encode('utf-8'), "stock_report.csv")
