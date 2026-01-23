import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from urllib.parse import quote

# --- [함수] EPS 성장률 및 상태 계산 (고도화) ---
def calculate_eps_growth(current, past):
    """
    턴어라운드 및 역성장을 구분하여 성장률 계산
    """
    if not isinstance(current, (int, float)) or not isinstance(past, (int, float)) or past == 0:
        return None, "데이터 부족"
    
    growth = ((current - past) / abs(past)) * 100
    
    if past < 0 and current > 0:
        status = "턴어라운드(흑자전환)"
    elif past < 0 and current < 0:
        status = "적자지속" if current < past else "적자축소"
    elif growth > 0:
        status = "성장"
    else:
        status = "역성장"
        
    return growth, status

# --- [함수] 주식 유형 분류 ---
def classify_stock_type(row):
    per = row.get('PER')
    pbr = row.get('PBR')
    eps_y3 = row.get('EPS_Y3')
    eps_ttm = row.get('EPS_TTM')
    
    eps_growth, status = calculate_eps_growth(eps_ttm, eps_y3)
    
    is_low_valuation = (0 < per < 15 and pbr < 1.5) if isinstance(per, (int, float)) and isinstance(pbr, (int, float)) else False
    is_high_growth = (eps_growth and (eps_growth > 15 or status == "턴어라운드(흑자전환)"))
    
    if is_high_growth and not is_low_valuation: return "성장주", eps_growth
    elif is_low_valuation and not is_high_growth: return "가치주", eps_growth
    elif is_high_growth and is_low_valuation: return "혼합형", eps_growth
    else: return "중립", eps_growth

# --- [함수] 가치주 평가 로직 ---
def evaluate_value_stock(row):
    score = 0
    reasons = []
    try:
        # 1. 저평가 (35점)
        per, pbr = row.get('PER'), row.get('PBR')
        if isinstance(per, (int, float)) and 0 < per < 12: score += 20; reasons.append("✅ 저PER(12미만)")
        if isinstance(pbr, (int, float)) and pbr < 1.0: score += 15; reasons.append("✅ PBR 1배 미만")

        # 2. 배당 (15점)
        div = row.get('Div_Yield(%)')
        if isinstance(div, (int, float)) and div >= 3: score += 15; reasons.append(f"💰 고배당 ({div}%)")

        # 3. 재무건전성 (30점)
        dte, cr = row.get('DTE(%)'), row.get('CR(%)')
        if isinstance(dte, (int, float)) and dte <= 70: score += 20; reasons.append("✅ 낮은 부채비율")
        if isinstance(cr, (int, float)) and cr >= 150: score += 10; reasons.append("✅ 유동성 양호")

        # 4. 수익성(20점)
        roe = row.get('ROE(%)')
        if isinstance(roe, (int, float)) and roe >= 10: score += 20; reasons.append("✅ 수익성(ROE) 우수")
    except: pass

    if score >= 80: grade = "S (초우량 가치)"
    elif score >= 60: grade = "A (우량 가치)"
    else: grade = "B (보통 이하)"
    return grade, score, ", ".join(reasons)

# --- [함수] 성장주 평가 로직 ---
def evaluate_growth_stock(row):
    score = 0
    reasons = []
    try:
        # 1. 성장성 (40점)
        eps_y3, eps_ttm = row.get('EPS_Y3'), row.get('EPS_TTM')
        growth, status = calculate_eps_growth(eps_ttm, eps_y3)
        if status == "턴어라운드(흑자전환)": score += 40; reasons.append("🚀 흑자전환 성공")
        elif growth and growth > 25: score += 30; reasons.append(f"✅ 고속성장({growth:.1f}%)")

        # 2. PEG (20점)
        per = row.get('PER')
        if isinstance(per, (int, float)) and growth and growth > 0:
            peg = per / growth
            if peg < 1.2: score += 20; reasons.append(f"💎 저평가 성장(PEG {peg:.2f})")

        # 3. 수익성/현금질 (40점)
        roe, cfq = row.get('ROE(%)'), row.get('CFQ_TTM')
        if isinstance(roe, (int, float)) and roe >= 15: score += 20; reasons.append("✅ 고수익성(ROE)")
        if isinstance(cfq, (int, float)) and cfq >= 1.0: score += 20; reasons.append("✅ 현금창출력 우수")
    except: pass

    if score >= 80: grade = "S (스타 종목)"
    elif score >= 60: grade = "A (우량 성장)"
    else: grade = "B (성장 둔화)"
    return grade, score, ", ".join(reasons)

# --- [함수] 통합 평가 및 데이터 추출 ---
def get_extended_financials(ticker_symbol):
    try:
        symbol = ticker_symbol.upper().strip()
        ticker = yf.Ticker(symbol)
        info = ticker.info
        fin, bs, cf = ticker.financials, ticker.balance_sheet, ticker.cashflow

        def get_val(df, label, idx):
            try: return df.loc[label].iloc[idx]
            except: return None

        raw_dte = info.get("debtToEquity")
        ttm_dte = (raw_dte if raw_dte and raw_dte > 5 else raw_dte * 100) if raw_dte else None
        ttm_cr = (info.get("currentRatio") * 100) if info.get("currentRatio") else None
        ttm_opm = (info.get("operatingMargins") * 100) if info.get("operatingMargins") else None
        ttm_roe = (info.get("returnOnEquity") * 100) if info.get("returnOnEquity") else None
        div_yield = (info.get("dividendYield") * 100) if info.get("dividendYield") else 0.0
        ttm_fcf = info.get("freeCashflow")
        ttm_ocf, ttm_net_inc = info.get("operatingCashflow"), info.get("netIncomeToCommon")

        metrics_order = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history = {m: [None]*4 for m in metrics_order}
        for i in range(min(len(fin.columns), 4)):
            idx = 3 - i
            ni, eq = get_val(fin, 'Net Income', i), get_val(bs, 'Total Equity Gross Minority Interest', i)
            ocf_v, ce = get_val(cf, 'Operating Cash Flow', i), get_val(cf, 'Capital Expenditure', i)
            history["DTE"][idx] = round((get_val(bs, 'Total Liabilities Net Minority Interest', i)/eq*100), 2) if eq else None
            history["ROE"][idx] = round((ni/eq*100), 2) if ni and eq else None
            history["EPS"][idx] = round(get_val(fin, 'Basic EPS', i), 2) if get_val(fin, 'Basic EPS', i) else None
            history["CFQ"][idx] = round(ocf_v/ni, 2) if ocf_v and ni and ni != 0 else None
            history["FCF"][idx] = round((ocf_v + ce)/1_000_000, 2) if ocf_v and ce else None

        base_res = [round(ttm_dte, 2) if ttm_dte else None, round(ttm_cr, 2) if ttm_cr else None, round(ttm_opm, 2) if ttm_opm else None, round(ttm_roe, 2) if ttm_roe else None, round(div_yield, 2), round(ttm_fcf/1_000_000, 2) if ttm_fcf else None, round(info.get("priceToBook"), 2) if info.get("priceToBook") else None, round(info.get("trailingPE"), 2) if info.get("trailingPE") else None, round(info.get("trailingEps"), 2) if info.get("trailingEps") else None]
        ttm_vals = {"DTE": base_res[0], "ROE": base_res[3], "EPS": base_res[8], "CFQ": round(ttm_ocf/ttm_net_inc, 2) if ttm_ocf and ttm_net_inc else None, "FCF": base_res[5]}
        
        history_flat = []
        for m in metrics_order: history_flat.extend(history[m] + [ttm_vals.get(m)])
        return base_res + history_flat
    except: return [None]*49

# --- [UI] 메인 로직 ---
st.set_page_config(page_title="Stock Grading V3 Full", layout="wide")
st.title("🚀 통합 주식 등급 평가 시스템 V3")

st.sidebar.header("📥 입력 방식 설정")
method = st.sidebar.radio("데이터 소스", ("텍스트 입력", "구글 시트 연동", "CSV 파일 업로드"))
tickers = []

if method == "텍스트 입력":
    raw = st.sidebar.text_area("티커를 입력하세요 (예: AAPL, TSLA)")
    if raw: tickers = [t.strip().upper() for t in raw.replace(',', '\n').split('\n') if t.strip()]
elif method == "구글 시트 연동":
    try:
        sid, sname = st.secrets["GOOGLE_SHEET_ID"], st.secrets["GOOGLE_SHEET_NAME"]
        url = f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={quote(sname)}"
        gs_df = pd.read_csv(url)
        t_col = st.sidebar.selectbox("티커 컬럼 선택", gs_df.columns)
        tickers = gs_df[t_col].dropna().astype(str).tolist()
    except Exception as e: st.sidebar.error("구글 시트 연결 실패. Secrets 설정을 확인하세요.")
elif method == "CSV 파일 업로드":
    up = st.sidebar.file_uploader("CSV 파일 선택", type=["csv"])
    if up:
        df = pd.read_csv(up); t_col = st.sidebar.selectbox("티커 컬럼 선택", df.columns)
        tickers = df[t_col].dropna().astype(str).tolist()

if tickers and st.button("🔍 전수 분석 시작"):
    prog = st.progress(0); status = st.empty(); results = []
    base_cols = ['ticker', 'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Div_Yield(%)', 'FCF(M$)', 'PBR', 'PER', 'EPS', 'Updated']
    metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
    history_cols = [f"{m}_{y}" for m in metrics for y in ["Y4", "Y3", "Y2", "Y1", "TTM"]]
    
    for idx, s in enumerate(tickers):
        status.info(f"분석 중: {s} ({idx+1}/{len(tickers)})")
        data = get_extended_financials(s)
        row = [s] + data[:9] + [datetime.now().strftime('%H:%M')] + data[9:]
        results.append(row)
        prog.progress((idx+1)/len(tickers)); time.sleep(0.2)

    res_df = pd.DataFrame(results, columns=base_cols + history_cols)
    eval_list = []
    for _, r in res_df.iterrows():
        stype, eps_g = classify_stock_type(r)
        grade, score, reason = (evaluate_growth_stock(r) if stype == "성장주" else evaluate_value_stock(r))
        eval_list.append({"유형": stype, "등급": grade, "점수": score, "성장률": f"{eps_g:.1f}%" if eps_g else "N/A", "주요지표": reason})
    
    final_df = pd.concat([res_df[['ticker']], pd.DataFrame(eval_list), res_df.drop(columns=['ticker'])], axis=1).fillna("-")
    st.success("✅ 분석 완료!"); st.dataframe(final_df, use_container_width=True)
    st.download_button("📥 결과 다운로드", final_df.to_csv(index=False).encode('utf-8'), "stock_report_v3.csv")
