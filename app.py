import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from urllib.parse import quote

# --- [함수] 데이터 누락 방지를 위한 레이블 매칭 ---
def safe_get_financials(ticker_obj):
    fin = ticker_obj.financials
    bs = ticker_obj.balance_sheet
    cf = ticker_obj.cashflow
    
    def find_label(df, candidates):
        if df is None or df.empty: return None
        for c in candidates:
            if c in df.index: return df.loc[c]
        return None

    return {
        "net_income": find_label(fin, ['Net Income', 'Net Income Common Stockholders', 'Net Income From Continuing Operation Net Minority Interest']),
        "total_equity": find_label(bs, ['Total Equity Gross Minority Interest', 'Stockholders Equity', 'Total Stockholders Equity']),
        "total_liabilities": find_label(bs, ['Total Liabilities Net Minority Interest', 'Total Liabilities']),
        "ocf": find_label(cf, ['Operating Cash Flow', 'Cash Flow From Continuing Operating Activities']),
        "capex": find_label(cf, ['Capital Expenditure', 'Investing Cash Flow']),
        "revenue": find_label(fin, ['Total Revenue', 'Total Operating Revenue']),
        "eps": find_label(fin, ['Basic EPS', 'Diluted EPS'])
    }

# --- [함수] EPS 성장률 및 상태 계산 (고도화) ---
def calculate_eps_growth(current, past):
    if not isinstance(current, (int, float)) or not isinstance(past, (int, float)) or past == 0:
        return None, "데이터 부족"
    
    growth = ((current - past) / abs(past)) * 100
    if past < 0 and current > 0: status = "턴어라운드(흑자전환)"
    elif past < 0 and current < 0: status = "적자지속" if current < past else "적자축소"
    elif growth > 0: status = "성장"
    else: status = "역성장"
    return growth, status

# --- [함수] 주식 유형 분류 ---
def classify_stock_type(row):
    per, pbr = row.get('PER'), row.get('PBR')
    eps_y3, eps_ttm = row.get('EPS_Y3'), row.get('EPS_TTM')
    growth, status = calculate_eps_growth(eps_ttm, eps_y3)
    
    is_low_val = (0 < per < 15 and pbr < 1.5) if isinstance(per, (int, float)) and isinstance(pbr, (int, float)) else False
    is_high_growth = (growth and (growth > 15 or status == "턴어라운드(흑자전환)"))
    
    if is_high_growth and not is_low_val: return "성장주", growth
    elif is_low_val and not is_high_growth: return "가치주", growth
    elif is_high_growth and is_low_val: return "혼합형", growth
    else: return "중립", growth

# --- [함수] 가치주/성장주별 정밀 평가 ---
def evaluate_value_stock(row):
    score, reasons = 0, []
    try:
        # 저평가(35), 배당(15), 건전성(30), 수익성(20)
        per, pbr = row.get('PER'), row.get('PBR')
        if isinstance(per, (int, float)) and 0 < per < 12: score += 20; reasons.append("✅ 저PER")
        if isinstance(pbr, (int, float)) and pbr < 1.0: score += 15; reasons.append("✅ PBR < 1.0")
        
        div = row.get('Div_Yield(%)')
        if isinstance(div, (int, float)) and div >= 3: score += 15; reasons.append(f"💰 고배당({div}%)")
        
        dte, cr = row.get('DTE(%)'), row.get('CR(%)')
        if isinstance(dte, (int, float)) and dte <= 70: score += 20; reasons.append("✅ 낮은 부채비율")
        if isinstance(cr, (int, float)) and cr >= 150: score += 10; reasons.append("✅ 유동성 양호")
        
        roe = row.get('ROE(%)')
        if isinstance(roe, (int, float)) and roe >= 10: score += 20; reasons.append("✅ ROE 양호")
    except: pass
    return "S" if score >= 80 else "A" if score >= 60 else "B", score, ", ".join(reasons)

def evaluate_growth_stock(row):
    score, reasons = 0, []
    try:
        # 성장성(40), PEG(20), 수익성향상(20), 현금흐름질(20)
        eps_y3, eps_ttm = row.get('EPS_Y3'), row.get('EPS_TTM')
        growth, status = calculate_eps_growth(eps_ttm, eps_y3)
        if status == "턴어라운드(흑자전환)": score += 40; reasons.append("🚀 흑자전환 성공")
        elif growth and growth > 25: score += 35; reasons.append(f"✅ 고속성장({growth:.1f}%)")
        
        per = row.get('PER')
        if isinstance(per, (int, float)) and growth and growth > 0:
            peg = per / growth
            if peg < 1.2: score += 20; reasons.append(f"💎 저평가 성장(PEG {peg:.2f})")
            
        roe, cfq = row.get('ROE(%)'), row.get('CFQ_TTM')
        if isinstance(roe, (int, float)) and roe >= 15: score += 20; reasons.append("✅ 고수익성(ROE)")
        if isinstance(cfq, (int, float)) and cfq >= 1.0: score += 20; reasons.append("✅ 이익의 질 우수")
    except: pass
    return "S" if score >= 80 else "A" if score >= 60 else "B", score, ", ".join(reasons)

# --- [함수] 재무 데이터 추출 (방어적 코드 적용) ---
def get_extended_financials(ticker_symbol):
    try:
        symbol = ticker_symbol.upper().strip()
        ticker = yf.Ticker(symbol)
        info = ticker.info
        f_data = safe_get_financials(ticker)

        # 1. 핵심 지표 계산 (Info 누락 대비)
        price = info.get("currentPrice") or info.get("previousClose")
        eps_ttm = info.get("trailingEps")
        per = info.get("trailingPE") or (price / eps_ttm if price and eps_ttm else None)
        pbr = info.get("priceToBook")
        
        raw_dte = info.get("debtToEquity")
        if raw_dte is None and f_data["total_liabilities"] is not None:
            raw_dte = (f_data["total_liabilities"].iloc[0] / f_data["total_equity"].iloc[0]) * 100
        ttm_dte = (raw_dte if raw_dte and raw_dte > 5 else raw_dte * 100) if raw_dte else None

        # 2. 히스토리 데이터 (최대 4년)
        metrics_order = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history = {m: [None]*4 for m in metrics_order}
        num_years = min(len(f_data["net_income"]), 4) if f_data["net_income"] is not None else 0

        for i in range(num_years):
            idx = 3 - i
            try:
                ni, eq = f_data["net_income"].iloc[i], f_data["total_equity"].iloc[i]
                ocf_v, ce = f_data["ocf"].iloc[i], f_data["capex"].iloc[i] if f_data["capex"] is not None else 0
                history["DTE"][idx] = round((f_data["total_liabilities"].iloc[i]/eq*100), 2) if eq else None
                history["ROE"][idx] = round((ni/eq*100), 2) if ni and eq else None
                history["EPS"][idx] = round(f_data["eps"].iloc[i], 2) if f_data["eps"] is not None else None
                history["CFQ"][idx] = round(ocf_v/ni, 2) if ocf_v and ni != 0 else None
                history["FCF"][idx] = round((ocf_v + ce)/1_000_000, 2) if ocf_v is not None else None
            except: continue

        # 3. TTM 값 구성
        base_res = [
            round(ttm_dte, 2) if ttm_dte else None, 
            round(info.get("currentRatio")*100, 2) if info.get("currentRatio") else None,
            round(info.get("operatingMargins")*100, 2) if info.get("operatingMargins") else None,
            round(info.get("returnOnEquity")*100, 2) if info.get("returnOnEquity") else None,
            round(info.get("dividendYield")*100, 2) if info.get("dividendYield") else 0.0,
            round(info.get("freeCashflow")/1_000_000, 2) if info.get("freeCashflow") else None,
            round(pbr, 2) if pbr else None, round(per, 2) if per else None, round(eps_ttm, 2) if eps_ttm else None
        ]
        
        flattened = []
        ttm_vals = {"DTE": base_res[0], "ROE": base_res[3], "EPS": base_res[8]}
        for m in metrics_order:
            flattened.extend(history[m] + [ttm_vals.get(m)])
        return base_res + flattened
    except: return [None]*49

# --- [UI] Streamlit 메인 로직 ---
st.set_page_config(page_title="Stock Grading System V3", layout="wide")
st.title("📊 통합 주식 평가 시스템 V3 (최적화 버전)")

st.sidebar.header("📥 데이터 입력 방식")
method = st.sidebar.radio("소스 선택", ("텍스트 입력", "구글 시트 연동", "CSV 업로드"))
tickers = []

if method == "텍스트 입력":
    raw = st.sidebar.text_area("티커를 입력하세요 (AAPL, TSLA 등)")
    if raw: tickers = [t.strip().upper() for t in raw.replace(',', '\n').split('\n') if t.strip()]
elif method == "구글 시트 연동":
    try:
        sid, sname = st.secrets["GOOGLE_SHEET_ID"], st.secrets["GOOGLE_SHEET_NAME"]
        url = f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={quote(sname)}"
        gs_df = pd.read_csv(url)
        t_col = st.sidebar.selectbox("티커 컬럼 선택", gs_df.columns)
        tickers = gs_df[t_col].dropna().astype(str).tolist()
    except: st.sidebar.error("구글 시트 연결 실패. Secrets 설정을 확인하세요.")
elif method == "CSV 업로드":
    up = st.sidebar.file_uploader("CSV 파일 선택", type=["csv"])
    if up:
        df = pd.read_csv(up); t_col = st.sidebar.selectbox("티커 컬럼 선택", df.columns)
        tickers = df[t_col].dropna().astype(str).tolist()

if tickers and st.button("🔍 전수 분석 시작"):
    prog = st.progress(0); status = st.empty(); results = []
    base_cols = ['ticker', 'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Div_Yield(%)', 'FCF(M$)', 'PBR', 'PER', 'EPS', 'Updated']
    metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
    hist_cols = [f"{m}_{y}" for m in metrics for y in ["Y4", "Y3", "Y2", "Y1", "TTM"]]

    for idx, s in enumerate(tickers):
        status.info(f"분석 중: {s} ({idx+1}/{len(tickers)})")
        data = get_extended_financials(s)
        row = [s] + data[:9] + [datetime.now().strftime('%H:%M')] + data[9:]
        results.append(row)
        
        # --- 지연 시간(Sleep) 최적화 ---
        # 0.5초는 yfinance API가 빈 값을 반환하거나 차단되는 것을 방지하는 가장 안정적인 시간입니다.
        time.sleep(0.5) 
        prog.progress((idx+1)/len(tickers))

    res_df = pd.DataFrame(results, columns=base_cols + hist_cols)
    eval_list = []
    for _, r in res_df.iterrows():
        stype, eps_g = classify_stock_type(r)
        grade, score, reason = (evaluate_growth_stock(r) if stype == "성장주" else evaluate_value_stock(r))
        eval_list.append({"유형": stype, "등급": grade, "점수": score, "성장률": f"{eps_g:.1f}%" if eps_g else "N/A", "주요지표": reason})
    
    final_df = pd.concat([res_df[['ticker']], pd.DataFrame(eval_list), res_df.drop(columns=['ticker'])], axis=1).fillna("-")
    st.success("✅ 분석 완료!"); st.dataframe(final_df, use_container_width=True)
    st.download_button("📥 결과 다운로드", final_df.to_csv(index=False).encode('utf-8'), "stock_report_v3.csv")
