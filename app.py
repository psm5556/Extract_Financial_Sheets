import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from urllib.parse import quote

# --- [함수] 투자 등급 평가 로직 (가치주/성장주 세분화) ---
def evaluate_investment(row):
    score = 0
    reasons = []
    
    # 데이터 추출 및 전처리
    per = row.get('PER')
    pbr = row.get('PBR')
    roe = row.get('ROE(%)')
    dte = row.get('DTE(%)')
    cfq = row.get('CFQ_TTM')
    eps_y3 = row.get('EPS_Y3')
    eps_ttm = row.get('EPS_TTM')
    rev_y1 = row.get('REV_Y1')  # 매출 데이터 (추가됨)
    rev_ttm = row.get('REV_TTM')
    
    # 0. 유형 판별 (기본값: 혼합형)
    # 성장주 조건: 높은 PER/PBR 혹은 높은 매출 성장
    is_growth = False
    if (isinstance(rev_ttm, (int, float)) and isinstance(rev_y1, (int, float)) and rev_ttm > rev_y1 * 1.15) or (isinstance(per, (int, float)) and per > 25):
        is_growth = True
        stock_type = "성장주(Growth)"
    else:
        stock_type = "가치주(Value)"

    try:
        if is_growth:
            # --- [성장주 평가 로직] ---
            # 1. EPS 성장성 (40점)
            if isinstance(eps_y3, (int, float)) and isinstance(eps_ttm, (int, float)) and eps_ttm > eps_y3:
                score += 40
                reasons.append("🚀 강력한 이익 성장세")
            
            # 2. 수익성 (ROE) (30점)
            if isinstance(roe, (int, float)) and roe >= 15:
                score += 30
                reasons.append("📈 고수익성(ROE 15%↑)")
            
            # 3. 매출 성장률 (20점)
            if isinstance(rev_ttm, (int, float)) and isinstance(rev_y1, (int, float)) and rev_ttm > rev_y1 * 1.1:
                score += 20
                reasons.append("📊 매출 확대 중")
            
            # 4. 재무 안전성 (10점)
            if isinstance(dte, (int, float)) and dte <= 150:
                score += 10
                reasons.append("🛡️ 적정 부채 수준")

        else:
            # --- [가치주 평가 로직] ---
            # 1. 저평가 지표 (PER/PBR) (40점)
            if isinstance(per, (int, float)) and 0 < per <= 15:
                score += 20
                reasons.append("💎 낮은 PER (저평가)")
            if isinstance(pbr, (int, float)) and 0 < pbr <= 1.2:
                score += 20
                reasons.append("🏢 자산 가치 우수 (저PBR)")
            
            # 2. 현금흐름 질 (CFQ) (30점)
            if isinstance(cfq, (int, float)) and cfq >= 1.0:
                score += 30
                reasons.append("💵 실제 현금 창출력 우수")
            
            # 3. 재무 건전성 (DTE) (20점)
            if isinstance(dte, (int, float)) and dte <= 100:
                score += 20
                reasons.append("🏦 매우 탄탄한 재무구조")
            
            # 4. 안정성 (ROE) (10점)
            if isinstance(roe, (int, float)) and roe >= 8:
                score += 10
                reasons.append("👍 꾸준한 수익성")

    except Exception:
        pass

    # 등급 결정
    if score >= 85: grade = "S (강력 추천)"
    elif score >= 65: grade = "A (매수 고려)"
    elif score >= 45: grade = "B (보유/관망)"
    else: grade = "C (투자 유의)"
    
    return stock_type, grade, ", ".join(reasons) if reasons else "데이터 부족으로 평가 제한"

# --- [함수] 재무 데이터 추출 로직 (매출 데이터 추가) ---
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

        # 1. TTM 기본 데이터
        ttm_dte = info.get("debtToEquity")
        ttm_cr = (info.get("currentRatio") * 100) if info.get("currentRatio") else None
        ttm_opm = (info.get("operatingMargins") * 100) if info.get("operatingMargins") else None
        ttm_roe = (info.get("returnOnEquity") * 100) if info.get("returnOnEquity") else None
        ttm_ocf = info.get("operatingCashflow")
        ttm_fcf = info.get("freeCashflow")
        ttm_net_inc = info.get("netIncomeToCommon")
        ttm_rev = info.get("totalRevenue")
        total_cash = info.get("totalCash")
        
        runway = round(total_cash / abs(ttm_fcf), 2) if total_cash and ttm_fcf and ttm_fcf < 0 else "Infinite"

        # 2. 5개년 추이 수집 (REV 추가)
        metrics_order = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF", "REV"]
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
            history["REV"][idx] = round(get_val(fin, 'Total Revenue', i)/1_000_000, 2) if get_val(fin, 'Total Revenue', i) else None

        ttm_fcf_m = round(ttm_fcf/1_000_000, 2) if ttm_fcf else None
        fcf_series = history["FCF"] + [ttm_fcf_m]
        stability = (sum(1 for v in fcf_series if v is not None and v > 0) / 5) * 100 if any(v is not None for v in fcf_series) else 0

        # 요약 결과
        base_results = [
            round(ttm_dte, 2) if ttm_dte is not None else None,
            round(ttm_cr, 2) if ttm_cr is not None else None,
            round(ttm_opm, 2) if ttm_opm is not None else None,
            round(ttm_roe, 2) if ttm_roe is not None else None,
            runway,
            round(total_cash / 1_000_000, 2) if total_cash else None,
            ttm_fcf_m,
            stability,
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
            "FCF": ttm_fcf_m,
            "REV": round(ttm_rev/1_000_000, 2) if ttm_rev else None
        }
        
        flattened_history = []
        for key in metrics_order:
            flattened_history.extend(history[key] + [ttm_vals_map.get(key)])

        return base_results + flattened_history
    except Exception:
        return [None] * (13 + 45) # 9개 지표 * 5개 시점 = 45

# --- [UI] Streamlit 설정 ---
st.set_page_config(page_title="Stock Grading System", layout="wide")
st.title("📊 유형별(가치/성장) 투자 등급 자동 평가 시스템")

st.sidebar.header("📥 분석 대상")
method = st.sidebar.radio("입력 방식", ("텍스트", "구글 시트", "CSV 업로드"))
tickers = []

if method == "텍스트":
    raw = st.sidebar.text_area("티커 입력 (한 줄에 하나씩)")
    if raw: tickers = [t.strip().upper() for t in raw.split('\n') if t.strip()]
elif method == "구글 시트":
    try:
        sid, sname = st.secrets["GOOGLE_SHEET_ID"], st.secrets["GOOGLE_SHEET_NAME"]
        url = f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={quote(sname)}"
        gs_df = pd.read_csv(url); t_col = st.sidebar.selectbox("티커 컬럼", gs_df.columns)
        tickers = gs_df[t_col].dropna().astype(str).tolist()
    except Exception as e: st.sidebar.error("시트 연결 확인 필요")
elif method == "CSV 업로드":
    up = st.sidebar.file_uploader("파일 선택", type=["csv"])
    if up:
        df = pd.read_csv(up); t_col = st.sidebar.selectbox("티커 컬럼", df.columns)
        tickers = df[t_col].dropna().astype(str).tolist()

if tickers:
    if st.button("🚀 분석 실행 및 등급 평가"):
        prog = st.progress(0); status = st.empty(); results = []
        
        # 헤더 정의 (REV 추가)
        base_cols = [
            'ticker', 'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Runway(Y)', 
            'TotalCash(M$)', 'FCF(M$)', 'FCF_Stability(%)', 'OCF(M$)', 
            'PBR', 'BPS', 'PER', 'EPS', 'Updated'
        ]
        metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF", "REV"]
        history_cols = [f"{m}_{y}" for m in metrics for y in ["Y4", "Y3", "Y2", "Y1", "TTM"]]
        final_cols = base_cols + history_cols

        for idx, symbol in enumerate(tickers):
            status.info(f"분석 중: {symbol} ({idx+1}/{len(tickers)})")
            data = get_extended_financials(symbol)
            row = [symbol] + data[:13] + [datetime.now().strftime('%H:%M:%S')] + data[13:]
            results.append(row)
            prog.progress((idx+1)/len(tickers))
            time.sleep(0.3)

        res_df = pd.DataFrame(results, columns=final_cols)

        # 투자 등급 평가 적용
        eval_data = []
        for _, row in res_df.iterrows():
            stype, grade, reason = evaluate_investment(row)
            eval_data.append({"투자 유형": stype, "최종 등급": grade, "핵심 평가": reason})
        
        eval_df = pd.DataFrame(eval_data)
        
        final_display_df = pd.concat([
            res_df[['ticker']], 
            eval_df, 
            res_df.drop(columns=['ticker'])
        ], axis=1).fillna("-")

        status.success("✅ 전수 분석 및 유형별 등급 평가 완료!")
        st.subheader("🎯 종목별 종합 투자 평가 결과")
        st.dataframe(final_display_df, use_container_width=True)
        st.download_button("📥 결과 CSV 다운로드", final_display_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'), "stock_grading_report.csv")
