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

        # 2. 항목별 추이 데이터 수집 (Y4 -> TTM)
        metrics_order = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history = {m: [None]*4 for m in metrics_order}
        num_years = min(len(fin.columns), 4) if not fin.empty else 0

        for i in range(num_years):
            idx = 3 - i 
            net_inc = get_val(fin, 'Net Income', i)
            ocf_val = get_val(cf, 'Operating Cash Flow', i)
            cap_ex = get_val(cf, 'Capital Expenditure', i)
            fcf_val = (ocf_val + cap_ex) if ocf_val is not None and cap_ex is not None else None
            
            history["DTE"][idx] = round((get_val(bs, 'Total Liabilities Net Minority Interest', i)/get_val(bs, 'Total Equity Gross Minority Interest', i)*100), 2) if get_val(bs, 'Total Liabilities Net Minority Interest', i) and get_val(bs, 'Total Equity Gross Minority Interest', i) else None
            history["CR"][idx] = round((get_val(bs, 'Current Assets', i)/get_val(bs, 'Current Liabilities', i)*100), 2) if get_val(bs, 'Current Assets', i) and get_val(bs, 'Current Liabilities', i) else None
            history["OPM"][idx] = round((get_val(fin, 'Operating Income', i)/get_val(fin, 'Total Revenue', i)*100), 2) if get_val(fin, 'Operating Income', i) and get_val(fin, 'Total Revenue', i) else None
            history["ROE"][idx] = round((net_inc/get_val(bs, 'Total Equity Gross Minority Interest', i)*100), 2) if net_inc and get_val(bs, 'Total Equity Gross Minority Interest', i) else None
            history["OCF"][idx] = round(ocf_val/1_000_000, 2) if ocf_val else None
            history["EPS"][idx] = round(get_val(fin, 'Basic EPS', i), 2) if get_val(fin, 'Basic EPS', i) else None
            history["CFQ"][idx] = round(ocf_val/net_inc, 2) if ocf_val and net_inc and net_inc != 0 else None
            history["FCF"][idx] = round(fcf_val/1_000_000, 2) if fcf_val else None

        # TTM 값 확정 (Stability 계산을 위해 시계열 합침)
        ttm_fcf_m = round(ttm_fcf/1_000_000, 2) if ttm_fcf else None
        fcf_series = history["FCF"] + [ttm_fcf_m]
        plus_count = sum(1 for v in fcf_series if v is not None and v > 0)
        stability = (plus_count / 5) * 100 if any(v is not None for v in fcf_series) else None

        # [수정된 base_results] FCF와 Stability를 OCF 앞으로 배치
        base_results = [
            round(ttm_dte, 2) if ttm_dte is not None else None,
            round(ttm_cr, 2) if ttm_cr is not None else None,
            round(ttm_opm, 2) if ttm_opm is not None else None,
            round(ttm_roe, 2) if ttm_roe is not None else None,
            runway,
            ttm_fcf_m,   # FCF 추가 (OCF 전)
            stability,   # Stability 추가
            round(ttm_ocf / 1_000_000, 2) if ttm_ocf else None,
            round(info.get("priceToBook"), 2) if info.get("priceToBook") else None,
            round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
            round(info.get("trailingEps"), 2) if info.get("trailingEps") else None
        ]

        # 시계열 추이 데이터 평탄화
        flattened_history = []
        ttm_vals_map = {
            "DTE": base_results[0], "CR": base_results[1], "OPM": base_results[2], 
            "ROE": base_results[3], "OCF": base_results[7], "EPS": base_results[10],
            "CFQ": round(ttm_ocf/ttm_net_inc, 2) if ttm_ocf and ttm_net_inc and ttm_net_inc != 0 else None,
            "FCF": ttm_fcf_m
        }
        
        for key in metrics_order:
            combined = history[key] + [ttm_vals_map[key]]
            flattened_history.extend(combined)

        # 리턴: [기본11개] + [추이40개]
        return base_results + flattened_history
    except Exception:
        return [None] * (11 + 40)

# --- [UI] Streamlit 설정 ---
st.set_page_config(page_title="Stock Master Analyzer", layout="wide")
st.title("📊 재무 핵심 요약 및 5개년 추이 분석")

# (사이드바 입력 로직 생략 - 이전과 동일)
# ... [이전 답변의 사이드바 코드와 동일하게 유지] ...

# --- [메인] 분석 실행 ---
if tickers:
    total = len(tickers)
    if st.button("🚀 분석 시작"):
        prog = st.progress(0); status = st.empty(); results = []
        
        # 1. 헤더 구성
        # base_cols: OCF 앞에 FCF와 Stability 배치
        base_cols = [
            'ticker', 'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Runway(Y)', 
            'FCF(M$)', 'FCF_Stability(%)', 'OCF(M$)', 'PBR', 'PER', 'EPS', 'Updated'
        ]
        
        metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history_cols = [f"{m}_{y}" for m in metrics for y in ["Y4", "Y3", "Y2", "Y1", "TTM"]]
        final_cols = base_cols + history_cols

        for idx, symbol in enumerate(tickers):
            status.markdown(f"### ⏳ 분석 중: **{symbol}** ({idx+1}/{total})")
            data = get_extended_financials(symbol)
            
            # row: [ticker] + [기본11개] + [시간] + [추이40개]
            row = [symbol] + data[:11] + [datetime.now().strftime('%H:%M:%S')] + data[11:]
            results.append(row)
            prog.progress((idx+1)/total); time.sleep(0.5)

        status.success(f"✅ 분석 완료!")
        res_df = pd.DataFrame(results, columns=final_cols).fillna("-")
        st.dataframe(res_df, use_container_width=True)
        st.download_button("📥 결과 다운로드", res_df.to_csv(index=False).encode('utf-8'), "financial_full_report.csv", "text/csv")
