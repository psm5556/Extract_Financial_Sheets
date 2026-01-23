import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from urllib.parse import quote

# --- [함수] 가치주/성장주 구분 및 맞춤형 평가 로직 ---
def evaluate_by_style(row):
    try:
        # 데이터 정제 (문자열 '-' 등을 0으로 처리)
        def to_float(val, default=0):
            try: return float(val) if val not in [None, '-', 'Infinite'] else default
            except: return default

        per = to_float(row.get('PER'))
        pbr = to_float(row.get('PBR'))
        roe = to_float(row.get('ROE(%)'))
        dte = to_float(row.get('DTE(%)'), 1000) # 부채는 없을 시 높은 값으로 가정
        cfq = to_float(row.get('CFQ_TTM'))
        
        # EPS 성장률 계산 (Y3 대비 TTM)
        eps_y3 = to_float(row.get('EPS_Y3'))
        eps_ttm = to_float(row.get('EPS_TTM'))
        eps_g = ((eps_ttm - eps_y3) / abs(eps_y3) * 100) if eps_y3 != 0 else 0

        # --- [1단계] 투자 스타일 분류 로직 ---
        # PER 20 초과 혹은 EPS 성장률 15% 이상이면 성장주로 분류
        if per > 20 or eps_g > 15 or pbr > 3:
            style = "성장주(Growth)"
            is_growth = True
        else:
            style = "가치주(Value)"
            is_growth = False
        
        score = 0
        reasons = []

        # --- [2단계] 스타일별 가중치 평가 ---
        if is_growth:
            # 성장주 핵심: ROE, EPS성장률, 재무건전성
            if roe >= 20: score += 40; reasons.append("🚀 초고수익 ROE")
            elif roe >= 10: score += 20; reasons.append("✅ 준수한 수익성")
            
            if eps_g >= 20: score += 40; reasons.append("📈 이익성장 폭발")
            elif eps_g > 0: score += 20; reasons.append("✅ 이익성장세")
            
            if dte <= 100: score += 20; reasons.append("🛡️ 낮은 부채비율")
        else:
            # 가치주 핵심: 저PBR, 현금흐름(CFQ), 수익성유지
            if pbr <= 1.2: score += 40; reasons.append("💎 장부가치 대비 저평가")
            elif pbr <= 2.0: score += 20; reasons.append("✅ 합리적 가격")
            
            if cfq >= 1.2: score += 40; reasons.append("💰 강력한 현금창출력")
            elif cfq >= 0.8: score += 20; reasons.append("✅ 안정적 현금흐름")
            
            if roe >= 10: score += 20; reasons.append("✅ 이익 유지력")

        # --- [3단계] 공통 감점 리스크 ---
        if dte > 250: score -= 20; reasons.append("🚨 고부채 리스크")
        if roe < 0: score -= 30; reasons.append("📉 적자 기업")

        # 최종 등급 확정
        if score >= 80: grade = "S (주도주/명품가치)"
        elif score >= 60: grade = "A (우량 종목)"
        elif score >= 40: grade = "B (보유/관망)"
        else: grade = "C (투자유의)"

        return style, grade, ", ".join(reasons) if reasons else "평가 데이터 부족"
    except Exception as e:
        return "미분류", "등급외", f"오류: {str(e)}"

# --- [함수] 재무 데이터 추출 엔진 ---
def get_financial_data(ticker_symbol):
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

        # TTM 및 주요 지표
        ttm_dte = info.get("debtToEquity")
        ttm_opm = (info.get("operatingMargins") * 100) if info.get("operatingMargins") else None
        ttm_roe = (info.get("returnOnEquity") * 100) if info.get("returnOnEquity") else None
        ttm_ocf = info.get("operatingCashflow")
        ttm_fcf = info.get("freeCashflow")
        total_cash = info.get("totalCash")
        
        runway = round(total_cash / abs(ttm_fcf), 2) if total_cash and ttm_fcf and ttm_fcf < 0 else "Infinite"

        # 5개년 시계열 추이 (Y4 -> TTM)
        metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history = {m: [None]*4 for m in metrics}
        num_years = min(len(fin.columns), 4) if not fin.empty else 0

        for i in range(num_years):
            idx = 3 - i 
            net_inc = get_val(fin, 'Net Income', i)
            equity = get_val(bs, 'Total Equity Gross Minority Interest', i)
            ocf_v = get_val(cf, 'Operating Cash Flow', i)
            cap_ex = get_val(cf, 'Capital Expenditure', i)
            fcf_v = (ocf_v + cap_ex) if ocf_v is not None and cap_ex is not None else None
            
            history["DTE"][idx] = round((get_val(bs, 'Total Liabilities Net Minority Interest', i)/equity*100), 2) if equity else None
            history["ROE"][idx] = round((net_inc/equity*100), 2) if equity else None
            history["EPS"][idx] = round(get_val(fin, 'Basic EPS', i), 2)
            history["CFQ"][idx] = round(ocf_v/net_inc, 2) if net_inc and net_inc != 0 else None
            history["FCF"][idx] = round(fcf_v/1_000_000, 2) if fcf_v else None

        ttm_fcf_m = round(ttm_fcf/1_000_000, 2) if ttm_fcf else None
        fcf_series = history["FCF"] + [ttm_fcf_m]
        stability = (sum(1 for v in fcf_series if v and v > 0) / 5) * 100

        # 결과 패킹
        base = [
            round(ttm_dte, 2) if ttm_dte else None, None, 
            round(ttm_opm, 2) if ttm_opm else None, round(ttm_roe, 2) if ttm_roe else None,
            runway, round(total_cash/1_000_000, 2) if total_cash else None,
            ttm_fcf_m, stability, round(ttm_ocf/1_000_000, 2) if ttm_ocf else None,
            round(info.get("priceToBook"), 2), round(info.get("bookValue"), 2),
            round(info.get("trailingPE"), 2), round(info.get("trailingEps"), 2)
        ]
        
        ttm_cfq = round(ttm_ocf/info.get("netIncomeToCommon"), 2) if ttm_ocf and info.get("netIncomeToCommon") else None
        ttm_map = {"DTE": base[0], "CR": None, "OPM": base[2], "ROE": base[3], "OCF": base[8], "EPS": base[12], "CFQ": ttm_cfq, "FCF": base[6]}
        
        flattened = []
        for m in metrics:
            flattened.extend(history[m] + [ttm_map[m]])

        return base + flattened
    except:
        return [None] * 53

# --- [UI] Streamlit ---
st.set_page_config(page_title="Style-Based Stock Analyzer", layout="wide")
st.title("⚖️ 가치주 vs 성장주 스타일별 투자 평가")

st.sidebar.markdown("### 🔍 분석 설정")
raw = st.sidebar.text_area("티커 입력 (한 줄에 하나)")
tickers = [t.strip().upper() for t in raw.split('\n') if t.strip()]

if tickers and st.sidebar.button("분석 실행"):
    prog = st.progress(0); status = st.empty(); all_results = []
    
    # 칼럼 정의
    base_cols = ['ticker', 'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Runway(Y)', 'TotalCash(M$)', 'FCF(M$)', 'FCF_Stability(%)', 'OCF(M$)', 'PBR', 'BPS', 'PER', 'EPS', 'Updated']
    metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
    history_cols = [f"{m}_{y}" for m in metrics for y in ["Y4", "Y3", "Y2", "Y1", "TTM"]]
    
    for idx, t in enumerate(tickers):
        status.info(f"데이터 수집 중: {t}")
        data = get_financial_data(t)
        row = [t] + data[:13] + [datetime.now().strftime('%H:%M')] + data[13:]
        all_results.append(row)
        prog.progress((idx+1)/len(tickers))
        time.sleep(0.3)

    res_df = pd.DataFrame(all_results, columns=base_cols + history_cols)

    # 스타일 평가 적용
    eval_rows = []
    for _, r in res_df.iterrows():
        style, grade, points = evaluate_by_style(r)
        eval_rows.append({'투자 스타일': style, '최종 등급': grade, '핵심 투자 포인트': points})
    
    eval_df = pd.DataFrame(eval_rows)
    final_df = pd.concat([res_df[['ticker']], eval_df, res_df.drop(columns=['ticker'])], axis=1).fillna("-")

    status.success("✅ 분석 완료!")
    
    # 결과 출력
    st.subheader("🎯 종합 투자 리포트")
    st.dataframe(final_df, use_container_width=True)

    # 스타일별 분류 요약
    c1, c2 = st.columns(2)
    with c1:
        st.info("📈 성장주(Growth) 상위 종목")
        st.table(final_df[final_df['투자 스타일'].str.contains("성장")].sort_values('최종 등급')[['ticker', '최종 등급', 'ROE(%)']].head(5))
    with c2:
        st.success("💎 가치주(Value) 상위 종목")
        st.table(final_df[final_df['투자 스타일'].str.contains("가치")].sort_values('최종 등급')[['ticker', '최종 등급', 'PBR']].head(5))

    st.download_button("📥 전체 결과 다운로드(CSV)", final_df.to_csv(index=False).encode('utf-8'), "investment_report.csv")
