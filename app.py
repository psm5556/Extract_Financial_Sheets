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
    적자->흑자 전환(Turnaround) 및 역성장을 구분하여 성장률 계산
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
    
    is_low_valuation = False
    is_high_growth = False
    
    # 가치주 기준: PER 15 미만, PBR 1.5 미만
    if isinstance(per, (int, float)) and isinstance(pbr, (int, float)):
        if 0 < per < 15 and pbr < 1.5:
            is_low_valuation = True
    
    # 성장주 기준: EPS 성장률 15% 이상 또는 흑자전환
    if eps_growth and (eps_growth > 15 or status == "턴어라운드(흑자전환)"):
        is_high_growth = True
    
    if is_high_growth and not is_low_valuation: return "성장주", eps_growth
    elif is_low_valuation and not is_high_growth: return "가치주", eps_growth
    elif is_high_growth and is_low_valuation: return "혼합형", eps_growth
    else: return "중립", eps_growth

# --- [함수] 가치주 평가 로직 (배당 지표 추가) ---
def evaluate_value_stock(row):
    score = 0
    reasons = []
    
    try:
        # 1. 저평가 지표 (35점)
        per, pbr = row.get('PER'), row.get('PBR')
        if isinstance(per, (int, float)) and 0 < per < 12:
            score += 20; reasons.append("✅ 저PER (12미만)")
        elif isinstance(per, (int, float)) and 0 < per < 15:
            score += 10; reasons.append("✅ 적정PER (15미만)")
            
        if isinstance(pbr, (int, float)) and pbr < 1.0:
            score += 15; reasons.append("✅ PBR 1배 미만(청산가치 미만)")

        # 2. 배당 수익률 (15점) - 신규 반영
        div_yield = row.get('Div_Yield(%)')
        if isinstance(div_yield, (int, float)):
            if div_yield >= 4: score += 15; reasons.append(f"💰 고배당 ({div_yield}%)")
            elif div_yield >= 2: score += 10; reasons.append(f"💰 보통배당 ({div_yield}%)")

        # 3. 재무 건전성 (30점)
        dte, cr = row.get('DTE(%)'), row.get('CR(%)')
        if isinstance(dte, (int, float)) and dte <= 70:
            score += 20; reasons.append("✅ 우량 부채비율 (70% 이하)")
        if isinstance(cr, (int, float)) and cr >= 150:
            score += 10; reasons.append("✅ 우수한 유동성")

        # 4. 수익성 및 현금흐름 (20점)
        fcf_stab, roe = row.get('FCF_Stability(%)'), row.get('ROE(%)')
        if isinstance(fcf_stab, (int, float)) and fcf_stab >= 80:
            score += 10; reasons.append("✅ 현금흐름 안정성")
        if isinstance(roe, (int, float)) and roe >= 8:
            score += 10; reasons.append("✅ 자본효율성(ROE) 양호")

    except Exception: pass
    
    if score >= 80: grade = "S (초우량 가치주)"
    elif score >= 65: grade = "A (우량 가치주)"
    elif score >= 45: grade = "B (보통 수준)"
    else: grade = "C (투자 유의)"
    
    return grade, score, ", ".join(reasons) if reasons else "데이터 부족"

# --- [함수] 성장주 평가 로직 (PEG 지표 추가) ---
def evaluate_growth_stock(row):
    score = 0
    reasons = []
    
    try:
        # 1. EPS 성장성 (40점)
        eps_y3, eps_ttm = row.get('EPS_Y3'), row.get('EPS_TTM')
        growth, status = calculate_eps_growth(eps_ttm, eps_y3)
        
        if status == "턴어라운드(흑자전환)":
            score += 40; reasons.append("🚀 흑자전환(Turnaround) 성공")
        elif growth and growth > 30:
            score += 35; reasons.append(f"✅ 고속성장 (3년 EPS {growth:.1f}% 증가)")
        elif growth and growth > 15:
            score += 20; reasons.append(f"✅ 견조한 성장 ({growth:.1f}%)")

        # 2. PEG (Price/Earnings to Growth) (20점) - 신규 반영
        per = row.get('PER')
        if isinstance(per, (int, float)) and growth and growth > 0:
            peg = per / growth
            if peg < 1.0: score += 20; reasons.append(f"💎 저평가 성장주 (PEG {peg:.2f})")
            elif peg < 1.5: score += 10; reasons.append(f"✅ 적정 성장가치 (PEG {peg:.2f})")
            elif peg > 2.5: score -= 10; reasons.append(f"⚠️ 고평가 성장주 (PEG {peg:.2f})")

        # 3. 수익성 개선 추세 (20점)
        roe_ttm, roe_y3 = row.get('ROE(%)'), row.get('ROE_Y3')
        if isinstance(roe_ttm, (int, float)) and isinstance(roe_y3, (int, float)):
            if roe_ttm > roe_y3 and roe_ttm >= 15:
                score += 20; reasons.append("✅ ROE 상승 + 고수익성")

        # 4. 현금흐름 질 (20점)
        cfq = row.get('CFQ_TTM')
        if isinstance(cfq, (int, float)) and cfq >= 1.0:
            score += 20; reasons.append("✅ 순이익 이상의 현금 창출")

    except Exception: pass
    
    if score >= 80: grade = "S (스타 종목)"
    elif score >= 65: grade = "A (우량 성장주)"
    elif score >= 45: grade = "B (성장 초기)"
    else: grade = "C (성장 둔화)"
    
    return grade, score, ", ".join(reasons) if reasons else "데이터 부족"

# --- [함수] 혼합형/중립 평가 ---
def evaluate_hybrid_stock(row):
    grade, score, reasons = evaluate_value_stock(row) # 기본적으로 가치 잣대 사용
    return "Hybrid-" + grade, score, reasons

# --- [함수] 통합 평가 분기 ---
def evaluate_investment_by_type(row):
    stock_type, eps_growth = classify_stock_type(row)
    
    if stock_type == "가치주":
        grade, score, reasons = evaluate_value_stock(row)
    elif stock_type == "성장주":
        grade, score, reasons = evaluate_growth_stock(row)
    else:
        grade, score, reasons = evaluate_hybrid_stock(row)
    
    eps_growth_text = f"{eps_growth:.1f}%" if eps_growth else "N/A"
    return stock_type, grade, score, eps_growth_text, reasons

# --- [함수] 재무 데이터 추출 (yfinance) ---
def get_extended_financials(ticker_symbol):
    try:
        symbol = ticker_symbol.upper().strip()
        ticker = yf.Ticker(symbol)
        info = ticker.info
        fin, bs, cf = ticker.financials, ticker.balance_sheet, ticker.cashflow

        def get_val(df, label, idx):
            try: return df.loc[label].iloc[idx]
            except: return None

        # 데이터 정규화: dte가 소수점(0.5)인 경우 50%로 변환
        raw_dte = info.get("debtToEquity")
        ttm_dte = (raw_dte if raw_dte and raw_dte > 5 else raw_dte * 100) if raw_dte else None
        
        ttm_cr = (info.get("currentRatio") * 100) if info.get("currentRatio") else None
        ttm_opm = (info.get("operatingMargins") * 100) if info.get("operatingMargins") else None
        ttm_roe = (info.get("returnOnEquity") * 100) if info.get("returnOnEquity") else None
        div_yield = (info.get("dividendYield") * 100) if info.get("dividendYield") else 0.0
        
        ttm_fcf = info.get("freeCashflow")
        ttm_ocf = info.get("operatingCashflow")
        ttm_net_inc = info.get("netIncomeToCommon")

        # 히스토리 수집 (Y4 -> TTM)
        metrics_order = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history = {m: [None]*4 for m in metrics_order}
        num_years = min(len(fin.columns), 4) if not fin.empty else 0

        for i in range(num_years):
            idx = 3 - i
            ni = get_val(fin, 'Net Income', i)
            eq = get_val(bs, 'Total Equity Gross Minority Interest', i)
            ocf_v = get_val(cf, 'Operating Cash Flow', i)
            ce = get_val(cf, 'Capital Expenditure', i)
            
            history["DTE"][idx] = round((get_val(bs, 'Total Liabilities Net Minority Interest', i)/eq*100), 2) if eq else None
            history["CR"][idx] = round((get_val(bs, 'Current Assets', i)/get_val(bs, 'Current Liabilities', i)*100), 2) if get_val(bs, 'Current Liabilities', i) else None
            history["OPM"][idx] = round((get_val(fin, 'Operating Income', i)/get_val(fin, 'Total Revenue', i)*100), 2) if get_val(fin, 'Total Revenue', i) else None
            history["ROE"][idx] = round((ni/eq*100), 2) if ni and eq else None
            history["EPS"][idx] = round(get_val(fin, 'Basic EPS', i), 2) if get_val(fin, 'Basic EPS', i) else None
            history["CFQ"][idx] = round(ocf_v/ni, 2) if ocf_v and ni and ni != 0 else None
            history["FCF"][idx] = round((ocf_v + ce)/1_000_000, 2) if ocf_v and ce else None

        ttm_fcf_m = round(ttm_fcf/1_000_000, 2) if ttm_fcf else None
        fcf_series = history["FCF"] + [ttm_fcf_m]
        stability = (sum(1 for v in fcf_series if v is not None and v > 0) / 5) * 100 if any(v is not None for v in fcf_series) else 0

        base_results = [
            round(ttm_dte, 2) if ttm_dte else None, round(ttm_cr, 2) if ttm_cr else None,
            round(ttm_opm, 2) if ttm_opm else None, round(ttm_roe, 2) if ttm_roe else None,
            round(div_yield, 2), ttm_fcf_m, stability,
            round(info.get("priceToBook"), 2) if info.get("priceToBook") else None,
            round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
            round(info.get("trailingEps"), 2) if info.get("trailingEps") else None
        ]
        
        ttm_vals = {
            "DTE": base_results[0], "CR": base_results[1], "OPM": base_results[2], 
            "ROE": base_results[3], "EPS": base_results[9], "FCF": ttm_fcf_m,
            "CFQ": round(ttm_ocf/ttm_net_inc, 2) if ttm_ocf and ttm_net_inc and ttm_net_inc != 0 else None,
            "OCF": round(ttm_ocf/1_000_000, 2) if ttm_ocf else None
        }
        
        flattened = []
        for key in metrics_order:
            flattened.extend(history[key] + [ttm_vals.get(key)])

        return base_results + flattened
    except Exception: return [None] * (10 + 40)

# --- [UI] Streamlit 설정 ---
st.set_page_config(page_title="Stock Grading System V3", layout="wide")
st.title("📊 가치주/성장주 맞춤형 평가 시스템 V3")
st.markdown("*PEG, 배당 수익률, EPS 성장성 정밀 분석 반영*")

st.sidebar.header("📥 분석 대상")
method = st.sidebar.radio("입력 방식", ("텍스트", "CSV 업로드"))
tickers = []

if method == "텍스트":
    raw = st.sidebar.text_area("티커 입력 (한 줄에 하나씩)")
    if raw: tickers = [t.strip().upper() for t in raw.split('\n') if t.strip()]
elif method == "CSV 업로드":
    up = st.sidebar.file_uploader("파일 선택", type=["csv"])
    if up:
        df = pd.read_csv(up); t_col = st.sidebar.selectbox("티커 컬럼", df.columns)
        tickers = df[t_col].dropna().astype(str).tolist()

if tickers:
    if st.button("🚀 분석 실행"):
        prog = st.progress(0); status = st.empty(); results = []
        
        base_cols = [
            'ticker', 'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Div_Yield(%)',
            'FCF(M$)', 'FCF_Stability(%)', 'PBR', 'PER', 'EPS', 'Updated'
        ]
        metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history_cols = [f"{m}_{y}" for m in metrics for y in ["Y4", "Y3", "Y2", "Y1", "TTM"]]
        final_cols = base_cols + history_cols

        for idx, symbol in enumerate(tickers):
            status.info(f"데이터 수집 중: {symbol} ({idx+1}/{len(tickers)})")
            data = get_extended_financials(symbol)
            row = [symbol] + data[:10] + [datetime.now().strftime('%H:%M:%S')] + data[10:]
            results.append(row)
            prog.progress((idx+1)/len(tickers))
            time.sleep(0.3)

        res_df = pd.DataFrame(results, columns=final_cols)

        # 평가 로직 적용
        eval_data = []
        for _, row in res_df.iterrows():
            stock_type, grade, score, eps_growth, reasons = evaluate_investment_by_type(row)
            eval_data.append({
                "종목 유형": stock_type,
                "최종 등급": grade,
                "점수": score,
                "EPS 성장률(3Y)": eps_growth,
                "핵심 평가": reasons
            })
        
        eval_df = pd.DataFrame(eval_data)
        final_display_df = pd.concat([res_df[['ticker']], eval_df, res_df.drop(columns=['ticker'])], axis=1).fillna("-")

        status.success("✅ 분석 완료!")
        
        col1, col2, col3, col4 = st.columns(4)
        type_counts = eval_df['종목 유형'].value_counts()
        col1.metric("가치주", type_counts.get('가치주', 0))
        col2.metric("성장주", type_counts.get('성장주', 0))
        col3.metric("혼합형", type_counts.get('혼합형', 0))
        col4.metric("중립", type_counts.get('중립', 0))
        
        st.subheader("🎯 종목별 투자 평가 리포트")
        st.dataframe(final_display_df, use_container_width=True)
        st.download_button("📥 CSV 결과 다운로드", final_display_df.to_csv(index=False).encode('utf-8'), "stock_report_v3.csv")
