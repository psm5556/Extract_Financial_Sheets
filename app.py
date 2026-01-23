import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from urllib.parse import quote

# --- [함수] EPS 성장률 및 상태 계산 ---
def calculate_eps_growth(current, past):
    """
    턴어라운드(적자->흑자) 및 역성장을 구분하여 성장률 계산
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
    
    # 가치주 기준 (보수적 접근)
    if isinstance(per, (int, float)) and isinstance(pbr, (int, float)):
        if 0 < per < 15 and pbr < 1.5:
            is_low_valuation = True
    
    # 성장주 기준
    if eps_growth and (eps_growth > 15 or status == "턴어라운드(흑자전환)"):
        is_high_growth = True
    
    if is_high_growth and not is_low_valuation: return "성장주", eps_growth
    elif is_low_valuation and not is_high_growth: return "가치주", eps_growth
    elif is_high_growth and is_low_valuation: return "혼합형", eps_growth
    else: return "중립", eps_growth

# --- [함수] 가치주 평가 로직 (배당 추가) ---
def evaluate_value_stock(row):
    score = 0
    reasons = []
    
    try:
        # 1. 저평가 지표 (35점)
        per, pbr = row.get('PER'), row.get('PBR')
        if isinstance(per, (int, float)) and 0 < per < 12:
            score += 20; reasons.append("✅ 저PER (12미만)")
        if isinstance(pbr, (int, float)) and pbr < 1.0:
            score += 15; reasons.append("✅ PBR 1배 미만")

        # 2. 배당 수익률 (15점) - 신규 추가
        div_yield = row.get('Div_Yield(%)')
        if isinstance(div_yield, (int, float)):
            if div_yield >= 4: score += 15; reasons.append(f"💰 고배당 ({div_yield}%)")
            elif div_yield >= 2: score += 10; reasons.append(f"💰 보통배당 ({div_yield}%)")

        # 3. 재무 건전성 (30점)
        dte, cr = row.get('DTE(%)'), row.get('CR(%)')
        if isinstance(dte, (int, float)) and dte <= 70:
            score += 20; reasons.append("✅ 낮은 부채비율")
        if isinstance(cr, (int, float)) and cr >= 150:
            score += 10; reasons.append("✅ 유동성 확보")

        # 4. 수익성 및 현금흐름 (20점)
        fcf_stab, roe = row.get('FCF_Stability(%)'), row.get('ROE(%)')
        if isinstance(fcf_stab, (int, float)) and fcf_stab >= 80:
            score += 10; reasons.append("✅ 현금흐름 안정성")
        if isinstance(roe, (int, float)) and roe >= 8:
            score += 10; reasons.append("✅ 최소 수익성(ROE) 충족")

    except Exception: pass
    
    if score >= 80: grade = "S (초우량 가치주)"
    elif score >= 60: grade = "A (우량 가치주)"
    elif score >= 40: grade = "B (보통 가치주)"
    else: grade = "C (관망 필요)"
    
    return grade, score, ", ".join(reasons)

# --- [함수] 성장주 평가 로직 (PEG 추가) ---
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
            score += 35; reasons.append(f"✅ 고속성장 ({growth:.1f}%)")
        elif growth and growth > 15:
            score += 20; reasons.append(f"✅ 견조한 성장 ({growth:.1f}%)")

        # 2. PEG (성장 가성비) (20점) - 신규 추가
        per = row.get('PER')
        if isinstance(per, (int, float)) and growth and growth > 0:
            peg = per / growth
            if peg < 1.0: score += 20; reasons.append(f"💎 저평가 성장주 (PEG {peg:.2f})")
            elif peg < 1.5: score += 10; reasons.append(f"✅ 적정 성장가치 (PEG {peg:.2f})")
            elif peg > 2.5: score -= 10; reasons.append(f"⚠️ 성장에 비해 고평가 (PEG {peg:.2f})")

        # 3. 수익성 개선 추세 (20점)
        roe_ttm, roe_y3 = row.get('ROE(%)'), row.get('ROE_Y3')
        if isinstance(roe_ttm, (int, float)) and isinstance(roe_y3, (int, float)):
            if roe_ttm > roe_y3 and roe_ttm >= 15:
                score += 20; reasons.append("✅ 고수익성 유지 및 개선")

        # 4. 현금흐름의 질 (20점)
        cfq = row.get('CFQ_TTM')
        if isinstance(cfq, (int, float)) and cfq >= 1.0:
            score += 20; reasons.append("✅ 순이익 이상의 현금 창출")

    except Exception: pass
    
    if score >= 80: grade = "S (스타 종목)"
    elif score >= 60: grade = "A (우량 성장주)"
    elif score >= 40: grade = "B (성장 초기)"
    else: grade = "C (성장성 둔화)"
    
    return grade, score, ", ".join(reasons)

# --- [함수] 재무 데이터 추출 로직 ---
def get_extended_financials(ticker_symbol):
    try:
        symbol = ticker_symbol.upper().strip()
        ticker = yf.Ticker(symbol)
        info = ticker.info
        fin, bs, cf = ticker.financials, ticker.balance_sheet, ticker.cashflow

        def get_val(df, label, idx):
            try: return df.loc[label].iloc[idx]
            except: return None

        # 데이터 정규화: dte가 0.5인 경우 50%로 변환
        raw_dte = info.get("debtToEquity")
        ttm_dte = (raw_dte if raw_dte and raw_dte > 5 else raw_dte * 100) if raw_dte else None
        
        ttm_cr = (info.get("currentRatio") * 100) if info.get("currentRatio") else None
        ttm_opm = (info.get("operatingMargins") * 100) if info.get("operatingMargins") else None
        ttm_roe = (info.get("returnOnEquity") * 100) if info.get("returnOnEquity") else None
        div_yield = (info.get("dividendYield") * 100) if info.get("dividendYield") else 0.0
        
        ttm_fcf = info.get("freeCashflow")
        ttm_ocf = info.get("operatingCashflow")
        ttm_net_inc = info.get("netIncomeToCommon")

        # 히스토리 데이터
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
            history["CFQ"][idx] = round(ocf_v/ni, 2) if ocf_v and ni else None
            history["FCF"][idx] = round((ocf_v + ce)/1_000_000, 2) if ocf_v and ce else None

        base_results = [
            round(ttm_dte, 2) if ttm_dte else None, round(ttm_cr, 2) if ttm_cr else None,
            round(ttm_opm, 2) if ttm_opm else None, round(ttm_roe, 2) if ttm_roe else None,
            round(div_yield, 2), round(ttm_fcf/1_000_000, 2) if ttm_fcf else None,
            round(info.get("priceToBook"), 2) if info.get("priceToBook") else None,
            round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
            round(info.get("trailingEps"), 2) if info.get("trailingEps") else None
        ]
        
        # TTM 값 매핑 및 히스토리 결합
        ttm_vals = {"DTE": base_results[0], "CR": base_results[1], "OPM": base_results[2], "ROE": base_results[3], "EPS": base_results[8], "CFQ": round(ttm_ocf/ttm_net_inc, 2) if ttm_ocf and ttm_net_inc else None}
        flattened = []
        for key in metrics_order:
            flattened.extend(history[key] + [ttm_vals.get(key)])

        return base_results + flattened
    except: return [None] * (9 + 40)

# --- [UI] Streamlit App ---
st.set_page_config(page_title="Stock Grading System V3", layout="wide")
st.title("📊 고도화된 주식 가치/성장 평가 시스템")

# (사이드바 입력 로직은 동일하므로 생략하거나 기존 코드 유지)
# ... [티커 입력 및 분석 실행 버튼 로직] ...

if st.sidebar.button("🚀 분석 실행"):
    # (기존 Ticker 루프 및 데이터프레임 생성 로직)
    # columns 정의 시 'Div_Yield(%)' 포함하도록 수정 필요
    pass
