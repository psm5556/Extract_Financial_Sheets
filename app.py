import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from urllib.parse import quote

# --- [함수] 주식 유형 분류 (가치주 vs 성장주) ---
def classify_stock_type(row):
    """
    PER, PBR, EPS 성장률을 기준으로 가치주/성장주/혼합형 분류
    """
    per = row.get('PER')
    pbr = row.get('PBR')
    eps_y3 = row.get('EPS_Y3')
    eps_ttm = row.get('EPS_TTM')
    
    # EPS 성장률 계산
    eps_growth = None
    if isinstance(eps_y3, (int, float)) and isinstance(eps_ttm, (int, float)) and eps_y3 != 0:
        eps_growth = ((eps_ttm - eps_y3) / abs(eps_y3)) * 100
    
    # 분류 기준
    is_low_valuation = False
    is_high_growth = False
    
    # 저평가 기준 (PER < 15, PBR < 1.5)
    if isinstance(per, (int, float)) and isinstance(pbr, (int, float)):
        if per > 0 and per < 15 and pbr < 1.5:
            is_low_valuation = True
    
    # 고성장 기준 (EPS 성장률 > 15%)
    if eps_growth and eps_growth > 15:
        is_high_growth = True
    
    # 최종 분류
    if is_high_growth and not is_low_valuation:
        return "성장주", eps_growth
    elif is_low_valuation and not is_high_growth:
        return "가치주", eps_growth
    elif is_high_growth and is_low_valuation:
        return "혼합형", eps_growth
    else:
        return "중립", eps_growth

# --- [함수] 가치주 평가 로직 ---
def evaluate_value_stock(row):
    """
    가치주 평가 기준:
    1. 저평가 지표 (PER, PBR)
    2. 재무 건전성 (DTE, CR)
    3. 현금흐름 안정성 (FCF_Stability)
    4. 수익성 (ROE)
    """
    score = 0
    reasons = []
    
    try:
        # 1. 저평가 지표 (40점)
        per = row.get('PER')
        pbr = row.get('PBR')
        
        if isinstance(per, (int, float)) and per > 0:
            if per < 10:
                score += 25
                reasons.append("✅ 매우 낮은 PER (10 미만)")
            elif per < 15:
                score += 15
                reasons.append("✅ 적정 PER (15 미만)")
        
        if isinstance(pbr, (int, float)):
            if pbr < 1.0:
                score += 15
                reasons.append("✅ 장부가치 이하 거래 (PBR < 1)")
            elif pbr < 1.5:
                score += 10
                reasons.append("✅ 적정 PBR (1.5 미만)")
        
        # 2. 재무 건전성 (30점)
        dte = row.get('DTE(%)')
        cr = row.get('CR(%)')
        
        if isinstance(dte, (int, float)):
            if dte <= 50:
                score += 20
                reasons.append("✅ 초우량 부채비율 (50% 이하)")
            elif dte <= 100:
                score += 15
                reasons.append("✅ 안정적 부채비율 (100% 이하)")
            elif dte > 200:
                score -= 15
                reasons.append("🚨 고부채 리스크")
        
        if isinstance(cr, (int, float)) and cr >= 150:
            score += 10
            reasons.append("✅ 우수한 유동성")
        
        # 3. 현금흐름 안정성 (20점)
        fcf_stability = row.get('FCF_Stability(%)')
        fcf = row.get('FCF(M$)')
        
        if isinstance(fcf_stability, (int, float)) and fcf_stability >= 80:
            score += 15
            reasons.append("✅ 안정적 현금 창출 (5년간)")
        
        if isinstance(fcf, (int, float)) and fcf > 0:
            score += 5
            reasons.append("✅ 양(+)의 잉여현금흐름")
        
        # 4. 수익성 (10점)
        roe = row.get('ROE(%)')
        if isinstance(roe, (int, float)):
            if roe >= 10:
                score += 10
                reasons.append("✅ 안정적 자본수익률")
            elif roe < 0:
                score -= 10
                reasons.append("⚠️ 자본 잠식")
        
    except Exception:
        pass
    
    # 등급 결정 (가치주)
    if score >= 85: grade = "S+ (최고 가치주)"
    elif score >= 70: grade = "A (우량 가치주)"
    elif score >= 50: grade = "B (양호 가치주)"
    elif score >= 30: grade = "C (보통 수준)"
    else: grade = "D (투자 부적합)"
    
    return grade, score, ", ".join(reasons) if reasons else "데이터 부족"

# --- [함수] 성장주 평가 로직 ---
def evaluate_growth_stock(row):
    """
    성장주 평가 기준:
    1. EPS 성장성 (3년 추세)
    2. ROE 성장 추세
    3. 현금흐름 질 (CFQ)
    4. 영업이익률 (OPM) 개선
    """
    score = 0
    reasons = []
    
    try:
        # 1. EPS 성장성 (40점)
        eps_y3 = row.get('EPS_Y3')
        eps_y2 = row.get('EPS_Y2')
        eps_ttm = row.get('EPS_TTM')
        
        eps_growth_3y = None
        if isinstance(eps_y3, (int, float)) and isinstance(eps_ttm, (int, float)) and eps_y3 != 0:
            eps_growth_3y = ((eps_ttm - eps_y3) / abs(eps_y3)) * 100
            
            if eps_growth_3y > 50:
                score += 40
                reasons.append(f"✅ 초고속 성장 (3년 EPS {eps_growth_3y:.1f}% 증가)")
            elif eps_growth_3y > 25:
                score += 30
                reasons.append(f"✅ 고속 성장 (3년 EPS {eps_growth_3y:.1f}% 증가)")
            elif eps_growth_3y > 15:
                score += 20
                reasons.append(f"✅ 성장 중 (3년 EPS {eps_growth_3y:.1f}% 증가)")
            elif eps_growth_3y < -10:
                score -= 20
                reasons.append("🚨 실적 역성장")
        
        # 2. ROE 성장 추세 (25점)
        roe_y3 = row.get('ROE_Y3')
        roe_ttm = row.get('ROE(%)')
        
        if isinstance(roe_y3, (int, float)) and isinstance(roe_ttm, (int, float)):
            if roe_ttm > roe_y3 and roe_ttm >= 15:
                score += 25
                reasons.append("✅ ROE 상승 + 고수익성")
            elif roe_ttm > roe_y3:
                score += 15
                reasons.append("✅ 자본효율 개선 중")
        
        # 3. 현금흐름 질 (20점)
        cfq_ttm = row.get('CFQ_TTM')
        if isinstance(cfq_ttm, (int, float)):
            if cfq_ttm >= 1.2:
                score += 20
                reasons.append("✅ 우수한 현금 전환율 (CFQ 120%↑)")
            elif cfq_ttm >= 0.8:
                score += 10
                reasons.append("✅ 적정 현금흐름")
            elif cfq_ttm < 0.5:
                score -= 10
                reasons.append("⚠️ 현금흐름 부족")
        
        # 4. 영업이익률 개선 (15점)
        opm_y3 = row.get('OPM_Y3')
        opm_ttm = row.get('OPM(%)')
        
        if isinstance(opm_y3, (int, float)) and isinstance(opm_ttm, (int, float)):
            if opm_ttm > opm_y3 and opm_ttm >= 15:
                score += 15
                reasons.append("✅ 마진 개선 + 고수익")
            elif opm_ttm > opm_y3:
                score += 10
                reasons.append("✅ 수익성 개선 중")
        
    except Exception:
        pass
    
    # 등급 결정 (성장주)
    if score >= 85: grade = "S+ (최고 성장주)"
    elif score >= 70: grade = "A (우량 성장주)"
    elif score >= 50: grade = "B (양호 성장주)"
    elif score >= 30: grade = "C (성장 둔화)"
    else: grade = "D (투자 부적합)"
    
    return grade, score, ", ".join(reasons) if reasons else "데이터 부족"

# --- [함수] 혼합형/중립 평가 ---
def evaluate_hybrid_stock(row):
    """
    혼합형(저평가+고성장) 또는 중립 종목 평가
    """
    score = 0
    reasons = []
    
    try:
        # 균형잡힌 평가 (가치+성장 요소 통합)
        
        # 1. 성장성 (30점)
        eps_y3 = row.get('EPS_Y3')
        eps_ttm = row.get('EPS_TTM')
        if isinstance(eps_y3, (int, float)) and isinstance(eps_ttm, (int, float)) and eps_y3 != 0:
            eps_growth = ((eps_ttm - eps_y3) / abs(eps_y3)) * 100
            if eps_growth > 20:
                score += 30
                reasons.append(f"✅ 성장성 우수 ({eps_growth:.1f}%)")
            elif eps_growth > 10:
                score += 20
                reasons.append("✅ 적정 성장세")
        
        # 2. 가치 평가 (30점)
        per = row.get('PER')
        pbr = row.get('PBR')
        if isinstance(per, (int, float)) and per > 0 and per < 20:
            score += 15
            reasons.append("✅ 적정 밸류에이션")
        if isinstance(pbr, (int, float)) and pbr < 2.0:
            score += 15
            reasons.append("✅ 합리적 PBR")
        
        # 3. 재무 건전성 (20점)
        dte = row.get('DTE(%)')
        if isinstance(dte, (int, float)) and dte <= 100:
            score += 20
            reasons.append("✅ 안정적 재무구조")
        
        # 4. 수익성 (20점)
        roe = row.get('ROE(%)')
        if isinstance(roe, (int, float)) and roe >= 12:
            score += 20
            reasons.append("✅ 우수한 ROE")
    
    except Exception:
        pass
    
    if score >= 80: grade = "S (균형 우량주)"
    elif score >= 60: grade = "A (안정 투자)"
    elif score >= 40: grade = "B (보통)"
    else: grade = "C (투자 유의)"
    
    return grade, score, ", ".join(reasons) if reasons else "데이터 부족"

# --- [함수] 통합 평가 (유형별 분기) ---
def evaluate_investment_by_type(row):
    """
    주식 유형을 먼저 분류한 후, 해당 유형에 맞는 평가 로직 적용
    """
    stock_type, eps_growth = classify_stock_type(row)
    
    if stock_type == "가치주":
        grade, score, reasons = evaluate_value_stock(row)
    elif stock_type == "성장주":
        grade, score, reasons = evaluate_growth_stock(row)
    else:  # 혼합형 또는 중립
        grade, score, reasons = evaluate_hybrid_stock(row)
    
    eps_growth_text = f"{eps_growth:.1f}%" if eps_growth else "N/A"
    
    return stock_type, grade, score, eps_growth_text, reasons

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

        # 1. TTM 기본 데이터 추출
        ttm_dte = info.get("debtToEquity")
        ttm_cr = (info.get("currentRatio") * 100) if info.get("currentRatio") else None
        ttm_opm = (info.get("operatingMargins") * 100) if info.get("operatingMargins") else None
        ttm_roe = (info.get("returnOnEquity") * 100) if info.get("returnOnEquity") else None
        ttm_ocf = info.get("operatingCashflow")
        ttm_fcf = info.get("freeCashflow")
        ttm_net_inc = info.get("netIncomeToCommon")
        total_cash = info.get("totalCash")
        
        runway = round(total_cash / abs(ttm_fcf), 2) if total_cash and ttm_fcf and ttm_fcf < 0 else "Infinite"

        # 2. 5개년 추이 수집 (Y4 -> TTM)
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
        stability = (sum(1 for v in fcf_series if v is not None and v > 0) / 5) * 100 if any(v is not None for v in fcf_series) else 0

        # 요약 결과 (13개 기본 지표)
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

        # TTM 맵 생성 (추이 데이터용)
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
    except Exception:
        return [None] * (13 + 40)

# --- [UI] Streamlit 설정 ---
st.set_page_config(page_title="Stock Grading System V2", layout="wide")
st.title("📊 가치주/성장주 구분 평가 시스템")
st.markdown("*종목 특성별 맞춤형 투자 등급 평가*")

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
        
        # 헤더 정의
        base_cols = [
            'ticker', 'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Runway(Y)', 
            'TotalCash(M$)', 'FCF(M$)', 'FCF_Stability(%)', 'OCF(M$)', 
            'PBR', 'BPS', 'PER', 'EPS', 'Updated'
        ]
        metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
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

        # 투자 등급 평가 적용 (유형별 분기)
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
        
        # 티커 옆에 평가 결과 배치
        final_display_df = pd.concat([
            res_df[['ticker']], 
            eval_df, 
            res_df.drop(columns=['ticker'])
        ], axis=1).fillna("-")

        status.success("✅ 전수 분석 및 유형별 평가 완료!")
        
        # 유형별 통계
        type_counts = eval_df['종목 유형'].value_counts()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("가치주", type_counts.get('가치주', 0))
        col2.metric("성장주", type_counts.get('성장주', 0))
        col3.metric("혼합형", type_counts.get('혼합형', 0))
        col4.metric("중립", type_counts.get('중립', 0))
        
        st.subheader("🎯 종목별 종합 투자 평가")
        st.dataframe(final_display_df, use_container_width=True)
        st.download_button(
            "📥 결과 CSV 다운로드", 
            final_display_df.to_csv(index=False).encode('utf-8'), 
            "stock_grading_v2_report.csv"
        )
