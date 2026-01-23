import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from urllib.parse import quote
import google.generativeai as genai

# --- [함수] Gemini AI 투자 분석 생성 (무료) ---
def generate_ai_analysis(ticker, data_summary):
    """
    Google Gemini 1.5 Flash API를 사용하여 무료로 재무 분석 리포트를 생성합니다.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        if not api_key:
            return "API 키가 설정되지 않았습니다."
            
        genai.configure(api_key=api_key)
        
        # 모델 설정 (가장 안정적인 최신 플래시 모델명 사용)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        당신은 노련한 주식 투자 분석가입니다. 아래 제공된 기업({ticker})의 재무 데이터를 정밀 분석하여 투자 의견을 한국어로 작성하세요.
        
        [핵심 재무 데이터]
        - 부채비율(DTE): {data_summary.get('DTE')}%
        - ROE: {data_summary.get('ROE')}% / OPM: {data_summary.get('OPM')}%
        - FCF 안정성(5년간 플러스 횟수): {data_summary.get('Stability')}%
        - Cash Flow Quality(이익의 질): {data_summary.get('CFQ')}
        - Runway(현금 여력): {data_summary.get('Runway')}년
        - 밸류에이션: PBR {data_summary.get('PBR')} / PER {data_summary.get('PER')}
        
        [작성 가이드라인]
        1. 첫 줄에 투자 등급 명시 (💎강력매수 / ✅매수 / 🟡보유 / 🚨주의)
        2. 재무 건전성과 현금흐름의 지속 가능성을 날카롭게 비평하세요 (2문장).
        3. 수치상 드러나지 않는 잠재적 기회나 리스크를 짚어주세요 (1문장).
        4. 어조는 전문적이고 단호한 평어체를 사용하세요.
        """

        # generate_content 호출 시 모델 경로 문제가 생기지 않도록 처리
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            return "AI가 응답을 생성했으나 내용이 비어있습니다."
            
    except Exception as e:
        # 에러 메시지를 더 구체적으로 파악하기 위해 출력
        return f"AI 분석 중 오류 발생: {str(e)}"

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
        
        if total_cash and ttm_fcf:
            runway = round(total_cash / abs(ttm_fcf), 2) if ttm_fcf < 0 else "Infinite"
        else:
            runway = None

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
        stability = (sum(1 for v in fcf_series if v is not None and v > 0) / 5) * 100 if any(v is not None for v in fcf_series) else None
        ttm_cfq = round(ttm_ocf/ttm_net_inc, 2) if ttm_ocf and ttm_net_inc and ttm_net_inc != 0 else None

        # 🤖 AI 분석 실행 (Gemini)
        ai_data_summary = {
            'DTE': ttm_dte, 'ROE': ttm_roe, 'OPM': ttm_opm, 
            'Stability': stability, 'CFQ': ttm_cfq, 'Runway': runway,
            'PBR': info.get("priceToBook"), 'PER': info.get("trailingPE")
        }
        ai_opinion = generate_ai_analysis(symbol, ai_data_summary)

        # 3. 데이터 패킹
        base_results = [
            round(ttm_dte, 2) if ttm_dte is not None else None,
            round(ttm_cr, 2) if ttm_cr is not None else None,
            round(ttm_opm, 2) if ttm_opm is not None else None,
            round(ttm_roe, 2) if ttm_roe is not None else None,
            runway, round(total_cash/1_000_000, 2) if total_cash else None,
            ttm_fcf_m, stability, round(ttm_ocf / 1_000_000, 2) if ttm_ocf else None,
            round(info.get("priceToBook"), 2) if info.get("priceToBook") else None,
            round(info.get("bookValue"), 2) if info.get("bookValue") else None,
            round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
            round(info.get("trailingEps"), 2) if info.get("trailingEps") else None,
            ai_opinion # AI 분석 결과 칼럼
        ]

        # 4. 시계열 데이터 결합
        ttm_vals_map = {
            "DTE": base_results[0], "CR": base_results[1], "OPM": base_results[2], 
            "ROE": base_results[3], "OCF": base_results[8], "EPS": base_results[12],
            "CFQ": ttm_cfq, "FCF": ttm_fcf_m
        }
        flattened_history = []
        for key in metrics_order:
            flattened_history.extend(history[key] + [ttm_vals_map.get(key)])

        return base_results + flattened_history
    except Exception as e:
        return [None] * 54

# --- [UI] Streamlit 설정 ---
st.set_page_config(page_title="AI Financial Intelligence", layout="wide")
st.title("🚀 Gemini AI 기반 주식 재무 전수 분석")

# --- [사이드바] ---
st.sidebar.header("📥 분석 대상 설정")
method = st.sidebar.radio("방식", ("텍스트 붙여넣기", "구글 스프레드시트", "CSV 파일 업로드"))
tickers = []
if method == "텍스트 붙여넣기":
    raw = st.sidebar.text_area("티커 입력 (한 줄에 하나)")
    if raw: tickers = [t.strip().upper() for t in raw.split('\n') if t.strip()]
# (구글 시트 및 CSV 로직은 이전과 동일하므로 유지)

# --- [메인] 실행 ---
if tickers:
    if st.button("🔍 전수 분석 및 AI 의견 생성 시작"):
        prog = st.progress(0); status = st.empty(); results = []
        
        base_cols = [
            'ticker', 'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Runway(Y)', 
            'TotalCash(M$)', 'FCF(M$)', 'FCF_Stability(%)', 'OCF(M$)', 
            'PBR', 'BPS', 'PER', 'EPS', 'AI_Opinion', 'Updated'
        ]
        metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history_cols = [f"{m}_{y}" for m in metrics for y in ["Y4", "Y3", "Y2", "Y1", "TTM"]]
        final_cols = base_cols + history_cols

        for idx, symbol in enumerate(tickers):
            status.markdown(f"### ⏳ **{symbol}** 분석 및 AI 리포트 작성 중... ({idx+1}/{len(tickers)})")
            data = get_extended_financials(symbol)
            row = [symbol] + data[:14] + [datetime.now().strftime('%H:%M:%S')] + data[14:]
            results.append(row)
            prog.progress((idx+1)/len(tickers))
            time.sleep(2) # 무료 티어 Rate Limit(분당 15건) 고려

        status.success("✅ 분석 완료!")
        res_df = pd.DataFrame(results, columns=final_cols).fillna("-")
        st.dataframe(res_df, use_container_width=True)
        st.download_button("📥 결과 다운로드", res_df.to_csv(index=False).encode('utf-8'), "ai_stock_analysis.csv")
