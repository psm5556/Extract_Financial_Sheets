import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from urllib.parse import quote
import json

# --- [함수] 재무 데이터 추출 로직 (원본 그대로) ---
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

        # 1. TTM (최근 12개월) 기본 데이터 추출
        ttm_dte = info.get("debtToEquity")
        ttm_cr = (info.get("currentRatio") * 100) if info.get("currentRatio") else None
        ttm_opm = (info.get("operatingMargins") * 100) if info.get("operatingMargins") else None
        ttm_roe = (info.get("returnOnEquity") * 100) if info.get("returnOnEquity") else None
        ttm_ocf = info.get("operatingCashflow")
        ttm_fcf = info.get("freeCashflow")
        ttm_net_inc = info.get("netIncomeToCommon")
        total_cash = info.get("totalCash")
        
        # Runway 계산
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

        # Stability 계산
        ttm_fcf_m = round(ttm_fcf/1_000_000, 2) if ttm_fcf else None
        fcf_series = history["FCF"] + [ttm_fcf_m]
        plus_count = sum(1 for v in fcf_series if v is not None and v > 0)
        stability = (plus_count / 5) * 100 if any(v is not None for v in fcf_series) else None

        # 3. 요약 섹션(base_results) 데이터 구성 (BPS 복구)
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
            round(info.get("bookValue"), 2) if info.get("bookValue") else None, # BPS 복구
            round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
            round(info.get("trailingEps"), 2) if info.get("trailingEps") else None
        ]

        # 4. 시계열 추이 데이터 매핑 (인덱스: BPS 추가로 하나씩 더 밀림)
        ttm_vals_map = {
            "DTE": base_results[0], "CR": base_results[1], "OPM": base_results[2], 
            "ROE": base_results[3], "OCF": base_results[8], "EPS": base_results[12],
            "CFQ": round(ttm_ocf/ttm_net_inc, 2) if ttm_ocf and ttm_net_inc and ttm_net_inc != 0 else None,
            "FCF": ttm_fcf_m
        }
        
        flattened_history = []
        for key in metrics_order:
            combined = history[key] + [ttm_vals_map[key]]
            flattened_history.extend(combined)

        return base_results + flattened_history
    except Exception:
        return [None] * (13 + 40)

# --- [함수] AI 투자 등급 분석 (새로 추가) ---
def analyze_with_ai(ticker, financial_data, llm_provider):
    """AI를 사용한 투자 등급 분석"""
    try:
        # Streamlit Secrets에서 API 키 가져오기
        if llm_provider == "gemini":
            if "GEMINI_API_KEY" not in st.secrets:
                return "-", "API 키 미설정"
            
            import google.generativeai as genai
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            
            # 재무 데이터 구성
            metrics = {
                "Ticker": ticker,
                "DTE(%)": financial_data[0],
                "CR(%)": financial_data[1],
                "OPM(%)": financial_data[2],
                "ROE(%)": financial_data[3],
                "Runway": financial_data[4],
                "Cash(M$)": financial_data[5],
                "FCF(M$)": financial_data[6],
                "Stability(%)": financial_data[7],
                "PBR": financial_data[9],
                "PER": financial_data[11],
                "EPS": financial_data[12]
            }
            
            prompt = f"""You are a financial analyst. Analyze this stock and provide:
1. Grade: A/B/C/D/F
2. Brief reason in Korean (max 50 words)

Data: {json.dumps(metrics, indent=2)}

Criteria:
- A: Excellent (ROE>15%, PER<20, Stable FCF, Low debt)
- B: Good (Most metrics positive)
- C: Average (Mixed results)
- D: Below average (Multiple weaknesses)
- F: Poor (Critical issues)

Return JSON: {{"grade": "A/B/C/D/F", "reason": "Korean text"}}"""
            
            # 여러 모델 시도
            for model_name in ['gemini-1.5-flash-latest', 'gemini-1.5-flash', 'gemini-pro']:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    result = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
                    return result.get("grade", "-"), result.get("reason", "-")
                except:
                    continue
            
            return "-", "모든 모델 실패"
            
        elif llm_provider == "groq":
            if "GROQ_API_KEY" not in st.secrets:
                return "-", "API 키 미설정"
            
            from groq import Groq
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            # 나머지 Groq 코드...
            return "-", "Groq 구현 예정"
            
        else:
            return "-", "지원하지 않는 LLM"
            
    except Exception as e:
        return "-", f"오류: {str(e)[:50]}"

# --- [UI] Streamlit 설정 (원본 그대로) ---
st.set_page_config(page_title="Stock Master Analyzer", layout="wide")
st.title("📊 주식 재무 시계열 분석 마스터 (Y4 → TTM) + AI")

# --- [사이드바] (원본에 AI 옵션만 추가) ---
st.sidebar.header("📥 데이터 소스")
method = st.sidebar.radio("방식", ("텍스트 붙여넣기", "구글 스프레드시트", "CSV 파일 업로드"))

# AI 옵션 추가
st.sidebar.markdown("---")
st.sidebar.header("🤖 AI 분석 옵션")
enable_ai = st.sidebar.checkbox("AI 투자 등급 분석", value=False)
if enable_ai:
    llm_provider = st.sidebar.selectbox("LLM 선택", ["gemini", "groq"])
    st.sidebar.info("💡 Streamlit Secrets에 API 키 설정 필요")

tickers = []
if method == "텍스트 붙여넣기":
    raw = st.sidebar.text_area("티커 입력 (한 줄에 하나씩)")
    if raw: tickers = [t.strip().upper() for t in raw.split('\n') if t.strip()]
elif method == "구글 스프레드시트":
    try:
        sid, sname = st.secrets["GOOGLE_SHEET_ID"], st.secrets["GOOGLE_SHEET_NAME"]
        url = f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={quote(sname)}"
        gs_df = pd.read_csv(url); t_col = st.sidebar.selectbox("티커 컬럼", gs_df.columns)
        tickers = gs_df[t_col].dropna().astype(str).tolist()
    except Exception as e: st.sidebar.error(f"연결 실패: {e}")
elif method == "CSV 파일 업로드":
    up = st.sidebar.file_uploader("CSV", type=["csv"])
    if up:
        df = pd.read_csv(up); t_col = st.sidebar.selectbox("티커 컬럼", df.columns)
        tickers = df[t_col].dropna().astype(str).tolist()

# --- [메인] 분석 실행 (원본 기반, AI만 추가) ---
if tickers:
    total = len(tickers)
    if st.button("🚀 전수 분석 시작"):
        prog = st.progress(0); status = st.empty(); results = []
        
        # 헤더 정의 (AI 컬럼 추가)
        if enable_ai:
            base_cols = [
                'ticker', 'AI_Grade', 'AI_Reason',
                'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Runway(Y)', 
                'TotalCash(M$)', 'FCF(M$)', 'FCF_Stability(%)', 'OCF(M$)', 
                'PBR', 'BPS', 'PER', 'EPS', 'Updated'
            ]
        else:
            # 원본 헤더 (AI 없음)
            base_cols = [
                'ticker', 'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Runway(Y)', 
                'TotalCash(M$)', 'FCF(M$)', 'FCF_Stability(%)', 'OCF(M$)', 
                'PBR', 'BPS', 'PER', 'EPS', 'Updated'
            ]
        
        metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history_cols = [f"{m}_{y}" for m in metrics for y in ["Y4", "Y3", "Y2", "Y1", "TTM"]]
        final_cols = base_cols + history_cols

        for idx, symbol in enumerate(tickers):
            status.markdown(f"### ⏳ 분석 중: **{symbol}** ({idx+1} / {total})")
            
            # 재무 데이터 추출 (원본 그대로)
            data = get_extended_financials(symbol)
            
            # AI 분석 (선택사항)
            if enable_ai:
                ai_grade, ai_reason = analyze_with_ai(symbol, data[:13], llm_provider)
                row = [symbol, ai_grade, ai_reason] + data[:13] + [datetime.now().strftime('%H:%M:%S')] + data[13:]
            else:
                # 원본 방식
                row = [symbol] + data[:13] + [datetime.now().strftime('%H:%M:%S')] + data[13:]
            
            results.append(row)
            prog.progress((idx+1)/total)
            time.sleep(0.5 if not enable_ai else 2)

        status.success(f"✅ 분석 완료!")
        res_df = pd.DataFrame(results, columns=final_cols).fillna("-")
        st.dataframe(res_df, use_container_width=True)
        
        # CSV 다운로드
        csv_filename = f"financial_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        st.download_button(
            "📥 결과 CSV 다운로드", 
            res_df.to_csv(index=False).encode('utf-8'), 
            csv_filename, 
            "text/csv"
        )
