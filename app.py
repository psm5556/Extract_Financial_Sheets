import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from urllib.parse import quote
import google.generativeai as genai
import json

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

# --- [함수] LLM 기반 투자 등급 분석 ---
def analyze_stock_with_llm(ticker, financial_data):
    """
    재무 데이터를 LLM에 전달하여 투자 등급(A~F) + 이유 반환
    """
    try:
        # Gemini API 설정
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            return "N/A", "API 키가 설정되지 않았습니다."
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 재무 데이터 딕셔너리 구성
        metrics = {
            "Ticker": ticker,
            "DTE(%)": financial_data[0],
            "CR(%)": financial_data[1],
            "OPM(%)": financial_data[2],
            "ROE(%)": financial_data[3],
            "Runway(Y)": financial_data[4],
            "TotalCash(M$)": financial_data[5],
            "FCF(M$)": financial_data[6],
            "FCF_Stability(%)": financial_data[7],
            "OCF(M$)": financial_data[8],
            "PBR": financial_data[9],
            "BPS": financial_data[10],
            "PER": financial_data[11],
            "EPS": financial_data[12]
        }
        
        # 프롬프트 구성
        prompt = f"""
You are a professional financial analyst. Analyze the following stock's financial metrics and provide:
1. Investment Grade: A (Excellent) / B (Good) / C (Average) / D (Below Average) / F (Poor)
2. Brief Reason (50 words max, Korean)

Financial Data for {ticker}:
{json.dumps(metrics, indent=2)}

Evaluation Criteria:
- PER < 15, PBR < 2: Undervalued
- ROE > 15%, OPM > 10%: Strong profitability
- FCF_Stability > 80%, Positive FCF: Healthy cash flow
- DTE < 100%, CR > 150%: Solid financial structure
- Runway > 5 years or Infinite: Good sustainability

Return ONLY in this JSON format:
{{"grade": "A/B/C/D/F", "reason": "Korean explanation"}}
"""
        
        response = model.generate_content(prompt)
        result = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
        
        return result.get("grade", "N/A"), result.get("reason", "분석 실패")
    
    except Exception as e:
        return "ERROR", f"분석 오류: {str(e)[:50]}"

# --- [UI] Streamlit 설정 ---
st.set_page_config(page_title="Stock Master Analyzer with AI", layout="wide")
st.title("📊 AI 투자 등급 분석 시스템 (Y4 → TTM)")

# --- [사이드바] ---
st.sidebar.header("📥 데이터 소스")
method = st.sidebar.radio("방식", ("텍스트 붙여넣기", "구글 스프레드시트", "CSV 파일 업로드"))

st.sidebar.markdown("---")
st.sidebar.header("🤖 AI 분석 옵션")
enable_ai = st.sidebar.checkbox("AI 투자 등급 분석 활성화", value=True)
if enable_ai:
    st.sidebar.info("💡 Gemini API 키가 필요합니다 (secrets.toml 설정)")

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

# --- [메인] 분석 실행 ---
if tickers:
    total = len(tickers)
    if st.button("🚀 전수 분석 시작"):
        prog = st.progress(0); status = st.empty(); results = []
        
        # 헤더 정의 (AI 등급 추가)
        base_cols = [
            'ticker', 'AI_Grade', 'AI_Reason',  # AI 분석 결과 추가
            'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Runway(Y)', 
            'TotalCash(M$)', 'FCF(M$)', 'FCF_Stability(%)', 'OCF(M$)', 
            'PBR', 'BPS', 'PER', 'EPS', 'Updated'
        ]
        
        metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history_cols = [f"{m}_{y}" for m in metrics for y in ["Y4", "Y3", "Y2", "Y1", "TTM"]]
        final_cols = base_cols + history_cols

        for idx, symbol in enumerate(tickers):
            status.markdown(f"### ⏳ 분석 중: **{symbol}** ({idx+1} / {total})")
            
            # 재무 데이터 추출
            data = get_extended_financials(symbol)
            
            # AI 등급 분석 (옵션)
            if enable_ai:
                grade, reason = analyze_stock_with_llm(symbol, data[:13])
            else:
                grade, reason = "-", "-"
            
            # row: [ticker] + [AI등급,이유] + [기본13개] + [시간] + [추이40개]
            row = [symbol, grade, reason] + data[:13] + [datetime.now().strftime('%H:%M:%S')] + data[13:]
            results.append(row)
            
            prog.progress((idx+1)/total)
            time.sleep(0.5)  # API 호출 제한 고려

        status.success(f"✅ 분석 완료!")
        res_df = pd.DataFrame(results, columns=final_cols).fillna("-")
        
        # 등급별 색상 표시를 위한 스타일링
        def highlight_grade(val):
            color_map = {
                'A': 'background-color: #d4edda; color: #155724',
                'B': 'background-color: #d1ecf1; color: #0c5460',
                'C': 'background-color: #fff3cd; color: #856404',
                'D': 'background-color: #f8d7da; color: #721c24',
                'F': 'background-color: #f5c6cb; color: #721c24'
            }
            return color_map.get(val, '')
        
        st.dataframe(
            res_df.style.applymap(highlight_grade, subset=['AI_Grade']),
            use_container_width=True
        )
        
        st.download_button(
            "📥 결과 CSV 다운로드", 
            res_df.to_csv(index=False).encode('utf-8'), 
            "financial_analysis_with_ai.csv", 
            "text/csv"
        )
        
        # 등급 분포 통계
        if enable_ai:
            st.markdown("### 📈 AI 등급 분포")
            grade_counts = res_df['AI_Grade'].value_counts()
            st.bar_chart(grade_counts)
