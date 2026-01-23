import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from urllib.parse import quote
from anthropic import Anthropic

# --- [함수] Claude AI 투자 분석 생성 ---
def generate_claude_analysis(ticker, data_summary):
    """
    Anthropic Claude API를 사용하여 정제된 재무 데이터를 바탕으로 전문 리포트를 생성합니다.
    """
    try:
        # st.secrets에서 API 키 로드
        client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        
        prompt = f"""
        당신은 월스트리트 출신의 전문 주식 분석가입니다. 
        다음 기업({ticker})의 최신 재무 데이터를 바탕으로 전문적인 투자 의견을 한국어로 작성하세요.
        
        [기업 핵심 재무 데이터]
        - 부채비율(DTE): {data_summary.get('DTE')}%
        - ROE(자기자본이익률): {data_summary.get('ROE')}%
        - OPM(영업이익률): {data_summary.get('OPM')}%
        - FCF Stability(5년간 현금흐름 유지력): {data_summary.get('Stability')}%
        - Cash Flow Quality(이익의 질): {data_summary.get('CFQ')}
        - Runway(현금 보유 기간): {data_summary.get('Runway')}
        - 밸류에이션: PBR {data_summary.get('PBR')} / PER {data_summary.get('PER')}
        
        [작성 가이드라인]
        1. 첫 줄에 투자 등급을 명시하세요 (💎강력매수 / ✅매수 / 🟡보유 / 🚨주의).
        2. 재무 건전성과 현금흐름의 질에 대해 날카로운 비평을 남기세요.
        3. 수치 이면에 숨겨진 리스크나 기회를 한 문장으로 언급하세요.
        문체는 '전문적이고 간결한 평어체'를 사용하세요.
        """

        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=400,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"AI 분석 불가 (오류: {str(e)})"

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

        # 2. 항목별 5개년 추이 수집 (Y4 -> TTM)
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

        # 🤖 Claude AI 분석 데이터 요약 및 실행
        ai_data_summary = {
            'DTE': ttm_dte, 'ROE': ttm_roe, 'OPM': ttm_opm, 
            'Stability': stability, 'CFQ': ttm_cfq, 'Runway': runway,
            'PBR': info.get("priceToBook"), 'PER': info.get("trailingPE")
        }
        ai_opinion = generate_claude_analysis(symbol, ai_data_summary)

        # 3. 요약 섹션 결과 패킹
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
            round(info.get("trailingEps"), 2) if info.get("trailingEps") else None,
            ai_opinion # AI 분석 결과
        ]

        # 4. 시계열 추이 패킹
        ttm_vals_map = {
            "DTE": base_results[0], "CR": base_results[1], "OPM": base_results[2], 
            "ROE": base_results[3], "OCF": base_results[8], "EPS": base_results[12],
            "CFQ": ttm_cfq, "FCF": ttm_fcf_m
        }
        
        flattened_history = []
        for key in metrics_order:
            combined = history[key] + [ttm_vals_map[key]]
            flattened_history.extend(combined)

        return base_results + flattened_history
    except Exception:
        return [None] * (14 + 40)

# --- [UI] Streamlit 설정 ---
st.set_page_config(page_title="Claude AI Financial Analyst", layout="wide")
st.title("📊 Claude 3.5 기반 주식 재무 심층 분석")

# --- [사이드바] ---
st.sidebar.header("📥 입력 설정")
method = st.sidebar.radio("방식", ("텍스트 붙여넣기", "구글 스프레드시트", "CSV 파일 업로드"))
tickers = []
if method == "텍스트 붙여넣기":
    raw = st.sidebar.text_area("티커 입력 (예: TSLA\nNVDA)")
    if raw: tickers = [t.strip().upper() for t in raw.split('\n') if t.strip()]
elif method == "구글 스프레드시트":
    try:
        sid, sname = st.secrets["GOOGLE_SHEET_ID"], st.secrets["GOOGLE_SHEET_NAME"]
        url = f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={quote(sname)}"
        gs_df = pd.read_csv(url); t_col = st.sidebar.selectbox("티커 컬럼", gs_df.columns)
        tickers = gs_df[t_col].dropna().astype(str).tolist()
    except: st.sidebar.error("시트 연결 오류")
elif method == "CSV 파일 업로드":
    up = st.sidebar.file_uploader("CSV", type=["csv"])
    if up:
        df = pd.read_csv(up); t_col = st.sidebar.selectbox("티커 컬럼", df.columns)
        tickers = df[t_col].dropna().astype(str).tolist()

# --- [메인] 분석 실행 ---
if tickers:
    total = len(tickers)
    if st.button("🚀 Claude AI 전수 분석 시작"):
        prog = st.progress(0); status = st.empty(); results = []
        
        # 헤더 정의
        base_cols = [
            'ticker', 'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Runway(Y)', 
            'TotalCash(M$)', 'FCF(M$)', 'FCF_Stability(%)', 'OCF(M$)', 
            'PBR', 'BPS', 'PER', 'EPS', 'Claude_Opinion', 'Updated'
        ]
        
        metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history_cols = [f"{m}_{y}" for m in metrics for y in ["Y4", "Y3", "Y2", "Y1", "TTM"]]
        final_cols = base_cols + history_cols

        for idx, symbol in enumerate(tickers):
            status.markdown(f"### ⏳ Claude 3.5가 **{symbol}** 분석 중... ({idx+1}/{total})")
            data = get_extended_financials(symbol)
            
            # row: [ticker] + [기본14개] + [시간] + [추이40개]
            row = [symbol] + data[:14] + [datetime.now().strftime('%H:%M:%S')] + data[14:]
            results.append(row)
            prog.progress((idx+1)/total)
            time.sleep(1) # API Rate Limit 방지

        status.success(f"✅ {total}개 종목에 대한 Claude AI 분석 리포트가 생성되었습니다!")
        res_df = pd.DataFrame(results, columns=final_cols).fillna("-")
        st.dataframe(res_df, use_container_width=True)
        st.download_button("📥 AI 분석 결과 CSV 다운로드", res_df.to_csv(index=False).encode('utf-8'), "claude_stock_analysis.csv", "text/csv")
else:
    st.info("👈 분석할 티커를 입력해주세요.")
