import streamlit as st
import yfinance as yf
import pandas as pd
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

        # 1. TTM (최근 12개월) 데이터 추출
        ttm_dte = info.get("debtToEquity")
        ttm_cr = (info.get("currentRatio") * 100) if info.get("currentRatio") else None
        ttm_opm = (info.get("operatingMargins") * 100) if info.get("operatingMargins") else None
        ttm_roe = (info.get("returnOnEquity") * 100) if info.get("returnOnEquity") else None
        ttm_ocf = info.get("operatingCashflow")
        ttm_net_income = info.get("netIncomeToCommon")
        
        # CF Quality (TTM)
        ttm_cf_quality = round(ttm_ocf / ttm_net_income, 2) if ttm_ocf and ttm_net_income and ttm_net_income != 0 else None
        
        ttm_pbr = info.get("priceToBook")
        ttm_per = info.get("trailingPE")
        ttm_eps = info.get("trailingEps")

        # Runway 계산
        total_cash = info.get("totalCash")
        free_cf = info.get("freeCashflow")
        if total_cash and free_cf:
            runway = round(total_cash / abs(free_cf), 2) if free_cf < 0 else "Infinite (Profit)"
        else:
            runway = None

        # 기본 10개 지표 리스트 (CFQ는 가독성을 위해 EPS 근처로 배치 가능하나, 요청대로 마지막 추이에 맞춤)
        base_results = [
            round(ttm_dte, 2) if ttm_dte is not None else None,
            round(ttm_cr, 2) if ttm_cr is not None else None,
            round(ttm_opm, 2) if ttm_opm is not None else None,
            round(ttm_roe, 2) if ttm_roe is not None else None,
            runway,
            round(ttm_ocf / 1_000_000, 2) if ttm_ocf else None,
            round(ttm_pbr, 2) if ttm_pbr is not None else None,
            round(ttm_per, 2) if ttm_per is not None else None,
            round(ttm_eps, 2) if ttm_eps is not None else None,
            ttm_cf_quality # 기본 지표 리스트에서도 마지막에 배치
        ]

        # 2. 항목별 추이 데이터 (순서: DTE, CR, OPM, ROE, OCF, EPS, CFQ)
        metrics_order = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ"]
        history = {m: [None]*4 for m in metrics_order}

        num_years = min(len(fin.columns), 4) if not fin.empty else 0

        for i in range(num_years):
            idx = 3 - i 
            
            # 재무 데이터 추출
            liab = get_val(bs, 'Total Liabilities Net Minority Interest', i)
            equity = get_val(bs, 'Total Equity Gross Minority Interest', i)
            net_inc = get_val(fin, 'Net Income', i)
            ocf = get_val(cf, 'Operating Cash Flow', i)
            
            # 항목별 계산 (Y4 -> Y1)
            history["DTE"][idx] = round((liab/equity*100), 2) if liab and equity else None
            history["CR"][idx] = round((get_val(bs, 'Current Assets', i)/get_val(bs, 'Current Liabilities', i)*100), 2) if get_val(bs, 'Current Assets', i) and get_val(bs, 'Current Liabilities', i) else None
            history["OPM"][idx] = round((get_val(fin, 'Operating Income', i)/get_val(fin, 'Total Revenue', i)*100), 2) if get_val(fin, 'Operating Income', i) and get_val(fin, 'Total Revenue', i) else None
            history["ROE"][idx] = round((net_inc/equity*100), 2) if net_inc and equity else None
            history["OCF"][idx] = round(ocf/1_000_000, 2) if ocf else None
            history["EPS"][idx] = round(get_val(fin, 'Basic EPS', i), 2) if get_val(fin, 'Basic EPS', i) else None
            history["CFQ"][idx] = round(ocf/net_inc, 2) if ocf and net_inc and net_inc != 0 else None

        # 최종 조합: 기본결과(10개) + [항목별(7개) * 시점(5개)]
        flattened_history = []
        ttm_map = {"DTE": 0, "CR": 1, "OPM": 2, "ROE": 3, "OCF": 5, "EPS": 8, "CFQ": 9}
        
        for key in metrics_order:
            combined = history[key] + [base_results[ttm_map[key]]]
            flattened_history.extend(combined)

        return base_results + flattened_history
    except Exception:
        return [None] * (10 + 35)

# --- [UI] Streamlit 설정 ---
st.set_page_config(page_title="Stock Analysis Pro", layout="wide")
st.title("📊 재무 추이 및 이익의 질 분석 (Y4 → TTM)")

# --- [사이드바] 데이터 입력 ---
st.sidebar.header("📥 데이터 설정")
method = st.sidebar.radio("입력 방식", ("텍스트 붙여넣기", "구글 스프레드시트", "CSV 파일 업로드"))

tickers = []
if method == "텍스트 붙여넣기":
    raw = st.sidebar.text_area("티커 입력 (한 줄에 하나)")
    if raw: tickers = [t.strip().upper() for t in raw.split('\n') if t.strip()]
elif method == "구글 스프레드시트":
    try:
        sid, sname = st.secrets["GOOGLE_SHEET_ID"], st.secrets["GOOGLE_SHEET_NAME"]
        url = f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={quote(sname)}"
        gs_df = pd.read_csv(url)
        t_col = st.sidebar.selectbox("티커 열 선택", gs_df.columns)
        tickers = gs_df[t_col].dropna().astype(str).tolist()
    except Exception as e: st.sidebar.error(f"연결 오류: {e}")
elif method == "CSV 파일 업로드":
    up = st.sidebar.file_uploader("CSV 선택", type=["csv"])
    if up:
        df = pd.read_csv(up)
        t_col = st.sidebar.selectbox("티커 열 선택", df.columns)
        tickers = df[t_col].dropna().astype(str).tolist()

# --- [메인] 분석 실행 ---
if tickers:
    total_count = len(tickers)
    st.write(f"📝 대상 종목: **{total_count}개**")
    
    if st.button("전수 분석 시작"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        
        # 칼럼 헤더 정의 (CFQ를 가장 뒤로 배치)
        base_cols = ['ticker', 'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Runway(Y)', 'OCF(M$)', 'PBR', 'PER', 'EPS', 'CF_Quality', 'Updated']
        metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ"]
        years = ["Y4", "Y3", "Y2", "Y1", "TTM"]
        history_cols = [f"{m}_{y}" for m in metrics for y in years]
        final_cols = base_cols + history_cols

        for idx, symbol in enumerate(tickers):
            status_text.markdown(f"### ⏳ 분석 중: **{symbol}** ({idx + 1} / {total_count})")
            data = get_extended_financials(symbol)
            # data: [10개 기본] + [35개 히스토리]
            row = [symbol] + data[:10] + [datetime.now().strftime('%H:%M:%S')] + data[10:]
            results.append(row)
            progress_bar.progress((idx + 1) / total_count)
            time.sleep(0.5)

        status_text.success(f"✅ 총 {total_count}개 종목 분석 완료!")
        res_df = pd.DataFrame(results, columns=final_cols).fillna("-")
        st.dataframe(res_df, use_container_width=True)
        st.download_button("결과 CSV 다운로드", res_df.to_csv(index=False).encode('utf-8'), f"analysis_{datetime.now().strftime('%m%d')}.csv", "text/csv")
else:
    st.info("👈 왼쪽 사이드바에서 분석할 종목(티커)을 제공해주세요.")
