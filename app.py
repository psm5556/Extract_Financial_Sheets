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
        
        # 데이터 로드
        info = ticker.info
        fin = ticker.financials      # 손익계산서
        bs = ticker.balance_sheet    # 대차대조표
        cf = ticker.cashflow         # 현금흐름표

        def get_val(df, label, idx):
            try: return df.loc[label].iloc[idx]
            except: return None

        # 1. 기존 기본 지표 (TTM/실시간 기반) - 7개 항목
        base_data = [
            info.get("debtToEquity"),
            (info.get("currentRatio") * 100) if info.get("currentRatio") else None,
            (info.get("operatingMargins") * 100) if info.get("operatingMargins") else None,
            (info.get("operatingCashflow") / 1_000_000) if info.get("operatingCashflow") else None,
            info.get("priceToBook"),
            info.get("trailingPE"),
            info.get("trailingEps")
        ]
        base_results = [round(v, 2) if v is not None else None for v in base_data]

        # 2. 항목별 5개년 데이터 (DTE, CR, OPM, OCF, EPS) - 각 5년씩 총 25개 항목
        history = { "DTE": [], "CR": [], "OPM": [], "OCF": [], "EPS": [] }
        num_years = min(len(fin.columns), 5) if not fin.empty else 0

        for i in range(5):
            if i < num_years:
                # DTE (부채비율)
                liab = get_val(bs, 'Total Liabilities Net Minority Interest', i)
                equity = get_val(bs, 'Total Equity Gross Minority Interest', i)
                history["DTE"].append(round((liab/equity*100), 2) if liab and equity else None)
                # CR (유동비율)
                ca = get_val(bs, 'Current Assets', i)
                cl = get_val(bs, 'Current Liabilities', i)
                history["CR"].append(round((ca/cl*100), 2) if ca and cl else None)
                # OPM (영업이익률)
                op_inc = get_val(fin, 'Operating Income', i)
                rev = get_val(fin, 'Total Revenue', i)
                history["OPM"].append(round((op_inc/rev*100), 2) if op_inc and rev else None)
                # OCF (영업현금흐름 M$)
                ocf = get_val(cf, 'Operating Cash Flow', i)
                history["OCF"].append(round(ocf/1_000_000, 2) if ocf else None)
                # EPS (주당순이익)
                eps = get_val(fin, 'Basic EPS', i)
                history["EPS"].append(round(eps, 2) if eps else None)
            else:
                for key in history: history[key].append(None)

        # 3. 항목별 평탄화 (DTE_Y1..5, CR_Y1..5 순서)
        flattened_history = []
        for key in ["DTE", "CR", "OPM", "OCF", "EPS"]:
            flattened_history.extend(history[key])

        return base_results + flattened_history
    except Exception:
        return [None] * (7 + 25)

# --- [UI] Streamlit 앱 설정 ---
st.set_page_config(page_title="Stock Analysis Pro", layout="wide")
st.title("📊 재무 지표 시계열 분석기")

# --- [사이드바] 입력 설정 ---
st.sidebar.header("📥 데이터 소스 설정")
input_method = st.sidebar.radio("입력 방식을 선택하세요", ("텍스트 붙여넣기", "구글 스프레드시트", "CSV 파일 업로드"))

tickers = []
if input_method == "텍스트 붙여넣기":
    raw_input = st.sidebar.text_area("티커 입력 (한 줄에 하나씩)", height=200)
    if raw_input: tickers = [t.strip().upper() for t in raw_input.split('\n') if t.strip()]
elif input_method == "구글 스프레드시트":
    try:
        sid, sname = st.secrets["GOOGLE_SHEET_ID"], st.secrets["GOOGLE_SHEET_NAME"]
        url = f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={quote(sname)}"
        gs_df = pd.read_csv(url)
        st.sidebar.success(f"✅ 연결 성공: {sname}")
        t_col = st.sidebar.selectbox("티커 열 선택", gs_df.columns)
        tickers = gs_df[t_col].dropna().astype(str).tolist()
    except Exception as e: st.sidebar.error(f"❌ 시트 로드 실패: {e}")
elif input_method == "CSV 파일 업로드":
    up_file = st.sidebar.file_uploader("CSV 업로드", type=["csv"])
    if up_file:
        df = pd.read_csv(up_file)
        t_col = st.sidebar.selectbox("티커 열 선택", df.columns)
        tickers = df[t_col].dropna().astype(str).tolist()

# --- [메인] 분석 실행 및 결과 ---
if tickers:
    st.write(f"📝 분석 대상: **{len(tickers)}개 종목**")
    if st.button("전수 분석 시작"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_list = []
        
        # 1. 칼럼 헤더 생성
        # 기본 7개 항목
        base_cols = ['ticker', 'debtToEquity(%)', 'currentRatio(%)', 'OperatingMargin(%)', 
                     'OperatingCashflow(M$)', 'PBR', 'PER', 'EPS', 'lastUpdated']
        # 항목별 5개년 항목
        metrics_5y = ["DTE", "CR", "OPM", "OCF", "EPS"]
        history_cols = [f"{m}_Y{y}" for m in metrics_5y for y in range(1, 6)]
        final_cols = base_cols + history_cols

        for idx, symbol in enumerate(tickers):
            status_text.text(f"⏳ {symbol} 분석 중... ({idx+1}/{len(tickers)})")
            raw_data = get_extended_financials(symbol)
            
            # 행 데이터 재조합: [티커] + [기본7개] + [시간] + [5개년25개]
            row = [symbol] + raw_data[:7] + [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] + raw_data[7:]
            results_list.append(row)
            
            progress_bar.progress((idx + 1) / len(tickers))
            time.sleep(0.5)

        res_df = pd.DataFrame(results_list, columns=final_cols)
        st.success("✅ 분석이 완료되었습니다!")
        st.dataframe(res_df, use_container_width=True)

        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button("결과 CSV 다운로드", csv, f"financial_full_{datetime.now().strftime('%m%d')}.csv", "text/csv")
else:
    st.warning("👈 사이드바에서 티커 목록을 먼저 입력해 주세요.")
