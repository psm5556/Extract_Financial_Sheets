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
        fin = ticker.financials      # 손익계산서
        bs = ticker.balance_sheet    # 대차대조표
        cf = ticker.cashflow         # 현금흐름표

        def get_val(df, label, idx):
            try: return df.loc[label].iloc[idx]
            except: return None

        # 1. 기존 기본 지표 (TTM/실시간 기반)
        # 순서: DTE, CR, OPM, OCF, PBR, PER, EPS
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

        # 2. 항목별 5개년 데이터 준비 (DTE, CR, OPM, OCF, EPS 각 5년치)
        # 구조: {항목명: [Y1, Y2, Y3, Y4, Y5]}
        history = { "DTE": [], "CR": [], "OPM": [], "OCF": [], "EPS": [] }
        
        num_years = min(len(fin.columns), 5) if not fin.empty else 0

        for i in range(5):
            if i < num_years:
                # DTE
                liab = get_val(bs, 'Total Liabilities Net Minority Interest', i)
                equity = get_val(bs, 'Total Equity Gross Minority Interest', i)
                history["DTE"].append(round((liab/equity*100), 2) if liab and equity else None)
                # CR
                ca = get_val(bs, 'Current Assets', i)
                cl = get_val(bs, 'Current Liabilities', i)
                history["CR"].append(round((ca/cl*100), 2) if ca and cl else None)
                # OPM
                op_inc = get_val(fin, 'Operating Income', i)
                rev = get_val(fin, 'Total Revenue', i)
                history["OPM"].append(round((op_inc/rev*100), 2) if op_inc and rev else None)
                # OCF
                ocf = get_val(cf, 'Operating Cash Flow', i)
                history["OCF"].append(round(ocf/1_000_000, 2) if ocf else None)
                # EPS
                eps = get_val(fin, 'Basic EPS', i)
                history["EPS"].append(round(eps, 2) if eps else None)
            else:
                for key in history: history[key].append(None)

        # 3. 데이터 결합: 기본지표 + (항목별 5년치 평탄화)
        # 평탄화 순서: DTE_Y1~5, CR_Y1~5, OPM_Y1~5 ...
        flattened_history = []
        for key in ["DTE", "CR", "OPM", "OCF", "EPS"]:
            flattened_history.extend(history[key])

        return base_results + flattened_history
    except Exception:
        return [None] * (7 + 25) # 기본 7개 + (5항목 * 5년)

# --- [UI] Streamlit 앱 ---
st.set_page_config(page_title="Stock Analysis Pro", layout="wide")
st.title("📊 재무 지표 시계열 분석기")

# (사이드바 입력 로직 생략 - 이전과 동일)
# ... [이전 코드의 사이드바 섹션 유지] ...

if tickers:
    st.write(f"📝 분석 대상: **{len(tickers)}개 종목**")
    
    if st.button("데이터 전수 조사 시작"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_list = []
        
        # --- 칼럼 이름 정의 (요청하신 순서대로) ---
        # 1. 기본 칼럼
        cols = ['ticker', 'debtToEquity(%)', 'currentRatio(%)', 'OperatingMargin(%)', 
                'OperatingCashflow(M$)', 'PBR', 'PER', 'EPS', 'lastUpdated']
        
        # 2. 항목별 5개년 칼럼 추가 (lastUpdated 뒤로 붙음)
        metrics_5y = ["DTE", "CR", "OPM", "OCF", "EPS"]
        for m in metrics_5y:
            for y in range(1, 6):
                cols.append(f"{m}_Y{y}")

        for idx, symbol in enumerate(tickers):
            status_text.text(f"⏳ {symbol} 분석 중... ({idx+1}/{len(tickers)})")
            data = get_extended_financials(symbol)
            
            # 데이터 배치: [ticker] + [기본7개] + [업데이트시간] + [5개년25개]
            # data에는 [기본7개] + [5개년25개]가 들어있음
            final_row = [symbol] + data[:7] + [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] + data[7:]
            results_list.append(final_row)
            
            progress_bar.progress((idx + 1) / len(tickers))
            time.sleep(0.5)

        res_df = pd.DataFrame(results_list, columns=cols)
        st.success("✅ 분석 완료!")
        st.dataframe(res_df, use_container_width=True)

        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button("결과 CSV 다운로드", csv, f"financial_report_{datetime.now().strftime('%m%d')}.csv", "text/csv")
