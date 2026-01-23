import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
from urllib.parse import quote

# --- [함수] 재무 데이터 추출 로직 ---
def get_extended_financials(ticker_symbol):
    """
    최근 12개월(TTM) 및 최근 5개년 연간 재무 지표를 추출합니다.
    항목: 부채비율, 유동비율, 영업이익률, 영업현금흐름, PBR, PER, EPS
    """
    try:
        symbol = ticker_symbol.upper().strip()
        ticker = yf.Ticker(symbol)
        
        # 데이터 로드
        info = ticker.info
        fin = ticker.financials      # 손익계산서
        bs = ticker.balance_sheet    # 대차대조표
        cf = ticker.cashflow         # 현금흐름표

        def get_val(df, label, idx):
            try:
                return df.loc[label].iloc[idx]
            except:
                return None

        # 1. TTM (최근 12개월) 데이터 구성
        ttm_data = [
            info.get("debtToEquity"),                                      # DTE
            (info.get("currentRatio") * 100) if info.get("currentRatio") else None, # CR
            (info.get("operatingMargins") * 100) if info.get("operatingMargins") else None, # OPM
            (info.get("operatingCashflow") / 1_000_000) if info.get("operatingCashflow") else None, # OCF (M$)
            info.get("priceToBook"),                                       # PBR
            info.get("trailingPE"),                                        # PER
            info.get("trailingEps")                                        # EPS
        ]
        
        # 반올림 처리
        all_results = [round(v, 2) if v is not None else None for v in ttm_data]

        # 2. 최근 5개년(Y1~Y5) 데이터 구성
        # 연간 데이터프레임의 열 개수를 확인하여 진행
        num_years = min(len(fin.columns), 5)
        
        for i in range(5):
            if i < num_years:
                try:
                    # 부채비율 (Total Liab / Total Equity)
                    liab = get_val(bs, 'Total Liabilities Net Minority Interest', i)
                    equity = get_val(bs, 'Total Equity Gross Minority Interest', i)
                    dte = (liab / equity * 100) if liab and equity else None
                    
                    # 유동비율 (Current Assets / Current Liab)
                    ca = get_val(bs, 'Current Assets', i)
                    cl = get_val(bs, 'Current Liabilities', i)
                    cr = (ca / cl * 100) if ca and cl else None
                    
                    # 영업이익률 (Op Income / Revenue)
                    op_inc = get_val(fin, 'Operating Income', i)
                    rev = get_val(fin, 'Total Revenue', i)
                    opm = (op_inc / rev * 100) if op_inc and rev else None
                    
                    # 영업현금흐름 (M$)
                    ocf = get_val(cf, 'Operating Cash Flow', i)
                    ocf_m = (ocf / 1_000_000) if ocf else None
                    
                    # EPS (Basic EPS)
                    eps = get_val(fin, 'Basic EPS', i)
                    
                    # 과거 PBR, PER은 시점별 주가 데이터가 추가로 필요하므로 None 처리
                    all_results.extend([
                        round(dte, 2) if dte is not None else None,
                        round(cr, 2) if cr is not None else None,
                        round(opm, 2) if opm is not None else None,
                        round(ocf_m, 2) if ocf_m is not None else None,
                        None, None, # PBR, PER
                        round(eps, 2) if eps is not None else None
                    ])
                except:
                    all_results.extend([None] * 7)
            else:
                # 데이터가 없는 연도는 None으로 채움
                all_results.extend([None] * 7)

        return all_results
    except Exception:
        return [None] * 42 # 7개 지표 * 6개 시점(TTM + 5Y)

# --- [UI] Streamlit 앱 설정 ---
st.set_page_config(page_title="Stock Deep Analyzer", layout="wide")
st.title("🚀 5개년 재무 추이 전수 분석기")

# --- [사이드바] 입력 방식 설정 ---
st.sidebar.header("📥 데이터 소스 설정")
input_method = st.sidebar.radio(
    "입력 방식을 선택하세요",
    ("텍스트 붙여넣기", "구글 스프레드시트", "CSV 파일 업로드")
)

tickers = []

if input_method == "텍스트 붙여넣기":
    raw_input = st.sidebar.text_area("티커를 입력하세요 (한 줄에 하나씩)", height=200, placeholder="AAPL\nTSLA\nNVDA")
    if raw_input:
        tickers = [t.strip().upper() for t in raw_input.split('\n') if t.strip()]

elif input_method == "구글 스프레드시트":
    try:
        sheet_id = st.secrets["GOOGLE_SHEET_ID"]
        sheet_name = st.secrets["GOOGLE_SHEET_NAME"]
        encoded_sheet_name = quote(sheet_name)
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={encoded_sheet_name}"
        
        gs_df = pd.read_csv(url)
        st.sidebar.success(f"✅ 연결 성공: {sheet_name}")
        ticker_col = st.sidebar.selectbox("티커 열 선택", gs_df.columns)
        tickers = gs_df[ticker_col].dropna().astype(str).tolist()
    except Exception as e:
        st.sidebar.error(f"❌ 시트 로드 실패: {e}")

elif input_method == "CSV 파일 업로드":
    uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        ticker_col = st.sidebar.selectbox("티커 열 선택", df.columns)
        tickers = df[ticker_col].dropna().astype(str).tolist()

# --- [메인] 실행 및 결과 출력 ---
if tickers:
    st.write(f"📝 분석 대상: **{len(tickers)}개 종목**")
    st.info("💡 5개년치 재무제표를 모두 분석하므로 종목당 약 2~3초가 소요됩니다.")

    if st.button("전수 분석 시작"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_list = []
        
        # 컬럼 이름 생성 루프
        metrics = ["DTE(%)", "CR(%)", "OPM(%)", "OCF(M$)", "PBR", "PER", "EPS"]
        periods = ["TTM", "Y1(최근)", "Y2", "Y3", "Y4", "Y5"]
        cols = ['ticker']
        for p in periods:
            for m in metrics:
                cols.append(f"{p}_{m}")
        cols.append("lastUpdated")

        # 분석 루프
        for idx, symbol in enumerate(tickers):
            status_text.text(f"⏳ {symbol} 재무제표 분석 중... ({idx+1}/{len(tickers)})")
            data = get_extended_financials(symbol)
            results_list.append([symbol] + data + [datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            progress_bar.progress((idx + 1) / len(tickers))
            time.sleep(0.5)

        # 결과 데이터프레임
        res_df = pd.DataFrame(results_list, columns=cols)
        
        st.success("✅ 모든 분석이 완료되었습니다!")
        st.dataframe(res_df, use_container_width=True)

        # CSV 다운로드 버튼
        csv_data = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="결과 CSV 다운로드",
            data=csv_data,
            file_name=f"financial_5y_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime='text/csv'
        )
else:
    st.warning("👈 사이드바에서 분석할 티커 목록을 먼저 제공해주세요.")
