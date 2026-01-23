import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time

# --- 함수 정의: 재무 지표 가져오기 ---
def get_financial_ratios(ticker_symbol):
    try:
        # yfinance는 대문자를 선호합니다
        ticker = yf.Ticker(ticker_symbol.upper().strip())
        info = ticker.info

        # 데이터 추출 (값이 없으면 None 반환)
        dte = info.get("debtToEquity")
        cr = info.get("currentRatio")
        roe = info.get("returnOnEquity")
        total_cash = info.get("totalCash")
        free_cf = info.get("freeCashflow")
        operating_cf = info.get("operatingCashflow")
        net_income = info.get("netIncomeToCommon")
        pbr = info.get("priceToBook")
        bps = info.get("bookValue")

        # 단위 변환 및 반올림
        cr = round(cr * 100, 2) if cr else None
        roe = round(roe * 100, 2) if roe else None
        
        def to_million(val):
            return round(val / 1_000_000, 2) if val else None

        total_cash_m = to_million(total_cash)
        free_cf_m = to_million(free_cf)
        operating_cf_m = to_million(operating_cf)
        net_income_m = to_million(net_income)
        
        pbr = round(pbr, 2) if pbr else None
        bps = round(bps, 2) if bps else None

        # Runway 계산
        runway_years = None
        if total_cash and free_cf:
            if free_cf < 0:
                runway_years = round(total_cash / abs(free_cf), 2)
            else:
                runway_years = float('inf')

        return dte, cr, roe, runway_years, total_cash_m, free_cf_m, operating_cf_m, net_income_m, pbr, bps

    except Exception as e:
        st.error(f"⚠️ {ticker_symbol} 데이터 오류: {e}")
        return [None] * 10

# --- Streamlit UI ---
st.title("📈 주식 재무 지표 대시보드")
st.markdown("CSV 파일을 업로드하면 Yahoo Finance에서 재무 지표를 가져옵니다.")

# 1. 파일 업로드
uploaded_file = st.file_uploader("티커가 포함된 CSV 파일을 업로드하세요 (컬럼명: ticker)", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    
    if 'ticker' not in input_df.columns:
        st.error("CSV 파일에 'ticker' 컬럼이 없습니다.")
    else:
        tickers = input_df['ticker'].tolist()
        results = []

        if st.button("데이터 불러오기 시작"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, symbol in enumerate(tickers):
                status_text.text(f"처리 중: {symbol} ({idx+1}/{len(tickers)})")
                
                # 데이터 가져오기
                data = get_financial_ratios(symbol)
                results.append([symbol] + list(data) + [datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                
                # 진행률 업데이트
                progress_bar.progress((idx + 1) / len(tickers))
                time.sleep(0.5) # API 과부하 방지

            # 결과 데이터프레임 생성
            columns = [
                'ticker', 'debtToEquity(%)', 'currentRatio(%)', 'ROE(%)', 
                'Runway(Years)', 'TotalCash(M$)', 'FreeCashflow(M$)', 
                'OperatingCashflow(M$)', 'NetIncome(M$)', 'PBR', 'BPS($)', 'lastUpdated'
            ]
            res_df = pd.DataFrame(results, columns=columns)

            # 결과 출력
            st.success("✅ 모든 데이터를 가져왔습니다!")
            st.dataframe(res_df)

            # 다운로드 버튼
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="결과 CSV 다운로드",
                data=csv,
                file_name=f"financial_results_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
            )
