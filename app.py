import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
from urllib.parse import quote

# --- 재무 지표 가져오기 함수 ---
def get_financial_ratios(ticker_symbol):
    try:
        symbol = ticker_symbol.upper().strip()
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # 데이터 추출
        dte = info.get("debtToEquity")
        cr = info.get("currentRatio")
        op_margin = info.get("operatingMargins")  # ✅ 영업이익률 추가
        roe = info.get("returnOnEquity")
        total_cash = info.get("totalCash")
        free_cf = info.get("freeCashflow")
        operating_cf = info.get("operatingCashflow")
        net_income = info.get("netIncomeToCommon")
        pbr = info.get("priceToBook")
        bps = info.get("bookValue")

        # 단위 변환 (%)
        cr = round(cr * 100, 2) if cr else None
        op_margin = round(op_margin * 100, 2) if op_margin else None  # ✅ % 변환
        roe = round(roe * 100, 2) if roe else None
        
        def to_million(val):
            return round(val / 1_000_000, 2) if val else None

        # Runway 계산
        runway_years = None
        if total_cash and free_cf:
            if free_cf < 0:
                runway_years = round(total_cash / abs(free_cf), 2)
            else:
                runway_years = float('inf')

        # 리스트 순서에 op_margin 삽입 (유동비율 다음)
        return [
            dte, cr, op_margin, roe, runway_years, 
            to_million(total_cash), to_million(free_cf), 
            to_million(operating_cf), to_million(net_income),
            round(pbr, 2) if pbr else None, 
            round(bps, 2) if bps else None
        ]
    except Exception:
        return [None] * 11  # 컬럼이 하나 늘었으므로 11개 반환

# --- (중략: 사이드바 및 입력 로직은 동일) ---

# --- 메인 분석 실행 섹션 내 컬럼 정의 수정 ---
if tickers:
    if st.button("분석 시작 (Start Analysis)"):
        # ... (진행률 로직 동일)
        
        # 컬럼 이름 리스트에 'OperatingMargin(%)' 추가
        columns = [
            'ticker', 'debtToEquity(%)', 'currentRatio(%)', 'OperatingMargin(%)', 'ROE(%)', 
            'Runway(Years)', 'TotalCash(M$)', 'FreeCashflow(M$)', 
            'OperatingCashflow(M$)', 'NetIncome(M$)', 'PBR', 'BPS($)', 'lastUpdated'
        ]
        res_df = pd.DataFrame(results_list, columns=columns)
        
        st.success("✅ 분석 완료!")
        st.dataframe(res_df, use_container_width=True)
        # ... (다운로드 로직 동일)
