import yfinance as yf
import pandas as pd
from datetime import datetime
import time
from pathlib import Path


def get_financial_ratios(ticker_symbol):
    """
    Yahoo Finance 제공 지표(D/E, Current Ratio, ROE) + freeCashflow 기반 Runway 계산
    + OperatingCashflow, NetIncome, PBR, BPS 추가
    Runway(Years) = totalCash / abs(freeCashflow)
    totalCash, freeCashflow, operatingCashflow, netIncome은 M달러(Million USD) 단위로 변환
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        # ✅ Yahoo 제공 기본 지표
        dte = info.get("debtToEquity")           # %
        cr = info.get("currentRatio")            # 배수 (1.25 → 125%)
        roe = info.get("returnOnEquity")         # 비율 (0.15 → 15%)

        # ✅ Runway 계산용 항목
        total_cash = info.get("totalCash")             # USD
        free_cf = info.get("freeCashflow")             # USD (연간)
        operating_cf = info.get("operatingCashflow")   # USD (연간)
        net_income = info.get("netIncomeToCommon")     # USD (연간)

        # ✅ PBR, BPS 항목
        pbr = info.get("priceToBook")                  # 배수
        bps = info.get("bookValue")                    # USD per share

        # 🔹 단위 변환
        if cr is not None:
            cr = round(cr * 100, 2)
        if roe is not None:
            roe = round(roe * 100, 2)

        total_cash_m = None
        free_cf_m = None
        operating_cf_m = None
        net_income_m = None
        if total_cash is not None:
            total_cash_m = round(total_cash / 1_000_000, 2)  # M달러로 변환
        if free_cf is not None:
            free_cf_m = round(free_cf / 1_000_000, 2)        # M달러로 변환
        if operating_cf is not None:
            operating_cf_m = round(operating_cf / 1_000_000, 2)  # M달러로 변환
        if net_income is not None:
            net_income_m = round(net_income / 1_000_000, 2)  # M달러로 변환

        # 🔹 PBR, BPS 반올림
        if pbr is not None:
            pbr = round(pbr, 2)
        if bps is not None:
            bps = round(bps, 2)

        # 🔹 Runway 계산 (연 단위)
        runway_years = None
        if total_cash and free_cf:
            if free_cf < 0:
                runway_years = round(total_cash / abs(free_cf), 2)
            elif free_cf >= 0:
                runway_years = float('inf')  # 흑자 기업은 Runway 무제한

        return dte, cr, roe, runway_years, total_cash_m, free_cf_m, operating_cf_m, net_income_m, pbr, bps

    except Exception as e:
        print(f"⚠️ Error fetching info for {ticker_symbol}: {e}")
        return None, None, None, None, None, None, None, None, None, None


def fetch_financial_data(input_file, output_file=None, ticker_column='ticker'):
    """CSV에서 티커를 읽고 Yahoo Finance 제공 지표 + Runway 계산 후 저장 (M달러 단위 포함, OperatingCashflow, NetIncome 추가)"""
    print(f"📂 Reading input file: {input_file}")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    if ticker_column not in df.columns:
        print(f"❌ Column '{ticker_column}' not found. Available: {df.columns.tolist()}")
        return

    # ✅ 결과파일명 자동 설정
    if output_file is None:
        path = Path(input_file)
        output_file = path.parent / f"{path.stem}_result.csv"

    # ✅ 결과 컬럼 초기화 (FreeCashflow 다음에 OperatingCashflow, NetIncome, PBR, BPS 추가)
    df['debtToEquity(%)'] = None
    df['currentRatio(%)'] = None
    df['ROE(%)'] = None
    df['Runway(Years)'] = None
    df['TotalCash(M$)'] = None
    df['FreeCashflow(M$)'] = None
    df['OperatingCashflow(M$)'] = None
    df['NetIncome(M$)'] = None
    df['PBR'] = None
    df['BPS($)'] = None
    df['lastUpdated'] = None

    print(f"💾 Output file: {output_file}")
    print(f"📊 Found {len(df)} tickers")
    print("-" * 60)

    success = 0

    for idx, row in df.iterrows():
        ticker_symbol = str(row[ticker_column]).strip()
        if not ticker_symbol or ticker_symbol.lower() == 'nan':
            continue

        print(f"[{idx + 1}/{len(df)}] {ticker_symbol} ...")

        dte, cr, roe, runway, total_cash, free_cf, operating_cf, net_income, pbr, bps = get_financial_ratios(ticker_symbol)

        df.at[idx, 'debtToEquity(%)'] = dte
        df.at[idx, 'currentRatio(%)'] = cr
        df.at[idx, 'ROE(%)'] = roe
        df.at[idx, 'Runway(Years)'] = runway
        df.at[idx, 'TotalCash(M$)'] = total_cash
        df.at[idx, 'FreeCashflow(M$)'] = free_cf
        df.at[idx, 'OperatingCashflow(M$)'] = operating_cf
        df.at[idx, 'NetIncome(M$)'] = net_income
        df.at[idx, 'PBR'] = pbr
        df.at[idx, 'BPS($)'] = bps
        df.at[idx, 'lastUpdated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        success += 1
        if (idx + 1) % 10 == 0:
            df.to_csv(output_file, index=False)
            print(f"💾 Progress saved ({idx + 1}/{len(df)})")

        time.sleep(0.5)  # 과도한 요청 방지

    # ✅ 컬럼 순서 재정렬 (FreeCashflow 다음에 OperatingCashflow, NetIncome, PBR, BPS)
    column_order = [
        ticker_column,
        'debtToEquity(%)',
        'currentRatio(%)',
        'ROE(%)',
        'Runway(Years)',
        'TotalCash(M$)',
        'FreeCashflow(M$)',
        'OperatingCashflow(M$)',
        'NetIncome(M$)',
        'PBR',
        'BPS($)',
        'lastUpdated'
    ]
    df = df[column_order]

    # ✅ 최종 저장
    df.to_csv(output_file, index=False)
    print("-" * 60)
    print(f"✅ Complete! Results saved to {output_file}")
    print(f"✅ Successful: {success}/{len(df)} tickers")

    print("\n=== Sample Results ===")
    print(df.head(10))


if __name__ == "__main__":
    fetch_financial_data("tickers.csv")
