import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from urllib.parse import quote
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

        # 3. 요약 섹션(base_results) 데이터 구성
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
            round(info.get("trailingEps"), 2) if info.get("trailingEps") else None
        ]

        # 4. 시계열 추이 데이터 매핑
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

# --- [함수] LLM 제공자별 API 키 확인 ---
def check_api_key(provider):
    """선택한 LLM 제공자의 API 키가 설정되어 있는지 확인"""
    key_map = {
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY"
    }
    
    required_key = key_map.get(provider)
    if not required_key:
        return False, "알 수 없는 제공자입니다."
    
    if required_key not in st.secrets:
        return False, f"{required_key}가 Streamlit Secrets에 설정되지 않았습니다."
    
    api_key = st.secrets[required_key]
    if not api_key or api_key == "":
        return False, f"{required_key} 값이 비어있습니다."
    
    return True, api_key

# --- [함수] LLM 기반 투자 등급 분석 ---
def analyze_stock_with_llm(ticker, financial_data, llm_provider="gemini"):
    """
    재무 데이터를 LLM에 전달하여 투자 등급(A~F) + 이유 반환
    """
    try:
        # API 키 확인
        is_valid, result = check_api_key(llm_provider)
        if not is_valid:
            return "N/A", f"⚠️ {result}"
        
        api_key = result
        
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
- Valuation: PER < 15, PBR < 2 (Undervalued) | PER 15-25, PBR 2-4 (Fair) | PER > 25, PBR > 4 (Overvalued)
- Profitability: ROE > 15%, OPM > 10% (Excellent) | ROE 10-15%, OPM 5-10% (Good) | ROE < 10% (Weak)
- Cash Flow: FCF_Stability > 80%, Positive FCF (Healthy) | 50-80% (Moderate) | < 50% (Risky)
- Financial Health: DTE < 100%, CR > 150% (Strong) | DTE 100-200%, CR 100-150% (Average) | DTE > 200% (Weak)
- Sustainability: Runway > 5 years or Infinite (Good) | 2-5 years (Moderate) | < 2 years (Risk)

Grade Assignment:
- A: 4+ Excellent criteria, 0 Weak
- B: 3+ Good criteria, max 1 Weak
- C: Mixed results, 2-3 Average
- D: 2+ Weak criteria
- F: 3+ Weak criteria or critical risks

Return ONLY in this JSON format:
{{"grade": "A/B/C/D/F", "reason": "Korean explanation"}}
"""
        
        # LLM 호출 (선택된 제공자)
        if llm_provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # 여러 모델명 시도 (fallback)
            model_names = [
                'gemini-1.5-flash-latest',
                'gemini-1.5-flash',
                'gemini-pro',
                'gemini-1.0-pro'
            ]
            
            last_error = None
            for model_name in model_names:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    result_text = response.text
                    break  # 성공하면 루프 종료
                except Exception as e:
                    last_error = str(e)
                    continue  # 다음 모델 시도
            else:
                # 모든 모델 실패
                raise Exception(f"모든 Gemini 모델 실패. 마지막 오류: {last_error}")
            
        elif llm_provider == "groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
            )
            result_text = chat_completion.choices[0].message.content
            
        elif llm_provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            result_text = message.content[0].text
        
        else:
            return "N/A", "지원하지 않는 LLM 제공자입니다."
        
        # JSON 파싱
        result = json.loads(result_text.strip().replace("```json", "").replace("```", ""))
        return result.get("grade", "N/A"), result.get("reason", "분석 실패")
    
    except json.JSONDecodeError as e:
        return "ERROR", f"JSON 파싱 실패: {str(e)[:50]}"
    except ImportError as e:
        return "ERROR", f"라이브러리 미설치: {str(e)[:50]}"
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            return "ERROR", "모델을 찾을 수 없습니다. API 키를 확인하세요"
        elif "401" in error_msg or "403" in error_msg:
            return "ERROR", "API 키가 유효하지 않습니다"
        elif "429" in error_msg:
            return "ERROR", "API 호출 한도 초과 (잠시 후 재시도)"
        elif "quota" in error_msg.lower():
            return "ERROR", "API 무료 할당량 초과"
        else:
            return "ERROR", f"{error_msg[:80]}"

# --- [UI] Streamlit 설정 ---
st.set_page_config(page_title="Stock Master Analyzer with AI", layout="wide")

# --- [헤더] ---
st.title("📊 AI 투자 등급 분석 시스템 (Multi-LLM)")
st.markdown("**yfinance** 재무 데이터 + **AI 자동 등급 분석** (Y4 → TTM)")

# --- [사이드바] ---
st.sidebar.header("📥 데이터 소스")
method = st.sidebar.radio("방식", ("텍스트 붙여넣기", "구글 스프레드시트", "CSV 파일 업로드"))

st.sidebar.markdown("---")
st.sidebar.header("🤖 AI 분석 설정")
enable_ai = st.sidebar.checkbox("AI 투자 등급 분석 활성화", value=True)

if enable_ai:
    # LLM 제공자 선택
    llm_options = {
        "gemini": "🟢 Google Gemini (무료, 추천)",
        "groq": "🟡 Groq Llama (무료, 초고속)",
        "anthropic": "🔵 Claude Sonnet (유료, 고품질)"
    }
    
    llm_provider = st.sidebar.selectbox(
        "LLM 모델 선택",
        list(llm_options.keys()),
        format_func=lambda x: llm_options[x]
    )
    
    # API 키 상태 확인
    is_valid, message = check_api_key(llm_provider)
    
    if is_valid:
        st.sidebar.success(f"✅ {llm_provider.upper()} API 키 확인됨")
    else:
        st.sidebar.error(f"❌ {message}")
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔑 API 키 설정 방법")
        st.sidebar.code(f"""
# Streamlit Cloud → Settings → Secrets에 추가:

{llm_provider.upper()}_API_KEY = "your-api-key-here"
""")
        if llm_provider == "gemini":
            st.sidebar.markdown("[Gemini API 키 발급하기](https://aistudio.google.com/app/apikey)")
        elif llm_provider == "groq":
            st.sidebar.markdown("[Groq API 키 발급하기](https://console.groq.com/keys)")
        elif llm_provider == "anthropic":
            st.sidebar.markdown("[Claude API 키 발급하기](https://console.anthropic.com/)")

# 티커 입력
tickers = []
if method == "텍스트 붙여넣기":
    raw = st.sidebar.text_area("티커 입력 (한 줄에 하나씩)", placeholder="AAPL\nMSFT\nGOOGL")
    if raw: 
        tickers = [t.strip().upper() for t in raw.split('\n') if t.strip()]
        
elif method == "구글 스프레드시트":
    try:
        if "GOOGLE_SHEET_ID" not in st.secrets or "GOOGLE_SHEET_NAME" not in st.secrets:
            st.sidebar.warning("⚠️ Google Sheets 연동을 위해 Secrets에 GOOGLE_SHEET_ID와 GOOGLE_SHEET_NAME을 설정하세요.")
        else:
            sid, sname = st.secrets["GOOGLE_SHEET_ID"], st.secrets["GOOGLE_SHEET_NAME"]
            url = f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={quote(sname)}"
            gs_df = pd.read_csv(url)
            t_col = st.sidebar.selectbox("티커 컬럼", gs_df.columns)
            tickers = gs_df[t_col].dropna().astype(str).tolist()
            st.sidebar.success(f"✅ {len(tickers)}개 티커 로드됨")
    except Exception as e: 
        st.sidebar.error(f"연결 실패: {e}")
        
elif method == "CSV 파일 업로드":
    up = st.sidebar.file_uploader("CSV 파일 선택", type=["csv"])
    if up:
        df = pd.read_csv(up)
        t_col = st.sidebar.selectbox("티커 컬럼", df.columns)
        tickers = df[t_col].dropna().astype(str).tolist()
        st.sidebar.success(f"✅ {len(tickers)}개 티커 로드됨")

# --- [메인] 분석 실행 ---
if tickers:
    total = len(tickers)
    st.info(f"📌 분석 대상: **{total}개** 종목")
    
    if st.button("🚀 전수 분석 시작", type="primary", use_container_width=True):
        # API 키 재확인
        if enable_ai:
            is_valid, message = check_api_key(llm_provider)
            if not is_valid:
                st.error(f"❌ {message}")
                st.stop()
        
        prog = st.progress(0)
        status = st.empty()
        results = []
        
        # 헤더 정의
        base_cols = [
            'ticker', 'AI_Grade', 'AI_Reason',
            'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Runway(Y)', 
            'TotalCash(M$)', 'FCF(M$)', 'FCF_Stability(%)', 'OCF(M$)', 
            'PBR', 'BPS', 'PER', 'EPS', 'Updated'
        ]
        
        metrics = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history_cols = [f"{m}_{y}" for m in metrics for y in ["Y4", "Y3", "Y2", "Y1", "TTM"]]
        final_cols = base_cols + history_cols

        for idx, symbol in enumerate(tickers):
            status.markdown(f"### ⏳ 분석 중: **{symbol}** ({idx+1} / {total})")
            
            # 재무 데이터 추출 (원래 코드 그대로 유지)
            data = get_extended_financials(symbol)
            
            # AI 등급 분석 (실패해도 데이터는 보존)
            if enable_ai:
                try:
                    grade, reason = analyze_stock_with_llm(symbol, data[:13], llm_provider)
                except Exception as e:
                    grade, reason = "ERROR", f"AI 분석 실패: {str(e)[:80]}"
            else:
                grade, reason = "-", "-"
            
            # row 생성
            row = [symbol, grade, reason] + data[:13] + [datetime.now().strftime('%H:%M:%S')] + data[13:]
            results.append(row)
            
            prog.progress((idx+1)/total)
            
            # API 호출 제한 고려
            if enable_ai:
                if llm_provider == "groq":
                    time.sleep(1)
                elif llm_provider == "gemini":
                    time.sleep(2)
                else:
                    time.sleep(0.5)
            else:
                time.sleep(0.3)

        status.success(f"✅ 분석 완료! ({total}개 종목)")
        res_df = pd.DataFrame(results, columns=final_cols).fillna("-")
        
        # AI 분석 오류만 체크
        if enable_ai:
            ai_errors = res_df[res_df['AI_Grade'] == 'ERROR'].shape[0]
            if ai_errors > 0:
                st.warning(f"⚠️ AI 분석 실패: {ai_errors}개 종목 (재무 데이터는 정상)")
        
        # 등급별 색상 표시
        def highlight_grade(val):
            color_map = {
                'A': 'background-color: #d4edda; color: #155724; font-weight: bold',
                'B': 'background-color: #d1ecf1; color: #0c5460',
                'C': 'background-color: #fff3cd; color: #856404',
                'D': 'background-color: #f8d7da; color: #721c24',
                'F': 'background-color: #f5c6cb; color: #721c24; font-weight: bold',
                'ERROR': 'background-color: #fff3cd; color: #856404'
            }
            return color_map.get(val, '')
        
        st.markdown("### 📋 분석 결과")
        st.dataframe(
            res_df.style.applymap(highlight_grade, subset=['AI_Grade']),
            use_container_width=True,
            height=600
        )
        
        # CSV 다운로드
        csv = res_df.to_csv(index=False).encode('utf-8')
        filename = f"stock_analysis_{llm_provider}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        st.download_button(
            "📥 결과 CSV 다운로드", 
            csv, 
            filename, 
            "text/csv",
            use_container_width=True
        )
        
        # 등급 분포 통계
        if enable_ai:
            valid_grades = res_df[~res_df['AI_Grade'].isin(['ERROR', '-', 'N/A'])]
            
            if len(valid_grades) > 0:
                st.markdown("---")
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.markdown("### 📈 등급 분포 차트")
                    grade_counts = valid_grades['AI_Grade'].value_counts().reindex(['A', 'B', 'C', 'D', 'F'], fill_value=0)
                    st.bar_chart(grade_counts)
                
                with col2:
                    st.markdown("### 📊 등급별 통계")
                    valid_total = len(valid_grades)
                    for grade in ['A', 'B', 'C', 'D', 'F']:
                        count = grade_counts.get(grade, 0)
                        pct = (count / valid_total) * 100 if valid_total > 0 else 0
                        emoji = {'A': '🟢', 'B': '🔵', 'C': '🟡', 'D': '🟠', 'F': '🔴'}
                        st.metric(f"{emoji[grade]} {grade} 등급", f"{count}개", f"{pct:.1f}%")
                    
                    # 분석 성공률
                    st.markdown("---")
                    success_rate = (valid_total / total) * 100 if total > 0 else 0
                    st.metric("✅ AI 분석 성공률", f"{success_rate:.1f}%", f"{valid_total}/{total}")
            else:
                st.warning("⚠️ AI 분석에 성공한 종목이 없습니다. API 키와 설정을 확인해주세요.")

else:
    st.info("👈 사이드바에서 티커를 입력하세요")
    
    # 예시 표시
    with st.expander("💡 사용 예시 보기"):
        st.markdown("""
        ### 티커 입력 예시
        ```
        AAPL
        MSFT
        GOOGL
        TSLA
        NVDA
        ```
        
        ### 출력 결과 예시
        | ticker | AI_Grade | AI_Reason |
        |--------|----------|-----------|
        | AAPL   | A        | ROE 30% 이상, 안정적 현금흐름, PER 적정 수준 |
        | MSFT   | B        | 강한 재무구조, FCF 안정적, PBR 다소 높음 |
        | TSLA   | C        | 성장성 우수하나 밸류에이션 부담 |
        """)

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Powered by yfinance + AI | LLM: {provider} | ⚠️ 투자 참고용이며, 실제 투자 결정은 본인 책임입니다</p>
</div>
""".format(provider=llm_provider.upper() if enable_ai else "None"), unsafe_allow_html=True)
