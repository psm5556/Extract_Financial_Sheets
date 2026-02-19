import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from urllib.parse import quote
import json
import requests
import random

# ============================================================
# [핵심 수정] 안정적인 yfinance 데이터 추출을 위한 유틸리티
# ============================================================

def create_yf_session():
    """
    Yahoo Finance 봇 차단 우회를 위한 커스텀 세션 생성.
    User-Agent를 실제 브라우저처럼 설정하고 헤더를 추가합니다.
    """
    session = requests.Session()
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    ]
    session.headers.update({
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })
    return session


def safe_get_ticker(symbol, max_retries=3):
    """
    재시도 로직이 포함된 안전한 Ticker 객체 생성.
    실패 시 최대 max_retries회 재시도하며 지수 백오프를 사용합니다.
    """
    for attempt in range(max_retries):
        try:
            session = create_yf_session()
            ticker = yf.Ticker(symbol, session=session)
            
            # 연결 유효성 빠른 검증 (fast_info는 가벼운 요청)
            _ = ticker.fast_info
            return ticker
        except Exception as e:
            wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)  # 지수 백오프
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                raise e
    return None


def safe_fetch_with_retry(fetch_func, max_retries=3, default=None):
    """
    임의의 데이터 페치 함수를 재시도 래퍼로 감쌉니다.
    rate limit(429) 감지 시 더 긴 대기 시간을 적용합니다.
    """
    for attempt in range(max_retries):
        try:
            result = fetch_func()
            # 결과가 None이거나 빈 DataFrame이면 재시도
            if result is None:
                raise ValueError("None 반환")
            if isinstance(result, pd.DataFrame) and result.empty:
                raise ValueError("빈 DataFrame 반환")
            return result
        except Exception as e:
            err_str = str(e).lower()
            # Rate limit 감지
            if "429" in err_str or "too many requests" in err_str:
                wait_time = 30 + random.uniform(5, 15)  # 30~45초 대기
                time.sleep(wait_time)
            else:
                wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
    return default


def safe_info_get(info, key, multiplier=1, divisor=1, digits=2):
    """
    info 딕셔너리에서 안전하게 값을 추출하고 계산합니다.
    None/오류 시 None을 반환합니다.
    """
    try:
        val = info.get(key)
        if val is None or val != val:  # NaN 체크
            return None
        result = val * multiplier / divisor
        return round(result, digits)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)  # 1시간 캐싱 (중복 요청 방지)
def get_extended_financials(ticker_symbol):
    """
    [개선] 재시도, 캐싱, 폴백 로직이 적용된 안정적인 재무 데이터 추출 함수.
    
    개선 사항:
    - safe_get_ticker: 커스텀 세션 + 재시도로 Ticker 생성
    - safe_fetch_with_retry: 각 API 호출(info, financials 등)에 재시도 적용
    - fast_info 폴백: info 실패 시 fast_info에서 핵심 지표 대체
    - 개별 예외 처리: 하나가 실패해도 나머지 데이터는 정상 수집
    """
    try:
        symbol = ticker_symbol.upper().strip()

        # ── 1. Ticker 생성 (재시도 포함) ──────────────────────────────
        ticker = safe_get_ticker(symbol, max_retries=3)
        if ticker is None:
            raise ValueError(f"{symbol}: Ticker 생성 실패")

        # ── 2. 각 데이터 소스를 개별 재시도로 가져오기 ─────────────────
        info = safe_fetch_with_retry(
            lambda: ticker.info, max_retries=3, default={}
        ) or {}

        fin = safe_fetch_with_retry(
            lambda: ticker.financials, max_retries=3, default=pd.DataFrame()
        ) or pd.DataFrame()

        bs = safe_fetch_with_retry(
            lambda: ticker.balance_sheet, max_retries=3, default=pd.DataFrame()
        ) or pd.DataFrame()

        cf = safe_fetch_with_retry(
            lambda: ticker.cashflow, max_retries=3, default=pd.DataFrame()
        ) or pd.DataFrame()

        # ── 3. info가 비어있으면 fast_info로 폴백 ──────────────────────
        if not info:
            try:
                fi = ticker.fast_info
                info = {
                    "debtToEquity":     getattr(fi, "debt_to_equity", None),
                    "currentRatio":     getattr(fi, "current_ratio", None),
                    "operatingMargins": getattr(fi, "operating_margins", None),
                    "returnOnEquity":   getattr(fi, "return_on_equity", None),
                    "operatingCashflow":getattr(fi, "operating_cashflow", None),
                    "freeCashflow":     getattr(fi, "free_cashflow", None),
                    "netIncomeToCommon":getattr(fi, "net_income_to_common", None),
                    "totalCash":        getattr(fi, "total_cash", None),
                    "priceToBook":      getattr(fi, "price_to_book", None),
                    "bookValue":        getattr(fi, "book_value", None),
                    "trailingPE":       getattr(fi, "pe_ratio", None),
                    "trailingEps":      getattr(fi, "trailing_eps", None),
                }
            except Exception:
                pass  # fast_info도 실패하면 None들로 진행

        # ── 4. 내부 헬퍼 ────────────────────────────────────────────────
        def get_val(df, label, idx):
            try:
                if df.empty:
                    return None
                val = df.loc[label].iloc[idx]
                return None if (pd.isna(val) or val is None) else val
            except Exception:
                return None

        # ── 5. TTM (최근 12개월) 기본 데이터 추출 ──────────────────────
        ttm_dte  = safe_info_get(info, "debtToEquity")
        ttm_cr   = safe_info_get(info, "currentRatio",     multiplier=100)
        ttm_opm  = safe_info_get(info, "operatingMargins", multiplier=100)
        ttm_roe  = safe_info_get(info, "returnOnEquity",   multiplier=100)
        ttm_ocf  = safe_info_get(info, "operatingCashflow", digits=0)
        ttm_fcf  = safe_info_get(info, "freeCashflow",      digits=0)
        ttm_net_inc = safe_info_get(info, "netIncomeToCommon", digits=0)
        total_cash  = safe_info_get(info, "totalCash",         digits=0)

        # Runway 계산
        if total_cash and ttm_fcf:
            runway = round(total_cash / abs(ttm_fcf), 2) if ttm_fcf < 0 else "Infinite (Profit)"
        else:
            runway = None

        # ── 6. 항목별 추이 데이터 수집 (Y4 → TTM) ─────────────────────
        metrics_order = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history = {m: [None] * 4 for m in metrics_order}
        num_years = min(len(fin.columns), 4) if not fin.empty else 0

        for i in range(num_years):
            idx = 3 - i
            try:
                net_inc  = get_val(fin, 'Net Income', i)
                equity   = get_val(bs,  'Total Equity Gross Minority Interest', i)
                ocf_val  = get_val(cf,  'Operating Cash Flow', i)
                cap_ex   = get_val(cf,  'Capital Expenditure', i)
                fcf_val  = (ocf_val + cap_ex) if (ocf_val is not None and cap_ex is not None) else None

                total_liab = get_val(bs, 'Total Liabilities Net Minority Interest', i)
                curr_assets = get_val(bs, 'Current Assets', i)
                curr_liab   = get_val(bs, 'Current Liabilities', i)
                op_inc      = get_val(fin, 'Operating Income', i)
                total_rev   = get_val(fin, 'Total Revenue', i)
                basic_eps   = get_val(fin, 'Basic EPS', i)

                history["DTE"][idx] = round(total_liab / equity * 100, 2)  if (total_liab and equity)               else None
                history["CR"][idx]  = round(curr_assets / curr_liab * 100, 2) if (curr_assets and curr_liab)        else None
                history["OPM"][idx] = round(op_inc / total_rev * 100, 2)   if (op_inc and total_rev)                else None
                history["ROE"][idx] = round(net_inc / equity * 100, 2)     if (net_inc and equity)                   else None
                history["OCF"][idx] = round(ocf_val / 1_000_000, 2)        if ocf_val is not None                   else None
                history["EPS"][idx] = round(basic_eps, 2)                  if basic_eps is not None                  else None
                history["CFQ"][idx] = round(ocf_val / net_inc, 2)          if (ocf_val and net_inc and net_inc != 0) else None
                history["FCF"][idx] = round(fcf_val / 1_000_000, 2)        if fcf_val is not None                   else None
            except Exception:
                pass  # 연도별 계산 실패 시 None 유지

        # ── 7. Stability 계산 ────────────────────────────────────────
        ttm_fcf_m = round(ttm_fcf / 1_000_000, 2) if ttm_fcf else None
        fcf_series = history["FCF"] + [ttm_fcf_m]
        plus_count = sum(1 for v in fcf_series if v is not None and v > 0)
        stability  = (plus_count / 5) * 100 if any(v is not None for v in fcf_series) else None

        # ── 8. 요약 섹션 데이터 구성 ─────────────────────────────────
        ttm_ocf_m = round(ttm_ocf / 1_000_000, 2) if ttm_ocf else None
        cash_m    = round(total_cash / 1_000_000, 2) if total_cash else None

        base_results = [
            ttm_dte,                                     # 0  DTE
            ttm_cr,                                      # 1  CR
            ttm_opm,                                     # 2  OPM
            ttm_roe,                                     # 3  ROE
            runway,                                      # 4  Runway
            cash_m,                                      # 5  TotalCash
            ttm_fcf_m,                                   # 6  FCF
            round(stability, 2) if stability else None,  # 7  FCF_Stability
            ttm_ocf_m,                                   # 8  OCF
            safe_info_get(info, "priceToBook"),          # 9  PBR
            safe_info_get(info, "bookValue"),            # 10 BPS
            safe_info_get(info, "trailingPE"),           # 11 PER
            safe_info_get(info, "trailingEps"),          # 12 EPS
        ]

        # ── 9. 시계열 추이 데이터 매핑 ───────────────────────────────
        ttm_vals_map = {
            "DTE": base_results[0],
            "CR":  base_results[1],
            "OPM": base_results[2],
            "ROE": base_results[3],
            "OCF": base_results[8],
            "EPS": base_results[12],
            "CFQ": round(ttm_ocf / ttm_net_inc, 2) if (ttm_ocf and ttm_net_inc and ttm_net_inc != 0) else None,
            "FCF": ttm_fcf_m,
        }

        flattened_history = []
        for key in metrics_order:
            combined = history[key] + [ttm_vals_map[key]]
            flattened_history.extend(combined)

        return base_results + flattened_history

    except Exception as e:
        # 전체 실패 시 상세 오류 로그 (디버깅용)
        print(f"[ERROR] {ticker_symbol}: {str(e)}")
        return [None] * (13 + 40)


# ============================================================
# AI 투자 등급 분석 (원본 그대로)
# ============================================================

def analyze_with_ai(ticker, financial_data, llm_provider):
    """AI를 사용한 투자 등급 분석"""
    try:
        metrics = {
            "Ticker": ticker,
            "ROE(%)":           financial_data[3],
            "OPM(%)":           financial_data[2],
            "EPS":              financial_data[12],
            "PER":              financial_data[11],
            "PBR":              financial_data[9],
            "BPS":              financial_data[10],
            "OCF(M$)":          financial_data[8],
            "FCF(M$)":          financial_data[6],
            "FCF_Stability(%)": financial_data[7],
            "DTE(%)":           financial_data[0],
            "CR(%)":            financial_data[1],
            "Cash(M$)":         financial_data[5],
            "Runway(Years)":    financial_data[4],
        }

        prompt = f"""You are a professional equity analyst conducting fundamental analysis on {ticker}.

Financial Data:
{json.dumps(metrics, indent=2)}

CRITICAL ANALYSIS FRAMEWORK:

Step 1: CLASSIFY STOCK TYPE
- Value Stock: PER < 15, PBR < 2, ROE > 15%, Stable business
- Growth Stock: EPS growth trend, High FCF growth, Expanding margins

Step 2: APPLY APPROPRIATE CRITERIA

VALUE STOCK CRITERIA (Warren Buffett Style):
✓ ROE consistently > 15% (경제적 해자)
✓ Cash Flow Quality Ratio (OCF/Net Income) > 100%
✓ Operating Margin > 10%
✓ Debt/Equity < 100%
✓ PER < Industry Average
✓ FCF/Revenue > 10%
✓ Stable or growing dividends

GROWTH STOCK CRITERIA (Peter Lynch Style):
✓ EPS growth trajectory (check Y4→Y3→Y2→Y1→TTM)
✓ FCF consistently positive and growing
✓ Operating Margin expanding
✓ High ROE (> 20%) with growth
✓ PER acceptable if justified by growth (PEG ratio concept)
✓ Low debt enabling reinvestment

Step 3: ASSIGN GRADE
- A (Excellent): Meets 5+ key criteria, no critical weaknesses
- B (Good): Meets 3-4 criteria, minor concerns
- C (Average): Mixed signals, 2-3 criteria met
- D (Below Average): Fails multiple criteria
- F (Poor): Critical red flags (negative FCF, ROE<10%, debt crisis)

Step 4: WRITE KOREAN EXPLANATION (40-80 words)

MUST INCLUDE:
1. Stock type classification (가치주 or 성장주)
2. 2-3 strongest points with specific numbers
3. 1-2 concerns or weaknesses
4. Overall investment thesis

EXAMPLE EXCELLENT RESPONSE (Value Stock):
{{"grade": "A", "reason": "전형적인 가치주로 ROE 18%를 5년간 유지하며 경제적 해자를 보유하고 있습니다. 현금흐름 질 비율 120%로 순이익 대비 실제 현금 유입이 우수하며, 영업이익률 12%는 업계 최상위권입니다. PER 14배는 저평가 구간이나, 부채비율 180%는 다소 부담스러운 수준입니다."}}

EXAMPLE EXCELLENT RESPONSE (Growth Stock):
{{"grade": "B", "reason": "성장주로서 EPS가 Y4 대비 45% 증가하며 강한 상승세를 보이고 있습니다. 영업이익률이 15%→18%→22%로 확대되며 규모의 경제 효과가 나타나고 있으나, PER 35배는 향후 성장률을 감안해도 다소 부담스러운 수준입니다. FCF 안정성 80%는 양호한 편입니다."}}

Now analyze {ticker}. Return ONLY valid JSON:
{{"grade": "A/B/C/D/F", "reason": "가치주/성장주 분류, 구체적 수치 포함, 40-80 한국어 단어"}}"""

        # === GEMINI ===
        if llm_provider == "gemini":
            if "GEMINI_API_KEY" not in st.secrets:
                return "N/A", "Gemini API 키가 Secrets에 없습니다"
            try:
                import google.generativeai as genai
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                models_to_try = [
                    'gemini-2.5-flash', 'gemini-3-flash',
                    'gemini-2.5-flash-lite', 'gemini-1.5-flash', 'gemini-1.5-pro'
                ]
                last_error = None
                for model_name in models_to_try:
                    try:
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content(prompt)
                        text = response.text.strip().replace("```json", "").replace("```", "").strip()
                        if "{" in text and "}" in text:
                            json_text = text[text.find("{"):text.rfind("}")+1]
                            result = json.loads(json_text)
                        else:
                            result = json.loads(text)
                        grade = result.get("grade", "C").upper()
                        reason = result.get("reason", "")
                        if len(reason) < 20: continue
                        if reason.count(",") > 5 and len(reason.split()) < 10: continue
                        if grade not in ['A', 'B', 'C', 'D', 'F']: grade = 'C'
                        return grade, reason
                    except Exception as e:
                        last_error = str(e); continue
                return "ERROR", f"Gemini 오류: {last_error[:60]}"
            except ImportError:
                return "ERROR", "google-generativeai 패키지 미설치"
            except Exception as e:
                return "ERROR", f"Gemini 설정 오류: {str(e)[:60]}"

        # === GROQ ===
        elif llm_provider == "groq":
            if "GROQ_API_KEY" not in st.secrets:
                return "N/A", "Groq API 키가 Secrets에 없습니다"
            try:
                from groq import Groq
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    max_tokens=1000,
                )
                text = chat_completion.choices[0].message.content.strip()
                text = text.replace("```json", "").replace("```", "").strip()
                if "{" in text and "}" in text:
                    json_text = text[text.find("{"):text.rfind("}")+1]
                    result = json.loads(json_text)
                else:
                    result = json.loads(text)
                grade = result.get("grade", "C").upper()
                reason = result.get("reason", "")
                if len(reason) < 20: return "C", "AI 응답이 너무 짧습니다"
                if reason.count(",") > 5 and len(reason.split()) < 10: return "C", "약어만 나열되었습니다"
                if grade not in ['A', 'B', 'C', 'D', 'F']: grade = 'C'
                return grade, reason
            except ImportError:
                return "ERROR", "groq 패키지 미설치"
            except Exception as e:
                return "ERROR", f"Groq 오류: {str(e)[:60]}"

        # === CLAUDE ===
        elif llm_provider == "claude":
            if "ANTHROPIC_API_KEY" not in st.secrets:
                return "N/A", "Claude API 키가 Secrets에 없습니다"
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = message.content[0].text.strip().replace("```json", "").replace("```", "").strip()
                if "{" in text and "}" in text:
                    json_text = text[text.find("{"):text.rfind("}")+1]
                    result = json.loads(json_text)
                else:
                    result = json.loads(text)
                grade = result.get("grade", "C").upper()
                reason = result.get("reason", "")
                if len(reason) < 20: return "C", "AI 응답이 너무 짧습니다"
                if reason.count(",") > 5 and len(reason.split()) < 10: return "C", "약어만 나열되었습니다"
                if grade not in ['A', 'B', 'C', 'D', 'F']: grade = 'C'
                return grade, reason
            except ImportError:
                return "ERROR", "anthropic 패키지 미설치"
            except Exception as e:
                return "ERROR", f"Claude 오류: {str(e)[:60]}"

        else:
            return "N/A", f"알 수 없는 LLM: {llm_provider}"

    except Exception as e:
        return "ERROR", f"예상치 못한 오류: {str(e)[:60]}"


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Stock Master Analyzer", layout="wide")
st.title("📊 주식 재무 시계열 분석 마스터 (Y4 → TTM) + AI")

# ── 사이드바 ────────────────────────────────────────────────
st.sidebar.header("📥 데이터 소스")
method = st.sidebar.radio("방식", ("텍스트 붙여넣기", "구글 스프레드시트", "CSV 파일 업로드"))

st.sidebar.markdown("---")
st.sidebar.header("🤖 AI 분석 옵션")
enable_ai = st.sidebar.checkbox("AI 투자 등급 분석", value=False)
if enable_ai:
    llm_provider = st.sidebar.selectbox(
        "LLM 선택",
        ["groq", "gemini", "claude"],
        format_func=lambda x: {
            "gemini": "🟢 Google Gemini (무료)",
            "groq":   "🟡 Groq Llama (무료, 빠름, 추천!)",
            "claude": "🔵 Claude Sonnet (유료)"
        }[x]
    )
    if llm_provider == "gemini":
        st.sidebar.warning("⚠️ Gemini는 가끔 404 오류가 발생합니다. Groq 추천!")
    st.sidebar.info("💡 Streamlit Secrets에 API 키 설정 필요")

# [추가] 재시도 설정 옵션
st.sidebar.markdown("---")
st.sidebar.header("⚙️ 데이터 수집 설정")
delay_between = st.sidebar.slider(
    "티커 간 딜레이 (초)", min_value=1, max_value=10, value=2,
    help="딜레이가 길수록 Rate Limit 오류가 줄어듭니다"
)
show_cache_info = st.sidebar.checkbox("캐시 상태 표시", value=False)
if show_cache_info:
    st.sidebar.info("✅ 같은 티커는 1시간 동안 캐시되어 재요청하지 않습니다")

tickers = []
if method == "텍스트 붙여넣기":
    raw = st.sidebar.text_area("티커 입력 (한 줄에 하나씩)")
    if raw:
        tickers = [t.strip().upper() for t in raw.split('\n') if t.strip()]
elif method == "구글 스프레드시트":
    try:
        sid   = st.secrets["GOOGLE_SHEET_ID"]
        sname = st.secrets["GOOGLE_SHEET_NAME"]
        url   = f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={quote(sname)}"
        gs_df = pd.read_csv(url)
        t_col = st.sidebar.selectbox("티커 컬럼", gs_df.columns)
        tickers = gs_df[t_col].dropna().astype(str).tolist()
    except Exception as e:
        st.sidebar.error(f"연결 실패: {e}")
elif method == "CSV 파일 업로드":
    up = st.sidebar.file_uploader("CSV", type=["csv"])
    if up:
        df    = pd.read_csv(up)
        t_col = st.sidebar.selectbox("티커 컬럼", df.columns)
        tickers = df[t_col].dropna().astype(str).tolist()

# ── 메인 분석 실행 ────────────────────────────────────────
if tickers:
    total = len(tickers)
    if st.button("🚀 전수 분석 시작"):
        prog    = st.progress(0)
        status  = st.empty()
        results = []
        failed  = []  # 실패 티커 추적

        # 헤더 정의
        if enable_ai:
            base_cols = [
                'ticker', 'AI_Grade', 'AI_Reason',
                'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Runway(Y)',
                'TotalCash(M$)', 'FCF(M$)', 'FCF_Stability(%)', 'OCF(M$)',
                'PBR', 'BPS', 'PER', 'EPS', 'Updated'
            ]
        else:
            base_cols = [
                'ticker',
                'DTE(%)', 'CR(%)', 'OPM(%)', 'ROE(%)', 'Runway(Y)',
                'TotalCash(M$)', 'FCF(M$)', 'FCF_Stability(%)', 'OCF(M$)',
                'PBR', 'BPS', 'PER', 'EPS', 'Updated'
            ]

        metrics      = ["DTE", "CR", "OPM", "ROE", "OCF", "EPS", "CFQ", "FCF"]
        history_cols = [f"{m}_{y}" for m in metrics for y in ["Y4", "Y3", "Y2", "Y1", "TTM"]]
        final_cols   = base_cols + history_cols

        for idx, symbol in enumerate(tickers):
            status.markdown(f"### ⏳ 분석 중: **{symbol}** ({idx+1} / {total})")
            
            try:
                data = get_extended_financials(symbol)
            except Exception:
                data = [None] * (13 + 40)

            # 데이터 품질 체크 (핵심 지표가 모두 None이면 실패로 기록)
            core_data = [d for d in data[:5] if d is not None]
            if len(core_data) == 0:
                failed.append(symbol)
                status.warning(f"⚠️ {symbol}: 데이터 추출 실패 (건너뜀)")

            if enable_ai:
                ai_grade, ai_reason = analyze_with_ai(symbol, data[:13], llm_provider)
                row = [symbol, ai_grade, ai_reason] + data[:13] + [datetime.now().strftime('%H:%M:%S')] + data[13:]
            else:
                row = [symbol] + data[:13] + [datetime.now().strftime('%H:%M:%S')] + data[13:]

            results.append(row)
            prog.progress((idx + 1) / total)
            time.sleep(delay_between if not enable_ai else max(delay_between, 2))

        status.success(f"✅ 분석 완료! (성공: {total - len(failed)}개 / 실패: {len(failed)}개)")

        # 실패 티커 표시
        if failed:
            st.warning(f"⚠️ 데이터 추출 실패 티커: {', '.join(failed)}")
            st.info("💡 실패 티커는 잠시 후 개별로 다시 시도하거나, 티커 심볼이 맞는지 확인하세요.")

        res_df = pd.DataFrame(results, columns=final_cols).fillna("-")
        st.dataframe(res_df, use_container_width=True)

        csv_filename = f"financial_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        st.download_button(
            "📥 결과 CSV 다운로드",
            res_df.to_csv(index=False).encode('utf-8'),
            csv_filename,
            "text/csv"
        )
