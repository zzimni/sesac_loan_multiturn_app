from dotenv import load_dotenv 
import streamlit as st 
from config import FINETUNED_MODEL, SYSTEM_PROMPT
from rag import search_context, build_context_string 
from langchain_openai import ChatOpenAI 
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import re 

load_dotenv()

# ------------------------ 페이지 설정 ------------------------
st.set_page_config(
    page_title="🏡 전세자금대출 AI 상담사",
    page_icon="🏡",
    layout="centered"
)
st.title("🏡 전세자금대출 AI 상담사")
st.caption(
    "국민은행 · 신한은행 · 하나은행 상품설명서 기반 |"
    f"파인튜닝 모델 `{FINETUNED_MODEL}`"
)

# ------------------------ LangChain ChatOpenAI 모델 정의 ------------------------
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model=FINETUNED_MODEL,
        temperature=0,
        streaming=True
    )

model = get_llm()

# ------------------------ 세션 상태 초기화 ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []                  # LangChain 메시지 객체 리스트 (모델 전달용)
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []          # 화면 표시용 딕셔너리 리스트

# ------------------------ 사이드바: 사용자 프로필 입력 ------------------------
with st.sidebar:
    st.header("📋 내 금융 프로필")
    st.caption("입력 시 더 정확한 진단이 가능합니다.")

    target_bank = st.selectbox(
        "희망 은행",
        ["선택 안 함", "국민은행", "신한은행", "하나은행", "비교해서 추천"]
    )

    loan_purpose = st.selectbox(
        "대출 목적",
        ["선택", "전세자금", "전세 연장", "보증기관 선택", "금리 유형 선택", "기타"]
    )

    annual_income = st.number_input(
        "연소득 (만원)", min_value=0, max_value=100_000, value=4_000, step=100
    )

    credit_score = st.number_input(
        "신용점수 (점)", min_value=0, max_value=100, value=0, step=10
    )

    existing_loan = st.number_input(
        "희망 전세금 (만원)", min_value=0, max_value=200_000, value=0, step=10
    )

    target_amount = st.number_input(
        "희망 대출액 (만원)", min_value=0, max_value=500_000, value=20_000, step=500
    )

    if st.button("📊 프로필 기반 진단 시작", use_container_width=True):
        if annual_income > 0:
            # 프로필 문자열 생성 (사용자 입력값(input_data) 재정의)
            profile_lines = [
                f"희망 은행: {target_bank if target_bank != '선택 안 함' else '미정'}",
                f"대출 목적: {loan_purpose if loan_purpose != '선택' else '미정'}",
                f"연소득: {annual_income:,}만원",
                f"신용점수: {credit_score}점{'(미입력)' if credit_score == 0 else ''}",
                f"희망 전세금: {existing_loan}만원",
                f"희망 대출액: {f'{target_amount:,}만원' if target_amount > 0 else '미정'}"
            ]

            input_data = ", ".join(profile_lines)

            # RAG 검색 
            _, contexts_text = search_context(input_data)
            context_str = build_context_string(contexts_text)

            # 모델 전달용 메시지 (RAG context 포함)
            model_msg = (
                f"[입력_고객프로필]\n{input_data}\n\n"
                f"[RAG_검색문서]\n{context_str}\n\n"
                "[응답_요청]\n"
                "- RAG 근거를 우선 사용하고 불확실한 내용은 '확인 필요'로 표시\n"
                "- turn별 고정 포맷을 지켜 1turn 답변 생성"
            )

            # UI 표시용 메시지 (context 제외)
            profile_text_display = "\n".join(profile_lines)
            display_msg = (
                "[입력_고객프로필]"
                f"{profile_text_display}\n\n"
            )

            st.session_state.messages.append(HumanMessage(content=model_msg))
            st.session_state.display_messages.append({"role": "user", "content": display_msg})
            st.rerun()

        else:
            st.warning("연소득을 입력해주세요")

    st.divider()
    st.markdown(f"**🤖 모델**\n\n`{FINETUNED_MODEL[:35]}`...")
    if st.button("🪣 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.display_messages = []
        st.rerun()


# ------------------------ 메인화면: 대화 기록 출력 ------------------------
# 환영 메시지: 항상 상단에 표시되도록 설정
with st.chat_message("assistant", avatar="🏠"):
    st.markdown(
        """안녕하세요! 👋 **국민은행·신한은행·하나은행 전세자금대출 전문 AI 상담사**입니다.

아래 형식으로 입력해주시면 학습 데이터 포맷에 맞춰 더 정확하게 진단할 수 있습니다.

`희망은행=국민/신한/하나/비교해서추천, 대출목적=아파트·오피스텔·빌라 전세계약(신규/갱신), 연소득=숫자만원, 신용점수=숫자, 희망 전세금=금액, 희망 대출액=금액`

답변은 멀티턴 고정 구조로 안내합니다:
- 1턴: **판정요약 → 근거(소득·신용·희망금액·기본자격) → 쉬운용어설명 → 다음확인질문**
- 2턴: **옵션비교 → 추천안 → 쉬운용어설명 → 추가확인질문**
- 3턴: **실행계획(체크리스트) → 필요서류·준비사항 → 주의리스크 → 쉬운용어설명 → 마무리안내**

왼쪽 사이드바에 프로필을 입력하거나, 아래에 직접 질문해 주세요! 😊

> 💡 **예시 질문**
> - 희망은행=신한, 대출목적=오피스텔 전세계약(갱신), 연소득=4,200만원, 신용점수=742점, 희망 전세금=3억1천만원, 희망 대출액=2억4천만원
> - 2turn: "신한은행에서 한도 때문에 막힐 가능성이 큰가요?"
> - 3turn: "부족할 수 있다면 어떤 순서로 보완하면 좋을지 체크리스트 주세요."
        
"""
    )

# 대화 이력 표시 
for msg in st.session_state.display_messages:
    avatar = "👩🏻" if msg["role"] == "user" else "🏠"
    with st.chat_message(msg["role"], avatar=avatar):
        # st.chat_message("user", avatar="👩🏻")
        st.markdown(msg["content"])

# 사용자 입력 처리 
if prompt := st.chat_input("예) 희망은행=신한은행, 대출목적=오피스텔 전세계약(갱신), 연소득=4,200만원 ..."):
    # RAG 검색 결과 갖고오기 
    _, contexts_text = search_context(prompt)
    context_str = build_context_string(contexts_text)

    # 모델 전달용 user prompt 작성 
    model_prompt = (
                f"[입력_고객프로필]\n{prompt}\n\n"
                f"[RAG_검색문서]\n{context_str}\n\n"
                "[응답_요청]\n"
                "- RAG 근거를 우선 사용하고 불확실한 내용은 '확인 필요'로 표시\n"
                "- turn별 고정 포맷을 지켜 1turn 답변 생성"
    )

    # UI 출력용 user promopt 작성
    display_prompt = (
                "[입력_고객프로필]"
                f"{prompt}\n\n"
    )

    st.session_state.messages.append(HumanMessage(content=model_prompt))
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    st.rerun()

    with st.chat_message("user", avatar="👩🏻"):
        st.markdown(prompt)

# AI 응답 생성 (스트리밍)
if st.session_state.messages and isinstance(
    st.session_state.messages[-1], HumanMessage
):
    with st.chat_message("assistant", avatar="🏠"):
        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + st.session_state.messages

        full_response = ""
        placeholder = st.empty()

        # 스트리밍 출력 
        for chunk in model.stream(full_messages):
            full_response += chunk.content 
            placeholder.markdown(full_response + "|") # 타이핑 커서 효과

    # 세션 저장 
    st.session_state.messages.append(AIMessage(content=full_response))
    st.session_state.display_messages.append({"role": "assistant",
                                              "content": full_response})