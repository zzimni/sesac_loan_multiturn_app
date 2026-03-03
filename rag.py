import os 
import streamlit as st 
from config import OPENAI_EMBEDDING_MODEL, PDF_PATH
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv 

load_dotenv()

# 임베딩 객체 정의 
@st.cache_resource
def get_embedding():
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

# vectorstore 생성 / 로드 
@st.cache_resource
def get_vectorstore(
    persist_directory: str = "./jeonse_Chroma",
    collection_name: str = "jeonse_loan_products"
) -> Chroma:
    """
    Vectorstore가 존재하면 로드하고, 없으면 새로 생성합니다.
    """

    embedding = get_embedding()

    # 기존 vectorstore 로드 시도 
    vectorstore_exists = (
        os.path.exists(persist_directory)
        and os.path.isdir(persist_directory)
    )

    if vectorstore_exists:
        try:
            vectorstore = Chroma(
                embedding_function=embedding,
                persist_directory=persist_directory,
                collection_name=collection_name
            )

            if vectorstore._collection.count() > 0:
                print(
                    f"✅ 기존 vectorstore 로드 완료"
                    f"(문서 수: {vectorstore._collection.count()})"
                )

                return vectorstore
            else:
                print("⚠️ vectorstore는 존재하지만 문서가 없습니다. 재생성합니다.")
        except Exception as e:
            print(f"⚠️ vectorstore 로드 오류: {e}. 재생성합니다.")
    else:
        print("⚠️ vectorstore가 존재하지 않습니디. 새로 생성합니다.")

    # 새 vectorstore 생성 
    # 1) 3개 은행 PDF 로드 
    all_docs = []
    for pdf_path in PDF_PATH:
        # 파일이 해당 경로에 존재 하지 않을 경우
        if not os.path.exists(pdf_path):
            print(f"⚠️ PDF 파일을 찾을 수 없습니다: {pdf_path}")
            continue
        # 파일이 해당 경로에 존재할 경우
        bank_docs = PyMuPDFLoader(pdf_path).load()

        # 어느 은행 문서인지 메타데이터에 추가 
        bank_name = os.path.basename(pdf_path).replace("_전세자금대출_상품설명서.pdf", "")
        for doc in bank_docs:
            doc.metadata["bank"] = bank_name
        all_docs.extend(bank_docs)
        print(f" 📄 로드 완료: {bank_name} ({len(bank_docs)}페이지)")

    if not all_docs:
        # raise: 예외를 발생시켜서 프로그래밍을 중단시킴
        raise FileNotFoundError(
            "PDF 파일을 하나도 찾을 수 없습니다."
            "./data/ 폴더에 상품설명서 PDF를 배치해주세요."
        )
    
    # 2) 문서 분할 
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    split_docs = splitter.split_documents(all_docs)

    # 3) vectorstore 생성
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    print(
        f"✅ vectorstore 생성 완료"
        f"(총 청크 수: {vectorstore._collection.count()})"
    )
    return vectorstore


# 문서 검색 함수 정의 
def search_context(
        input_data: str,
        k: int = 4,
        persist_directory: str = "./jeonse_chroma",
        collection_name: str = "jeonse_loan_products"
) -> tuple[list, list[str]]:
    """
    쿼리와 관련된 전세자금 대출 상품 정보를 검색합니다. 

    Returns
    -------
    contexts_with_metadata : list[Document]
    contexts_only_text     : list[str]      - "[은행명] 내용" 형식
    """

    vectorstore = get_vectorstore(persist_directory, collection_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    contexts_with_metadata = retriever.invoke(input_data)

    contexts_only_text = []
    for chunk in contexts_with_metadata:
        bank = chunk.metadata.get("bank", "")
        prefix = f"[{bank}]" if bank else ""
        contexts_only_text.append(
            f"[START]🏦{prefix}:\n"
            f"{chunk.page_content}[END]"
        )
        print(f"📋 상위 검색 문서 [{bank}]:\n{contexts_with_metadata[0].page_content[:200]}")

    return contexts_with_metadata, contexts_only_text

# 검색된 컨텍스트를 하나의 문자열로 합치는 헬퍼 정의 
def build_context_string(contexts_only_text: list[str]) -> str:
    """검색된 청크들을 개행으로 연결하여 프롬프트 삽입용 문자열로 반환"""
    return "\n".join(contexts_only_text)


# 단독 실행 테스트 
if __name__ == "__main__": 
    import sys 

    # sys.argv: 
    # - 터미널에서 Python 파일을 실행할 때 같이 넘겨주는 값들을 담고있는 리스트
    # - 파이썬에서 명령줄 인자(command-line arguments)를 받는 방법 
    input_data = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "희망은행=신한은행, 대출목적=오피스텔 전세계약(갱신), 연소득=4,200만원, 신용점수=90점, 희망 전세금=3억, 희망 대출액 2억"
    )
    print(f"\n🔎 검색 쿼리: {input_data}\n")
    docs, texts = search_context(input_data)
    for i, t in enumerate(texts, 1):
        print(f"--- 청크 {i} ---\n{t}\n")