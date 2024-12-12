import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
import os
# OpenAI API 설정
os.environ["OPENAI_API_KEY"] = ""  # 실제 API 키 입력
# Streamlit 제목
st.title("💬 화장품 추천 챗봇 피카추 ⚡")
# CSV 파일 경로
csv_file_path = "data/total_reviews.csv"
# 대화 내역 초기화
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
# 벡터스토어와 QA 체인 초기화
if "qa_chain" not in st.session_state:
    if os.path.exists(csv_file_path):
        try:
            # CSV 파일 로드
            df = pd.read_csv(csv_file_path)
            # 텍스트 데이터 처리
            if '불용어 제거 리뷰' in df.columns:
                text_data = "\n".join(df['불용어 제거 리뷰'].dropna())
            else:
                text_data = "\n".join(df.iloc[:, 0].dropna())
            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)
            split_docs = text_splitter.split_text(text_data)
            # 벡터스토어 생성
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(split_docs, embeddings)
            # 메모리 생성
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            # QA 체인 생성
            llm = ChatOpenAI(
                model_name="gpt-4o",
                temperature=0.2,
                openai_api_key=os.environ.get("OPENAI_API_KEY")
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 500})
            st.session_state["qa_chain"] = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                memory=memory
            )
        except Exception as e:
            st.error(f"데이터 처리 중 오류 발생: {e}")
            st.stop()
    else:
        st.error(f"CSV 파일이 존재하지 않습니다: {csv_file_path}")
        st.stop()
# 채팅 메시지 형식으로 출력
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
# 사용자 입력
if user_query := st.chat_input("질문을 입력하세요 (예: '건성 피부에 적합한 크림 추천')"):
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(user_query)
    # 사용자 메시지 기록
    st.session_state["chat_history"].append({"role": "user", "content": user_query})
    # AI 응답 처리
    if "qa_chain" in st.session_state:
        try:
            response = st.session_state["qa_chain"].run(user_query)
            # AI 응답 메시지 표시
            with st.chat_message("assistant"):
                st.markdown(response)
            # AI 응답 기록
            st.session_state["chat_history"].append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"질문 처리 중 오류 발생: {e}")