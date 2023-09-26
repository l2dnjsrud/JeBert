import torch
import streamlit as st
import pandas as pd


from transformers import (
    EncoderDecoderModel,
    BertTokenizer,
)

import sentencepiece as spm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#from lib.tokenization_kobert import KoBertTokenizer

if 'tokenizer' not in st.session_state:
    #src_tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    #trg_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer = BertTokenizer('./dump/wpm-vocab-extend-30522.txt', do_lower_case=False)
    st.session_state.tokenizer = tokenizer
else:
    tokenizer = st.session_state.tokenizer

@st.cache
def get_model(bos_token_id = None):
    model = EncoderDecoderModel.from_pretrained('./dump/models/best_model(final_jeko)').to(device)
    #model.config.decoder_start_token_id = bos_token_id
    model.eval()
    #model.cuda()

    return model

@st.cache
def get_model2(bos_token_id = None):
    model = EncoderDecoderModel.from_pretrained('./dump/models/best_model(final_koje)').to(device)
    #model.config.decoder_start_token_id = bos_token_id
    model.eval()
    #model.cuda()

    return model


model = get_model(tokenizer.bos_token_id)
model2 = get_model2(tokenizer.bos_token_id)

# 페이지 구성
st.title("한국 표준어-제주도 방언 번역기")
st.subheader("한-제 번역기에 오신 것을 환영합니다!")

# 왼쪽에 사이드바 추가
add_selectbox = st.sidebar.selectbox("번역 방향을 선택해주세요!", 
                                     ('제주도 방언->표준어','표준어->제주도 방언')) 
# 선택 박스 만들기
#lan_option = st.selectbox('번역 방향을 선택해주세요!',
#                          ('제주도 방언->표준어','표준어->제주도 방언'))
st.sidebar.write("선택된 방향은 ", add_selectbox)


# 레이아웃
col1, col2 = st.columns(2)

with col1:
    st.subheader('제주도 방언-표준어 예시')
    df = {'제주도 방언':['혼저옵서예','빙애기','그것ᄀᆞ란 저 거세기 , 정지 .','예 . 그거 좀 ᄀᆞᆯ아 줍서 . 헛불 .','어디 갔단 왐수과?','놀당 갑서','어드레 감디?'],
      '표준어': ['어서오세요','병아리','그것보고 저 거시기 , 부엌.','예 . 그거 좀 얘기해 주십시오 . 헛불 .','어디 갔다 오십니까?','놀다가 가세요','어디 가세요?']}
    df = pd.DataFrame(df)
    st.table(df)

with col2:

    # 번역기
    st.subheader('제주도 방언 -> 표준어')
    if add_selectbox == '제주도 방언->표준어':
        kor = st.text_area("제주도 방언", placeholder="번역할 제주도 방언을 넣어주세요.")

        if st.button("번역!", help="해당 제주도 방언을 번역합니다."):
            embeddings = tokenizer(kor, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
            embeddings = {k: v.cuda() for k, v in embeddings.items()}
            output = model.generate(**embeddings, max_length = 256, eos_token_id=3)[0, 1:-1]
            st.text_area("표준어", value=tokenizer.decode(output[1:]), disabled=True)
    else:
        je = st.text_area('표준어', placeholder = '번역할 표준어를 넣어주세요.')

        if st.button('번역!', help = '해당 표준어를 번역합니다.'):
            embeddings = tokenizer(je, return_attention_mask = False, return_token_type_ids = False, return_tensors = 'pt')
            embeddings = {k:v.cuda() for k,v in embeddings.items()}
            output = model2.generate(**embeddings, max_length = 256, eos_token_id=3)[0, 1:-1]
            st.text_area('제주도 방언', value = tokenizer.decode(output[1:]), disabled = True)

