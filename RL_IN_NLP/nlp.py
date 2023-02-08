from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
import tensorflow as tf
import stanza
from streamlit_option_menu import option_menu
import numpy as np
import random
# from functions import *
import json
import streamlit as st
from tensorflow import keras
def reform():
    with open("reform.json") as toolkit:
        table=json.load(toolkit)
    if len(table)==0:
        table["1"]=1
        table["2"]=1
        table["3"]=1
    return table
st.set_page_config(layout="wide",initial_sidebar_state="collapsed")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
# model_ = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
acurray=0.01



user_input = st.text_input("Find the needed information from the policies")
with open("POLICY.txt",encoding="utf-8") as value:
    text=value.readlines()
text="".join(text)
text=st.text_area("text to analyse",text)
item_k=["None"]
if len(user_input)!=0:
    with open("items.json") as speed_check:
        work_file=json.load(speed_check)
    if str((text,user_input)) in work_file:
        for j in work_file[str((text,user_input))]:
            item_k.append(j)
    else:
        sentences=[]
        nlp = stanza.Pipeline(lang='en', processors='tokenize')
        doc = nlp(text)
        for i, sentence in enumerate(doc.sentences):
            sentence_=""
            for token in sentence.tokens:
                sentence_+=token.text+" "
            sentences.append(sentence_)
        def direct_search(value):
            direct_list=[]
            for lines in sentences:
                if value.lower() in lines.lower():
                    direct_list.append(lines)
            return direct_list
        def sentence_symentics(value):
            select_sentence_list=[]
            current=model.encode(value)
            for lines in sentences:
                line_code=model.encode(lines)
                if util.cos_sim(line_code,current)[0][0]>acurray:
                    select_sentence_list.append(lines)
            return select_sentence_list
        def token_model(value):
            token_list=[]
            value_token=tokenizer.tokenize(value)
            current=model.encode(value)
            for lines in sentences:
                line_token=tokenizer.tokenize(lines)
                for word_ in line_token:
                    value_encode=model.encode(word_)
                    # print(util.cos_sim(current,value_encode))
                    if util.cos_sim(current,value_encode)[0][0]>0.45:
                        token_list.append(lines)
                        break
            return token_list    
        def selector_model(data_tokens):
            data_direct=direct_search(data_tokens)
            data_symmantic=sentence_symentics(data_tokens)
            token_symentics=token_model(data_tokens)
            return [data_direct,data_symmantic,token_symentics]
        data_token=str(user_input)
        check=selector_model(data_token)
        value_print=""
        reform_val=reform()

        paragraphs=[]
        for i,file_ in enumerate(check):
            print("*********",i,"*****************" ,end="\n")
            para=""
            for val in file_:
                para=para+val+"\n"
            paragraphs.append((i+1,str(para)))
        value_pred=[]
        for i in reform_val:
            for j in range(reform_val[i]):
                value_pred.append(i)
        need=[]
        for i in reform_val:
            rand_idx = random.randrange(len(value_pred))
            remove_item=value_pred[rand_idx]
            need.append(value_pred[rand_idx])
            while(remove_item in value_pred):
                value_pred.remove(remove_item)  
        for seq in need:
            for j in paragraphs:
                if j[0]==int(seq):
                    item_k.append((j[-1],seq))
        with open("items.json","w") as speed_check:
            json.dump({str((text,user_input)):item_k},speed_check,indent=2)
while item_k.count("None")>1:
    item_k.remove("None")
for  complete_ in range(len(item_k)):
    if item_k[complete_][0]=="":
        item_k[complete_]=("function is not capable to find results",item_k[complete_][1])
selected = option_menu("select the prominent one",["None"]+[i[0][:200] for i in item_k[1:]])
if selected:
    reform_val=reform()
    for i in item_k:
        if i!="None" and i[0][:200]==selected :
            st.write(i[0])
            reform_val[i[1]]+=1
            with open("reform.json","w") as changed:
                json.dump(reform_val,changed,indent=2)
        
