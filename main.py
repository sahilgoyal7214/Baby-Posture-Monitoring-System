import streamlit as st
import numpy as np
import joblib 


knn_classifier_uncovered= joblib.load("models/knn_classifier_uncovered.sav")
rf_classifier_uncovered= joblib.load(r"models\rf_classifier_uncovered.sav")
svm_classifier_uncovered= joblib.load(r"models\svm_classifier_uncovered.sav")

knn_classifier_cover1= joblib.load(r"models\knn_classifier_cover1.sav")
rf_classifier_cover1= joblib.load(r"models\rf_classifier_cover1.sav")
svm_classifier_cover1= joblib.load(r"models\svm_classifier_cover1.sav")

knn_classifier_cover2= joblib.load(r"models\knn_classifier_cover2.sav")
rf_classifier_cover2= joblib.load(r"models\rf_classifier_cover2.sav")
svm_classifier_cover2= joblib.load(r"models\svm_classifier_cover2.sav")

l= []
pridicted=[]
l2=[]
d = {'back':'images//back.jpg','lside':'images//left side.jpeg','rside':'images//right side.jpeg'}
d1 = {'back':'back','lside':'left side','rside':'right side'}
s = ["KNN","RF","SVM"]


css = """
<style>
    body {
        background-color: #222;
        color: #fff;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

st.markdown(f'''   <style>
   .stApp {{
   background-image: url("https://s3-alpha.figma.com/hub/file/1490175958/4f1af9be-e64d-4092-bcfb-43f14a79c1b7-cover.png");
   background-attachment: fixed;
    background-size: cover
   }}
   </style>   ''',   unsafe_allow_html=True)

st.title('Baby Posture Monitoring System')
st.divider()
uploaded_files = st.file_uploader("Upload Baby Image in npy format", accept_multiple_files=False)
if uploaded_files is not None:
    uploaded_files = np.load(uploaded_files)
    option = st.selectbox('what type of image you uploaded?',    ('SELECT','Uncovered', 'cover1', 'cover2'))
    model_type = st.selectbox('Select Model Type',    ('SELECT','KNN', 'RF', 'SVM','ALL'))
    if st.button("Predict"):
        if option=='Uncovered':
            if model_type=='KNN':

                knn=knn_classifier_uncovered.predict(uploaded_files)
                if knn==['back']:
                    st.image('images//back.jpg', width=200)
                    st.success('Predicted Label: back')
                elif knn==['lside']:
                    st.image('images//left side.jpeg', width=200)
                    st.success('Predicted Label: left side')
                elif knn==['rside']:
                    st.image('images//right side.jpeg', width=200)
                    st.success('Predicted Label: right side')
            elif model_type=='RF':
                rf=rf_classifier_uncovered.predict(uploaded_files)
                if rf==['back']:
                    st.image('images//back.jpg', width=200)
                    st.success('Predicted Label: back')
                elif rf==['lside']:
                    st.image('images//left side.jpeg', width=200)
                    st.success('Predicted Label: left side')
                elif rf==['rside']:
                    st.image('images//right side.jpeg', width=200)
                    st.success('Predicted Label: right side')

            elif model_type=='SVM':
                svm=svm_classifier_uncovered.predict(uploaded_files)
                if svm==['back']:
                    st.image('images//back.jpg', width=200)
                    st.success('Predicted Label: back')
                elif svm==['lside']:
                    st.image('images//left side.jpeg', width=200)
                    st.success('Predicted Label: left side')
                elif svm==['rside']:
                    st.image('images//right side.jpeg', width=200)
                    st.success('Predicted Label: right side')
            elif model_type=='ALL':
                knn=knn_classifier_uncovered.predict(uploaded_files)
                rf=rf_classifier_uncovered.predict(uploaded_files)
                svm=svm_classifier_uncovered.predict(uploaded_files)
                if knn==['back'] :
                    pridicted.append(list(knn))
                if knn==['lside']:
                    pridicted.append(list(knn))
                if knn==['rside']:
                    pridicted.append(list(knn))
                if rf==['back']:
                    pridicted.append(list(rf))
                if rf==['lside']:
                    pridicted.append(list(rf))
                if rf==['rside']:
                    pridicted.append(list(rf))
                if svm==['back']:
                    pridicted.append(list(svm))
                if svm==['lside']:
                    pridicted.append(list(svm))
                if svm==['rside']:
                    pridicted.append(list(svm))
        elif option=='cover2':
            if model_type=='KNN':
                knn_cover2=knn_classifier_cover2.predict(uploaded_files)
                if knn_cover2==['back']:
                    st.image('images//back.jpg', width=200)
                    st.success('Predicted Label: back')
                elif knn_cover2==['lside']:
                    st.image('images//left side.jpeg', width=200)
                    st.success('Predicted Label: left side')
                elif knn_cover2==['rside']:
                    st.image('images//right side.jpeg', width=200)
                    st.success('Predicted Label: right side')
            elif model_type=='RF':
                rf_cover2=rf_classifier_cover2.predict(uploaded_files)
                if rf_cover2==['back']:
                    st.image('images//back.jpg', width=200)
                    st.success('Predicted Label: back')
                elif rf_cover2==['lside']:
                    st.image('images//left side.jpeg', width=200)
                    st.success('Predicted Label: left side')
                elif rf_cover2==['rside']:
                    st.image('images//right side.jpeg', width=200)
                    st.success('Predicted Label: right side')
            elif model_type=='SVM':
                svm_cover2=svm_classifier_cover2.predict(uploaded_files)
                if svm_cover2==['back']:
                    st.image('images//back.jpg', width=200)
                    st.success('Predicted Label: back')
                elif svm_cover2==['lside']:
                    st.image('images//left side.jpeg', width=200)
                    st.success('Predicted Label: left side')
                elif svm_cover2==['rside']:
                    st.image('images//right side.jpeg', width=200)
                    st.success('Predicted Label: right side')
            elif model_type=='ALL':
                knn_cover2=knn_classifier_cover2.predict(uploaded_files)
                rf_cover2=rf_classifier_cover2.predict(uploaded_files)
                svm_cover2=svm_classifier_cover2.predict(uploaded_files)
                if knn_cover2==['back']:
                    pridicted.append(knn_cover2)
                if knn_cover2==['lside']:
                    pridicted.append(knn_cover2)
                if knn_cover2==['rside']:
                    pridicted.append(knn_cover2)
                if rf_cover2==['back']:
                    pridicted.append(rf_cover2)
                if rf_cover2==['lside']:
                    pridicted.append(rf_cover2)
                if rf_cover2==['rside']:
                    pridicted.append(rf_cover2)
                if svm_cover2==['back']:
                    pridicted.append(svm_cover2)
                if svm_cover2==['lside']:
                    pridicted.append(svm_cover2)
                if svm_cover2==['rside']:
                    pridicted.append(svm_cover2)
        elif option=='cover1':
            if model_type=='KNN':
                knn_cover1=knn_classifier_cover1.predict(uploaded_files)
                if knn_cover1==['back']:
                    st.image('images//back.jpg', width=200)
                    st.success('Predicted Label: back')
                elif knn_cover1==['lside']:
                    st.image('images//left side.jpeg', width=200)
                    st.success('Predicted Label: left side')
                elif knn_cover1==['rside']:
                    st.image('images//right side.jpeg', width=200)
                    st.success('Predicted Label: right side')
            elif model_type=='RF':
                rf_cover1=rf_classifier_cover1.predict(uploaded_files)
                if rf_cover1==['back']:
                    st.image('images//back.jpg', width=200)
                    st.success('Predicted Label: back')
                elif rf_cover1==['lside']:
                    st.image('images//left side.jpeg', width=200)
                    st.success('Predicted Label: left side')
                elif rf_cover1==['rside']:
                    st.image('images//right side.jpeg', width=200)
                    st.success('Predicted Label: right side')
            elif model_type=='SVM':
                svm_cover1=svm_classifier_cover1.predict(uploaded_files)
                if svm_cover1==['back']:
                    st.image('images//back.jpg', width=200)
                    st.success('Predicted Label: back')
                elif svm_cover1==['lside']:
                    st.image('images//left side.jpeg', width=200)
                    st.success('Predicted Label: left side')
                elif svm_cover1==['rside']:
                    st.image('right side.jpeg', width=200)
                    st.success('Predicted Label: right side')
            elif model_type=='ALL':
                knn_cover1=knn_classifier_cover1.predict(uploaded_files)
                rf_cover1=rf_classifier_cover1.predict(uploaded_files)
                svm_cover1=svm_classifier_cover1.predict(uploaded_files)
                if knn_cover1==['back']:
                    pridicted.append(knn_cover1)
                if knn_cover1==['lside']:
                    pridicted.append(knn_cover1)
                if knn_cover1==['rside']:
                    pridicted.append(knn_cover1)
                if rf_cover1==['back']:
                    pridicted.append(rf_cover1)
                if rf_cover1==['lside']:
                    pridicted.append(rf_cover1)
                if rf_cover1==['rside']:
                    pridicted.append(rf_cover1)
                if svm_cover1==['back']:
                    pridicted.append(svm_cover1)
                if svm_cover1==['lside']:
                    pridicted.append(svm_cover1)
                if svm_cover1==['rside']:
                    pridicted.append(svm_cover1)
    if model_type=='ALL':
        for i in pridicted:
            l.append(d[i[0]])
            l2.append(d1[i[0]])
        col=st.columns(3)
        for i , image in enumerate(l):
            col[i].image(image=image,width=200, caption=f'Predicted Label by {s[i]} : {l2[i]}')
    
