import streamlit as st

def page1():
    st.title("Calculator Program")
    st.text("This is a calculator program and it received input and calculate it")

    with st.form("my_form"):
        a=st.number_input("1st Number")
        b=st.number_input("2nd Number")
        submitted = st.form_submit_button("Submit")

        if submitted:
            c=a*b
            st.write("Answer = ",c)