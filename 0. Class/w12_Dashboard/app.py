import first_page as fp
import second_page as sp
import streamlit as st

def main():
    st.sidebar.title("menu")
    selection=st.sidebar.radio(
        "Go to",
        ("Page 1", "Page 2")
    )
    if selection == "Page 1":
        fp.page1()
    elif selection == "Page 2":
        sp.page2()

if __name__ == "__main__":
    main()