import streamlit as st

def page1():
    st.title("Page 1")
    st.write("This is page 1")

def page2():
    st.title("Page 2")
    st.write("This is page 2")

def page3():
    st.title("Page 3")
    st.write("This is page 3")

def main():
    page_names = ["Page 1", "Page 2", "Page 3"]
    page_funcs = [page1, page2, page3]

    st.sidebar.title("Navigation")
    selected_page = st.sidebar.selectbox("Go to:", page_names)

    for page_name, page_func in zip(page_names, page_funcs):
        if selected_page == page_name:
            page_func()
            break

if __name__ == "__main__":
    main()