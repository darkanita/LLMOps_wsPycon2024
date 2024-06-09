import streamlit as st

st.set_page_config(layout="wide")

def main():
    st.write(
    """
    # LLMOps Workshop - PyCon 2024

    This Streamlit dashboard is intended to serve as a proof of concept of LLMs functionality for chat with PDFs.  It is not intended to be a production-ready application.

    Use the navigation bar on the left to navigate to the different pages of the dashboard.

    Pages include:
    1. Chat with Data. 
    """
    )

if __name__ == "__main__":
    main()