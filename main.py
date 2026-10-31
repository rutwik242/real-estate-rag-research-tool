import streamlit as st

from rag import generate_answer, process_urls

st.title("Real Estate Research Tool")

# Initialize session state
if "processed_urls" not in st.session_state:
    st.session_state.processed_urls = []


# Sidebar: URL inputs
st.sidebar.header("Article URLs")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

urls = list(dict.fromkeys(url.strip() for url in (url1, url2, url3) if url.strip()))

process_url_button = st.sidebar.button("Process URLs", type="primary")

status_placeholder = st.empty()


# Process submitted URLs
if process_url_button:
    # Reset the previous processing state
    st.session_state.processed_urls = []

    if not urls:
        st.warning("Please provide at least one URL.")

    else:
        try:
            with st.spinner("Processing articles..."):
                for status in process_urls(urls):
                    status_placeholder.info(status)

            # Save URLs only after successful processing
            st.session_state.processed_urls = urls.copy()

            status_placeholder.success("All articles are ready for research.")

        except Exception as e:  # noqa: BLE001
            status_placeholder.error(f"Failed to process URLs: {e}")


# Display question input only when URLs are ready
if st.session_state.processed_urls:
    if urls != st.session_state.processed_urls:
        st.info("The article URLs have changed. Please click Process URLs again.")

    else:
        st.divider()

        st.subheader("Ask a Question")

        query = st.text_input(
            "Question",
            placeholder="What are the key developments in the real estate market?",
        )

        st.caption("Press Enter to get your answer.")

        if query.strip():
            try:
                with st.spinner("Generating answer..."):
                    answer, sources = generate_answer(
                        query, st.session_state.processed_urls
                    )

                st.subheader("Answer")
                st.write(answer)

                if sources:
                    st.subheader("Sources")

                    for source in sources.replace(",", "\n").splitlines():
                        if source.strip():
                            st.write(source.strip())

            except Exception as e:  # noqa: BLE001
                st.error(f"Unable to generate answer: {e}")
