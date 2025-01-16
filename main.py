import streamlit as st
import os
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI
from PIL import Image
import pdf2image
import faiss
import pytesseract
import tempfile
import base64
from langdetect import detect
import arabic_reshaper
from bidi.algorithm import get_display

# Configure page to support RTL
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Add custom CSS for RTL support
st.markdown("""
    <style>
        .stApp {
            direction: rtl;
        }
        .stButton button {
            float: right;
        }
        .stTextInput input {
            text-align: right;
            direction: rtl;
        }
        .stMarkdown {
            text-align: right;
        }
        div[data-testid="stMarkdownContainer"] {
            direction: rtl;
            text-align: right;
        }
        div[data-testid="stMarkdownContainer"] > * {
            direction: rtl;
            text-align: right;
            unicode-bidi: bidi-override;
        }
        .persian-text {
            font-family: 'Vazirmatn', 'Iran Sans', 'Tahoma', sans-serif;
            line-height: 1.6;
        }
    </style>
""", unsafe_allow_html=True)

class PersianDocumentRetriever:
    def __init__(self, api_key: str, base_url: str = None):
        """Initialize the document retriever with OpenAI API key and optional base URL."""
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.document_embeddings = []
        self.page_images = []
        self.index = None

    def _reshape_persian_text(self, text: str) -> str:
        """Reshape Persian text for proper display."""
        reshaped_text = arabic_reshaper.reshape(text)
        return get_display(reshaped_text)

    def _preprocess_image_for_persian_ocr(self, image: Image) -> Image:
        """Preprocess image for better Persian OCR results."""
        # Convert to grayscale
        img_gray = image.convert('L')
        # Enhance contrast
        return img_gray.point(lambda x: 0 if x < 128 else 255, '1')

    def process_pdf(self, pdf_path: str, poppler_path: str = None):
        """Process a PDF document and create embeddings for each page."""
        # Convert PDF to images
        images = pdf2image.convert_from_path(
            pdf_path,
            poppler_path=poppler_path
        )
        self.page_images = images
        
        # Process each page
        embeddings = []
        progress_bar = st.progress(0)
        for i, img in enumerate(images):
            # Get embedding for the page
            embedding = self._get_page_embedding(img)
            embeddings.append(embedding)
            # Update progress
            progress = (i + 1) / len(images)
            progress_bar.progress(progress)
            
        # Store embeddings and create index
        self.document_embeddings = embeddings
        self._create_index(embeddings)

    def _get_page_embedding(self, image: Image) -> np.ndarray:
        """Get embedding for a page using Persian-aware OCR and text embeddings."""
        # Preprocess image for Persian OCR
        processed_image = self._preprocess_image_for_persian_ocr(image)
        
        # Extract text using OCR with Persian language support
        text = pytesseract.image_to_string(
            processed_image,
            lang='fas',  # Use Persian language model
            config='--psm 3'  # Page segmentation mode for multiple columns
        )
        
        # Get embedding directly from Persian text
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"  # Using the latest model that better supports Persian
        )
        
        return np.array(response.data[0].embedding)

    def _create_index(self, embeddings: List[np.ndarray]):
        """Create FAISS index for fast similarity search."""
        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))

    def search(self, query: str, k: int = 3) -> List[Dict[Any, Any]]:
        """Search for relevant pages given a Persian text query."""
        # Get query embedding directly from Persian text
        query_response = self.client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = np.array(query_response.data[0].embedding)
        
        # Search index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), 
            k
        )
        
        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'page_num': idx + 1,
                'distance': float(distances[0][i]),
                'page_image': self.page_images[idx]
            })
            
        return results

    def get_answer(self, query: str, context_image: Image) -> str:
        """Get answer using Persian query and OCR-extracted text from the image."""
        # Preprocess image for Persian OCR
        processed_image = self._preprocess_image_for_persian_ocr(context_image)
        
        # Extract text from image with Persian support
        text = pytesseract.image_to_string(
            processed_image,
            lang='fas',
            config='--psm 3'
        )
        
        # Format prompt in Persian
        prompt = f"متن:\n{text}\n\nسوال: {query}\n\nپاسخ:"
        
        # Get response from GPT with Persian system message
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4 for better Persian support
            messages=[
                {"role": "system", "content": "شما یک دستیار مفید هستید که به سوالات به زبان فارسی پاسخ می‌دهید. لطفاً پاسخ‌های خود را به زبان فارسی و به صورت واضح و مختصر ارائه دهید."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250
        )
        
        return self._reshape_persian_text(response.choices[0].message.content)

def format_persian_text(text: str) -> str:
    """Format Persian text for proper HTML display with RTL support."""
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return f'<div dir="rtl" style="text-align: right; unicode-bidi: bidi-override;">{bidi_text}</div>'

def main():
    # Persian translations for UI text
    UI_TRANSLATIONS = {
        "title": "سیستم جستجو و پرسش و پاسخ اسناد",
        "config_header": "پیکربندی",
        "api_key_label": "کلید آی پی آی",
        "base_url_label": "آدرس پایه آی پی آی (اختیاری)",
        "api_key_warning": "لطفاً  کلید آی پی آی خود را در نوار کناری وارد کنید.",
        "upload_label": "یک سند پی دی اف آپلود کنید",
        "processing": "در حال پردازش سند...",
        "success": "سند با موفقیت پردازش شد!",
        "search_header": "جستجو و پرسش سؤالات",
        "query_label": "سؤال یا عبارت جستجوی خود را وارد کنید",
        "searching": "در حال جستجو...",
        "search_results": "نتایج جستجو",
        "result": "نتیجه",
        "page": "صفحه",
        "get_answer": "دریافت پاسخ از صفحه",
        "answer": "پاسخ"
    }
    
    st.title(UI_TRANSLATIONS["title"])
    
    # Sidebar for API configuration
    st.sidebar.header(UI_TRANSLATIONS["config_header"])
    api_key = st.sidebar.text_input(UI_TRANSLATIONS["api_key_label"], type="password")
    base_url = st.sidebar.text_input(UI_TRANSLATIONS["base_url_label"])
    
    if not api_key:
        st.warning(UI_TRANSLATIONS["api_key_warning"])
        return

    # Initialize retriever
    retriever = PersianDocumentRetriever(api_key=api_key, base_url=base_url)
    
    # File upload
    uploaded_file = st.file_uploader(UI_TRANSLATIONS["upload_label"], type="pdf")
    
    if uploaded_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name
        
        # Process document
        with st.spinner(UI_TRANSLATIONS["processing"]):
            retriever.process_pdf(pdf_path)
        st.success(UI_TRANSLATIONS["success"])
        
        # Search interface
        st.header(UI_TRANSLATIONS["search_header"])
        query = st.text_input(UI_TRANSLATIONS["query_label"])
        
        if query:
            # Search for relevant pages
            with st.spinner(UI_TRANSLATIONS["searching"]):
                results = retriever.search(query)
            
            # Display results
            st.subheader(UI_TRANSLATIONS["search_results"])
            for i, result in enumerate(results):
                with st.expander(f"{UI_TRANSLATIONS['result']} {i+1} ({UI_TRANSLATIONS['page']} {result['page_num']})"):
                    # Display page image
                    st.image(result['page_image'], caption=f"{UI_TRANSLATIONS['page']} {result['page_num']}")
                    
                    # Get and display answer
                    if st.button(f"{UI_TRANSLATIONS['get_answer']} {result['page_num']}", key=f"btn_{i}"):
                        with st.spinner(UI_TRANSLATIONS["searching"]):
                            answer = retriever.get_answer(query, result['page_image'])
                        # Display answer with proper Persian formatting
                        st.markdown(
                            f"{UI_TRANSLATIONS['answer']}: {format_persian_text(answer)}",
                            unsafe_allow_html=True
                        )
        
        # Cleanup temporary file
        os.unlink(pdf_path)

if __name__ == "__main__":
    main()