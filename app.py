import streamlit as st
import os
import numpy as np
from groq import Groq
import cv2
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found. Please add it to secrets or .env file.")
    st.stop()

try:
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.stop()

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0d0d0d 100%);
}
.caption-box {
    background: rgba(20, 20, 20, 0.95);
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #00ff88;
    box-shadow: 0 15px 35px rgba(0,0,0,0.5);
    margin: 20px 0;
    color: #e0e0e0;
    font-size: 18px;
    line-height: 1.8;
}
.stButton>button {
    background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
    color: #000000;
    font-weight: bold;
    border: none;
    border-radius: 50px;
    padding: 15px 30px;
    font-size: 18px;
    box-shadow: 0 4px 15px rgba(0, 255, 136, 0.3);
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(0, 255, 136, 0.4);
}
.main-header {
    font-size: 3.5rem;
    background: linear-gradient(135deg, #00ff88 0%, #00cc6a 50%, #00ff88 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}
.sub-header {
    text-align: center;
    color: #888;
    margin-bottom: 2rem;
}
[data-testid="metric-container"] {
    background: rgba(20, 20, 20, 0.8);
    border-radius: 10px;
    padding: 15px;
    border: 1px solid #333;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🖼️ AI Image Caption Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an image and let AI describe it</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    
    model_choice = st.selectbox(
        "Choose AI Model",
        [
            "Llama-3.3-70B (Most Powerful)",
            "Llama-3.1-8B (Fast & Efficient)",
            "Gemma2-9B (Balanced)"
        ]
    )
    
    model_map = {
        "Llama-3.3-70B (Most Powerful)": "llama-3.3-70b-versatile",
        "Llama-3.1-8B (Fast & Efficient)": "llama-3.1-8b-instant",
        "Gemma2-9B (Balanced)": "gemma2-9b-it"
    }
    
    tone = st.selectbox(
        "Description Style",
        [
            "Detailed & Descriptive",
            "Short & Punchy",
            "Professional",
            "Humorous & Fun",
            "Poetic & Artistic"
        ]
    )
    
    tone_map = {
        "Detailed & Descriptive": "Describe in vivid detail. Include colors, objects, people, mood.",
        "Short & Punchy": "Describe in 2-3 short, impactful sentences.",
        "Professional": "Provide a professional, technical description.",
        "Humorous & Fun": "Describe with a funny, witty tone.",
        "Poetic & Artistic": "Describe with poetic language and metaphors."
    }
    
    st.divider()
    st.markdown("Use high-quality images with clear subjects.")

if "caption_history" not in st.session_state:
    st.session_state.caption_history = []
if "current_caption" not in st.session_state:
    st.session_state.current_caption = ""

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "webp", "bmp"]
)

if uploaded_file:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image_rgb, caption="Your Image", use_container_width=True)
        
        height, width = image.shape[:2]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Size", f"{width}×{height}")
        with col2:
            st.metric("Format", uploaded_file.type.split('/')[1].upper())
        with col3:
            st.metric("File Size", f"{uploaded_file.size // 1024} KB")
        
        if st.button("Generate Caption", type="primary", use_container_width=True):
            with st.spinner("AI is analyzing your image..."):
                try:
                    filename = os.path.splitext(uploaded_file.name)[0]
                    
                    response = client.chat.completions.create(
                        model=model_map[model_choice],
                        messages=[
                            {"role": "system", "content": "You are an expert image describer."},
                            {"role": "user", "content": f"""
                            {tone_map[tone]}
                            
                            Image filename: "{filename}"
                            Image dimensions: {width}x{height}
                            
                            Describe this image in detail.
                            """}
                        ],
                        temperature=0.7,
                        max_tokens=300
                    )
                    
                    st.session_state.current_caption = response.choices[0].message.content
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        if st.session_state.current_caption:
            st.markdown("### AI Description")
            st.markdown(f'<div class="caption-box">{st.session_state.current_caption}</div>', unsafe_allow_html=True)
            
            st.download_button(
                label="Download Caption",
                data=st.session_state.current_caption,
                file_name=f"{filename}_caption.txt",
                mime="text/plain"
            )
            
            st.markdown("### Rate This Caption")
            rating = st.slider("How accurate is this description?", 1, 5, 3)
            
            if rating >= 4:
                st.balloons()
                st.success("Amazing!")
            elif rating <= 2:
                st.warning("Try a different style")
            
            if st.button("Save to History"):
                st.session_state.caption_history.append({
                    "image": uploaded_file.name,
                    "caption": st.session_state.current_caption,
                    "model": model_choice,
                    "tone": tone,
                    "rating": rating
                })
                st.success("Saved!")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

if st.session_state.caption_history:
    st.divider()
    st.markdown("### History")
    
    for entry in reversed(st.session_state.caption_history[-5:]):
        with st.expander(f"{entry['image']} - Rating: {'⭐' * entry.get('rating', 3)}"):
            st.write(entry['caption'])
            
    if st.button("Clear History"):
        st.session_state.caption_history = []
        st.session_state.current_caption = ""
        st.rerun()

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #555; font-size: 0.8rem;'>
    Powered by Groq AI | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
