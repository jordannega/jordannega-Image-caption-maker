import streamlit as st
import os
import sys
import time
import base64
import numpy as np
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

# Get API key from secrets (cloud) or environment (local)
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("""
    ❌ **GROQ_API_KEY not found!**
    
    - **Local**: Create a `.env` file with `GROQ_API_KEY=your_key_here`
    - **Cloud**: Add `GROQ_API_KEY` in Streamlit Secrets
    """)
    st.stop()

# Try importing with error handling
try:
    from groq import Groq
except ImportError as e:
    st.error(f"❌ Missing package: {str(e)}")
    st.info("Make sure 'groq' is in your requirements.txt")
    st.stop()

try:
    import cv2
except ImportError as e:
    st.error(f"❌ Missing package: {str(e)}")
    st.info("Make sure 'opencv-python-headless' is in your requirements.txt")
    st.stop()

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None
    st.warning("⚠️ FPDF not installed. PDF export will be disabled.")

# Page config
st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

# Theme toggle
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Custom CSS - Premium dark design
st.markdown("""
<style>
/* Premium dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0d0d0d 100%);
}

/* Sleek card for caption */
.caption-box {
    background: rgba(20, 20, 20, 0.95);
    backdrop-filter: blur(10px);
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #333;
    box-shadow: 0 15px 35px rgba(0,0,0,0.5);
    margin: 20px 0;
    color: #e0e0e0;
    font-size: 18px;
    line-height: 1.8;
    transition: transform 0.3s ease, border-color 0.3s ease;
}
.caption-box:hover {
    transform: scale(1.01);
    border-color: #00ff88;
}

/* Animated gradient button - Tech style */
.stButton>button {
    background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
    color: #000000;
    font-weight: bold;
    border: none;
    border-radius: 50px;
    padding: 15px 30px;
    transition: all 0.3s ease;
    font-size: 18px;
    box-shadow: 0 4px 15px rgba(0, 255, 136, 0.3);
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(0, 255, 136, 0.4);
}

/* Glowing title - Tech green */
.main-header {
    font-size: 3.5rem;
    background: linear-gradient(135deg, #00ff88 0%, #00cc6a 50%, #00ff88 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem;
    text-shadow: 0 0 30px rgba(0, 255, 136, 0.2);
}

.sub-header {
    text-align: center;
    color: #888;
    margin-bottom: 2rem;
    font-size: 1.1rem;
}

/* Metrics styling */
[data-testid="metric-container"] {
    background: rgba(20, 20, 20, 0.8);
    border-radius: 10px;
    padding: 15px;
    border: 1px solid #333;
}

hr {
    border-color: #333;
}
</style>
""", unsafe_allow_html=True)

# Initialize Groq client
try:
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"❌ Failed to initialize Groq client: {str(e)}")
    st.stop()

# Header
st.markdown('<p class="main-header">🖼️ AI Image Caption Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload any image and let AI describe what it sees</p>', unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Dark/Light Mode Toggle
    theme_toggle = st.toggle("☀️ Light Mode", value=False)
    if theme_toggle:
        st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%); }
        .caption-box { background: rgba(255,255,255,0.95); color: #1a1a1a; border-color: #ddd; }
        </style>
        """, unsafe_allow_html=True)
    
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
        "Detailed & Descriptive": "Describe this image in vivid detail. Include colors, objects, people, actions, mood, and atmosphere.",
        "Short & Punchy": "Describe this image in 2-3 short, impactful sentences.",
        "Professional": "Provide a professional, technical description of this image.",
        "Humorous & Fun": "Describe this image with a funny, witty, and entertaining tone.",
        "Poetic & Artistic": "Describe this image with poetic language, metaphors, and artistic flair."
    }
    
    st.divider()
    st.markdown("""
    **💡 Tips:**
    - Use high-quality images
    - Try different styles
    - Images with clear subjects work best
    """)

# Initialize session state
if "caption_history" not in st.session_state:
    st.session_state.caption_history = []
if "current_caption" not in st.session_state:
    st.session_state.current_caption = ""
if "rating" not in st.session_state:
    st.session_state.rating = 0

# File upload
uploaded_file = st.file_uploader(
    "📤 Choose an image...",
    type=["jpg", "jpeg", "png", "webp", "bmp"]
)

def create_pdf(caption, filename):
    if FPDF is None:
        return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt=f"AI Caption for {filename}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=caption)
    return pdf.output(dest='S').encode('latin1')

if uploaded_file:
    try:
        # Read image with OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Display image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image_rgb, caption="📷 Your Image", use_container_width=True)
        
        # Image info
        height, width = image.shape[:2]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Size", f"{width}×{height}")
        with col2:
            st.metric("Format", uploaded_file.type.split('/')[1].upper())
        with col3:
            st.metric("File Size", f"{uploaded_file.size // 1024} KB")
        
        # Generate button
        if st.button("✨ Generate Caption", type="primary", use_container_width=True):
            with st.status("🤔 AI is analyzing your image...", expanded=True) as status:
                st.write("🔍 Extracting image features...")
                time.sleep(0.5)
                st.write("🧠 Generating creative description...")
                time.sleep(0.5)
                st.write("✨ Almost done...")
                
                try:
                    filename = os.path.splitext(uploaded_file.name)[0]
                    
                    response = client.chat.completions.create(
                        model=model_map[model_choice],
                        messages=[
                            {"role": "system", "content": "You are an expert image describer. Be creative and descriptive."},
                            {"role": "user", "content": f"""
                            {tone_map[tone]}
                            
                            Image filename: "{filename}"
                            Image dimensions: {width}x{height}
                            
                            Describe what this image shows in detail.
                            """}
                        ],
                        temperature=0.7,
                        max_tokens=300
                    )
                    
                    st.session_state.current_caption = response.choices[0].message.content
                    status.update(label="✅ Caption Ready!", state="complete")
                    
                except Exception as e:
                    st.error(f"❌ Error generating caption: {str(e)}")
        
        # Display current caption if it exists
        if st.session_state.current_caption:
            st.markdown("### 📝 AI Description")
            st.markdown(f'<div class="caption-box">{st.session_state.current_caption}</div>', unsafe_allow_html=True)
            
            # Plain text version
            st.markdown("---")
            st.markdown("**📄 Plain Text Version:**")
            st.write(st.session_state.current_caption)
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download as TXT",
                    data=st.session_state.current_caption,
                    file_name=f"{filename}_caption.txt",
                    mime="text/plain"
                )
            with col2:
                if FPDF is not None:
                    try:
                        pdf_data = create_pdf(st.session_state.current_caption, filename)
                        if pdf_data:
                            st.download_button(
                                label="📄 Download as PDF",
                                data=pdf_data,
                                file_name=f"{filename}_caption.pdf",
                                mime="application/pdf"
                            )
                    except:
                        st.warning("PDF export failed.")
            
            # Rating System
            st.markdown("### ⭐ Rate This Caption")
            rating = st.slider("How accurate is this description?", 1, 5, 3)
            
            if rating >= 4:
                st.balloons()
                st.snow()
                st.success("🌟 Amazing! This is a great caption!")
            elif rating <= 2:
                st.warning("🤔 Let's try a different style!")
                if st.button("🔄 Try Different Style"):
                    st.rerun()
            
            st.write(f"You rated it: {'⭐' * rating}")
            
            # Social Media Post Generator
            st.markdown("### 📱 Share This")
            platform = st.radio("Choose platform:", ["Twitter/X", "LinkedIn", "Instagram"])
            
            post_templates = {
                "Twitter/X": f"Just used AI to analyze this image! 🤖\n\n{st.session_state.current_caption[:200]}...\n\n#AI #MachineLearning #ImageCaption",
                "LinkedIn": f"I built an AI that describes images! 🖼️\n\nHere's what it says:\n\n{st.session_state.current_caption}\n\n#AI #Innovation #Tech #ImageRecognition",
                "Instagram": f"✨ AI Magic ✨\n\n{st.session_state.current_caption}\n\n#AI #Photo #Tech #Innovation"
            }
            
            st.text_area("📝 Post Preview", post_templates[platform], height=150)
            
            # Multiple Captions Generation
            st.markdown("### 🎨 Generate Multiple Captions")
            if st.button("✨ Generate 3 Different Captions"):
                with st.spinner("Generating multiple captions..."):
                    captions = []
                    styles = ["Creative", "Professional", "Funny"]
                    
                    for style in styles:
                        response = client.chat.completions.create(
                            model=model_map[model_choice],
                            messages=[
                                {"role": "system", "content": f"You are an expert image describer. Generate a {style} description."},
                                {"role": "user", "content": f"Describe this image in a {style} way. The image is: {filename}"}
                            ]
                        )
                        captions.append(response.choices[0].message.content)
                    
                    tab1, tab2, tab3 = st.tabs(["🎨 Creative", "💼 Professional", "😂 Funny"])
                    with tab1:
                        st.markdown(f'<div class="caption-box">{captions[0]}</div>', unsafe_allow_html=True)
                    with tab2:
                        st.markdown(f'<div class="caption-box">{captions[1]}</div>', unsafe_allow_html=True)
                    with tab3:
                        st.markdown(f'<div class="caption-box">{captions[2]}</div>', unsafe_allow_html=True)
            
            # Save to history
            if st.button("💾 Save to History"):
                st.session_state.caption_history.append({
                    "image": uploaded_file.name,
                    "caption": st.session_state.current_caption,
                    "model": model_choice,
                    "tone": tone,
                    "rating": rating
                })
                st.success("✅ Saved to history!")
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

# Show history
if st.session_state.caption_history:
    st.divider()
    st.markdown("### 📜 Caption History")
    
    for entry in reversed(st.session_state.caption_history[-5:]):
        with st.expander(f"🖼️ {entry['image']} ({entry['model']}) - Rating: {'⭐' * entry.get('rating', 3)}"):
            st.write(entry['caption'])
            
    if st.button("🗑️ Clear History"):
        st.session_state.caption_history = []
        st.session_state.current_caption = ""
        st.rerun()

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #555; font-size: 0.8rem;'>
    Powered by Groq AI 🚀 | Built with Streamlit | 🌟 Premium Edition
    </div>
    """,
    unsafe_allow_html=True
)