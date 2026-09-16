import streamlit as st
from PIL import Image
from inference import DEVICE, generate_caption, load_checkpoint

CHECKPOINT_PATH = 'checkpoints/caption_attention_checkpoint.pt'


st.set_page_config(page_title='Captorian', page_icon=':camera:', layout='centered')
st.title('Captorian')
st.caption('Image captioning with ResNet50, spatial attention, and an LSTM decoder')

try:
    encoder, decoder, checkpoint = load_checkpoint(CHECKPOINT_PATH)
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Input image', use_container_width=True)
    if st.button('Generate caption', type='primary'):
        with st.spinner('Generating caption...'):
            caption = generate_caption(image, encoder, decoder, checkpoint['vocab_stoi'], checkpoint['vocab_itos'], checkpoint['max_length'])
        st.success(caption)
