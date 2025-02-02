import streamlit as st
from utils import *

device = torch.device('cpu')
model = LogoEncoder(embedding_dim=1024).to(device)
model.load_state_dict(torch.load('models/model_sim.pth', map_location=device))
model.eval()

st.title('Logo similarity check')

st.header('Load reference logos')
ref_files = st.file_uploader('Select images (formats jpg, png)', type=['jpg', 'png'], accept_multiple_files=True)

st.header('Load test logo')
test_file = st.file_uploader('Select test image (formats jpg, png)', type=['jpg', 'png'])


if st.button('Compare logos') and ref_files and test_file:
    ref_images = []
    ref_embeddings = []

    for file in ref_files:
        image = load_image(file)
        ref_images.append(image)
        tensor = preprocess_image(image)
        with torch.no_grad():
            emb = model(tensor)
        ref_embeddings.append(emb)

    test_image = load_image(test_file)
    test_tensor = preprocess_image(test_image)
    with torch.no_grad():
        test_embedding = model(test_tensor)

    max_sim = max(F.cosine_similarity(emb, test_embedding) for emb in ref_embeddings).item()

    cols = st.columns(len(ref_images) + 1)
    for idx, image in enumerate(ref_images):
        cols[idx].image(image, caption=f'Reference {idx+1}', use_container_width=True)
    cols[idx+1].image(test_image, caption='Test image', use_container_width=True)

    st.write(f'Objects similarity: **{max_sim:.4f}**')
    if max_sim > 0.7:
        st.success('Objects match!')
    else:
        st.error('Objects do not match!')
