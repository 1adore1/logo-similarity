# Logo Similarity

This project was a test assignment for VK. It demonstrates a system that, given a cropped region of an image (e.g., a logo extracted from a video with embedded advertising), answers the question: 

**"Is this the logo of the target organization?"**

For this task, it is suitable to create embeddings of reference and test images and compare them using any metric, usually the cosine distance or the Euclidean distance. To build embeddings, you can use classical algorithms (SIFT, HOG), or CNN as a feature extractor, but usually neural network solutions show the best result.
It was decided to use ResNet-50 to build logo embeddings and cosine distance as a metric of logo similarity. The model was trained on an open dataset with 3000 tripletloss logos with a cosine distance, that is, at each iteration, 2 elements from one class (anchor and positive object) were taken, 1 from the other (negative object), and the distance between the anchor and the negative object was maximized, and the distance between the anchor and the positive object was minimized. 
Also, for convenience, a web interface was implemented (using streamlit) for uploading reference and test images and displaying the result of an embedding comparison. 

The system compares the provided test logo against several reference logo samples of the organization. It is designed to handle logos with varying scales, brightness, contrast, rotations, and minor design variations. The underlying idea is to use a deep learning model to generate embeddings for each logo and then compute the similarity (via cosine similarity) to decide if the test logo matches one of the provided samples.

Try it: [logo-similarity.streamlit.app](https://logo-similarity.streamlit.app/)

## How It Works

1. **Upload Reference Logos:** Users can upload multiple images representing the target organization's logos.
2. **Upload Test Logo:** Users upload a cropped image containing the test logo.
3. **Comparison:** The system processes each image through a deep learning model, computes embeddings, and calculates cosine similarity between the test logo and each reference.
4. **Result Display:** The system displays the similarity score and a match/no-match result based on a predefined threshold.

## Getting Started

### Requirements

- Python 3.x
- [Streamlit](https://streamlit.io/)
- PyTorch
- PIL

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/1adore1/logo-similarity.git
   cd logo-similarity
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   
### Running the Application

Launch the app using Streamlit:
```
streamlit run src/main.py
```

Then, open the provided URL in your browser to interact with the system.


## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML Model**: `ResNet50`
