# Towards Time-Aware Music Interpolation in Latent Space

![image](https://github.com/user-attachments/assets/da36a9f4-a5ff-4bcb-a380-24da36b77c14)

[Demo page with audio samples](https://realfolkcode.github.io/interpolation_demo/index.html)

In this preliminary work, we interpolate between two music samples in the embedding space of [Music2Latent](https://github.com/SonyCSLParis/music2latent). We compare two classical schemes that allow for averaging two feature sequences with arbitrary lengths: symmetric averaging (by Kruskal and Liberman) and DTW Barycenter Averaging. The former method results in stable samples that are coherent with temporal alignment.

## How to use

1. Clone the repository
2. Install the required packaged via `pip install -r requirements.txt` in your virtual or conda environment
3. Follow the notebooks `barycenters.ipynb` and `interpolation.ipynb` for the examples of usage
