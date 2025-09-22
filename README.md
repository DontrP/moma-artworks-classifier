# MoMa Artworks Classification with ResNet50

**Art's classification recognition can help museums and archivist organize and recommend artworks.**

Some artworks in digitalized format blur the boundaries between classifications. For example, some printings resemble photographs, while certain photographs are made to look like drawings. This makes classifying artworks challenging but also interesting for machine learning applications.

The goal of this project is to build machine learning model that can automatically classify digitalized artworks into three categories - photographs, drawings, and printings - using data from Museum of Modern Art (MoMA).

To tackle this problem, I use ResNet50, a deep convolutional neutal network known for its ability to recognize patterns in images. I freeze first three layer and finetune layer 4 and the fully connected layer. The dataset comes from [Maven Analytics’ Data Playground](https://mavenanalytics.io/data-playground/the-museum-of-modern-art-(moma)-collection), which contains over 150,000 records of artworks with metadata such as title, artist, medium, classification, and image links.

The final model reach validation accuracy at 88.4% with 0.88 F1 score - showing that the model is balanced and performes consitently on three classes of artworks

The project was inspired by [Alaeddine Grine’s article: Artwork Classification in PyTorch](https://medium.com/@alaeddine.grine/artwork-classification-in-pytorch-b4f3395b877e). In his work, Grine focused on classifying artworks by artistic style (e.g., painting, mural, sculpture) using a curated dataset that spans from the 14th to the 21st century. His project showed how deep learning can be applied to art history, helping us understand the evolution of artistic styles.

