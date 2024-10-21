---
license: apache-2.0
---

<p align="center">
  <img src="https://i.ibb.co/68ZBkzQ/narrow.png" alt="Description of the image", height="300", width="400">
</p>

# **🖼️** Model description

This model, fine-tuned with LoRA to specialize in art criticism, is based on PaliGemma. It is trained on a diverse dataset that includes a wide range of artworks, from traditional paintings to contemporary art, focusing on 11 detailed aesthetic elements such as form, composition, and symbolism. The combination of PaliGemma's image and language processing capabilities with the LoRA fine-tuning technique provides users with insights into understanding the visual features and artistic intentions of a work from multiple perspectives.

# **🔍** Intended Uses & Limitations

**GemmArte** is designed for:

- Analyzing artworks from various perspectives (formal, historical, emotional, technical, cultural, comparative)
- Providing detailed interpretations of visual elements in paintings
- Assisting in art education, appreciation, and research

**Limitations**:

- The model's knowledge is limited to its training data and the base PaliGemma model
- It may not accurately analyze very recent artworks or obscure art styles
- Performance may vary depending on the quality and clarity of the input image

# **📊** Training dataset

The model training was conducted using the HuggingFace [🔗 Painting-Form dataset](https://huggingface.co/datasets/steven16/Painting-Form). This dataset consists of images of art pieces along with 215 questions and answers pertaining to each image. By leveraging this rich dataset, the model is able to provide detailed insights and critiques, enhancing its ability to understand and interpret various elements of art.

**Datset key features**
- High-quality images of paintings
- Detailed descriptions and analyses of these paintings

<p align="center">
  <img src="https://i.postimg.cc/fLBvbQZJ/image.png)](https://postimg.cc/8fh6G6dX" alt="Description of the image", width="250">
</p>

| question | `Please construct a formal analysis paragraph of approximately 175 words that is coherent and focuses solely on visual characteristics.\n<image>` |
| --- | --- |
| answer | `The painting depicts a young shepherdess, her gaze directed towards the viewer. Her figure is rendered with meticulous detail, her skin glowing with a soft, porcelain-like quality. The folds of her dress are depicted with a delicate touch, creating a sense of movement and fluidity...`  |

## **🔄**  Training Dataset example

The Painting-Form dataset was preprocessed by organizing the 215 questions into 11 categories. Each category is represented by a new, representative question (instruction), creating a refined training dataset. This approach enhances the model's ability to provide structured critiques, focusing on key aesthetic and interpretative elements of art.

[🔗 Preprecessed_dataset](https://www.kaggle.com/datasets/baeksj01/gemmaarte-dataset/data)

| **id** | **image** | **category** | **question** | **answer** |
| --- | --- | --- | --- | --- |
| identity_0 | 2505.png | General Visual Analysis | Create a detailed and cohesive analysis paragr... | The painting depicts a young shepherdess, her ... |
| identity_1 | 13963.png | General Visual Analysis | Create a detailed and cohesive analysis paragr... | The artwork presents a captivating visual symp... |
| identity_2 | 6686.png | General Visual Analysis | Create a detailed and cohesive analysis paragr... | The painting features a rich and vibrant color... |

## **📈** Training Dataset statistics
| images | 13,812 |
| :---: | :---: |
| question- answers | 224,850 |
| question categories | 11 |
| dataset size | 7.5 (GB) |

## **⚙️** Training Procedure

### **🔧** Training Hyperparameters

- Learning rate: 2e-5
- Batch size: 4
- Weight decay: 1e-6
- Number of epochs: 2
- Optimizer: paged_adamw_8bit

### **🖥️** Training Code

To train the model, use the following command:

```bash
python article_base_train.py \
  --dataset_dir "/path/to/your/dataset" \
  --model_id "google/paligemma-3b-pt-224" \
  --output_dir "/path/to/your/output" \
  --use_qlora [true|false]
  --metadata_type [csv|parquet]
```

Note: Replace `/path/to/your/dataset` and `/path/to/your/output` with your actual dataset and output directory paths.

[🔗 article_base_train.py script](https://github.com/guu980-dev/GemmArte)

This script handles the entire training process for the GemmArte model, including data loading, model initialization, and training loop setup.

## **🚀**  Inference

To run inference using the trained model, use the following command

```bash
python test_inference.py
```

[🔗 test_inference.py script](https://github.com/guu980-dev/GemmArte)

<p align="center">
  <img src="https://i.postimg.cc/D0NdqQSh/starry-night.jpg)](https://postimg.cc/8fh6G6dX" alt="Description of the image", width="500" height="500">
</p>

```python
 prompt = "Provide a step-by-step formal analysis of this image, covering: 1) Composition, 2) Color, 3) Line, 4) Texture, 5) Space, and 6) Form.”
 '''
 The composition of this artwork is striking, with a strong emphasis on the vertical and horizontal lines. The starry night sky dominates the canvas, filling the majority of the space with swirling patterns of light and dark. The use of bold, contrasting colors further enhances the dramatic effect of the scene, with the bright yellow stars against the deep blue sky creating a sense of tension and energy. The forms are simplified and stylized, with the only recognizable features being the church steeple and the distant village. The overall effect is one of a dynamic and expressive composition that captures the viewer's attention and draws them into the world of the painting.
 '''
 
 prompt = "Please construct a formal analysis paragraph that is coherent and focuses solely on visual characteristic.”
 '''
 The painting under consideration is a masterful example of the artist's use of color and brushwork to convey a sense of emotion and atmosphere. The dominant hues of blue and green create a striking contrast against the dark, swirling sky, which is rendered in a variety of shades of blue and white. The artist's use of thick, impasto brushstrokes adds texture and depth to the painting, particularly in the depiction of the swirling clouds and the distant mountains. The artist's attention to detail is evident in the intricate rendering of the stars and the church steeple, which are depicted with a combination of delicate, almost ethereal brushstrokes and bold, expressive strokes. The overall effect is a visually striking and emotionally charged work of art that captures the viewer's attention and invites them to contemplate the scene before them.
 '''
```

The model will generate an analysis based on the specified prompt and the provided image. By using these structured prompts, you can guide the model to provide more detailed and comprehensive analyses of artworks.
