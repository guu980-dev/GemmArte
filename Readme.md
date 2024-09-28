# Dataset Structure

/custom_vqa_project/
│
├── /dataset/
│   ├── /images/
│   │   ├── train/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   └── val/
│   │       ├── image3.jpg
│   │       └── image4.jpg
│   ├── train.json  # Metadata for the training set
│   └── val.json    # Metadata for the validation set
│
├── /scripts/
│   └── train.py   # Your fine-tuning script
│
└── README.md