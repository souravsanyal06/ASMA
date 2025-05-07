# ASMA

# 🧠 ASMA: Adaptive Safety Margin Algorithm for Vision-Language Drone Navigation

ASMA is a modular framework for instruction-following drone navigation that integrates vision-language perception with real-time safety through scene-aware control barrier functions (CBFs). It enables collision-aware navigation guided by natural language prompts in complex environments.

## 🛠️ Setup Instructions

1. Clone the repository:
git clone https://github.com/souravsanyal06/ASMA.git
cd ASMA

2. Build the ROS workspace:
cd ros_ws
bash build.sh

3. Run the ASMA demo:
cd ../scripts
python3 asma_city.py

You will be prompted to enter an instruction number corresponding to a predefined natural language command (e.g., 1 for “Go to the house on the left”).

## 📁 Download Required Data

Download the following files from Google Drive and extract them into the root ASMA/ directory:

- dataset.zip: https://drive.google.com/file/d/1GMgwVvNk5HmPSz_MLvAc9g2P6_qAKfTd/view?usp=drive_link
- pretrained.zip: https://drive.google.com/file/d/1gHSTfwxWUhXHuuLAlRK94-qcY9jB_P58/view?usp=drive_link
- videos.zip: https://drive.google.com/file/d/1VBoGVONh2tylEz2IwCaIYHMSyFjxDfDZ/view?usp=drive_link

After extracting, the directory should look like:
ASMA/
├── dataset/
├── pretrained/
└── videos/

## 🧾 Citation

If you use this repository in your work, please cite:

```bibtex
@misc{sanyal2025asmaadaptivesafetymargin,
  title={ASMA: An Adaptive Safety Margin Algorithm for Vision-Language Drone Navigation via Scene-Aware Control Barrier Functions}, 
  author={Sourav Sanyal and Kaushik Roy},
  year={2025},
  eprint={2409.10283},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2409.10283}
}
```

