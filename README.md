# ASMA

# 🧠 ASMA: Adaptive Safety Margin Algorithm for Vision-Language Drone Navigation

ASMA is a modular framework for instruction-following drone navigation that integrates vision-language perception with real-time safety through scene-aware control barrier functions (CBFs). It enables collision-aware navigation guided by natural language prompts in complex environments.


## 🎥 Demo Visualizations

<table>
  <tr>
    <td align="center"><b>Instruction 1</b><br>
    <em>Go past the first traffic light and go straight past the blue car. After crossing a blue mailbox, turn right at the stop sign, and land in front of the gas station.</em><br>
    <img src="media/cmd1.gif" width="250px">
    </td>
    <td align="center"><b>Instruction 2</b><br>
    <em>Follow the road. After the crossing, fly through the alley before the blue mailbox. Turn left, pass between buildings, turn left at the oak tree, and if a white truck is visible, land in front of it.</em><br>
    <img src="media/cmd2.gif" width="250px">
    </td>
  </tr>
  <tr>
    <td align="center"><b>Instruction 3</b><br>
    <em>Head past the second traffic light. If an ambulance is on the left, fly past it and the stop sign. Enter the gazebo and land inside.</em><br>
    <img src="media/cmd3.gif" width="250px">
    </td>
    <td align="center"><b>Instruction 4</b><br>
    <em>Fly past the first traffic light, then turn right before the gas station. Before the white truck, turn left. At the apartment with stairs, ascend to the third floor and land inside the hallway.</em><br>
    <img src="media/cmd4.gif" width="250px">
    </td>
  </tr>
</table>


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

You will be prompted to enter an instruction number corresponding to a predefined natural language command (e.g., 1 , 2, 3, 4).

## 📁 Download Required Data

Download the following files from Google Drive and extract them into the root ASMA/ directory:

- dataset.zip: https://drive.google.com/file/d/1GMgwVvNk5HmPSz_MLvAc9g2P6_qAKfTd/view?usp=drive_link
- pretrained.zip: https://drive.google.com/file/d/1gHSTfwxWUhXHuuLAlRK94-qcY9jB_P58/view?usp=drive_link
- videos.zip: https://drive.google.com/file/d/1VBoGVONh2tylEz2IwCaIYHMSyFjxDfDZ/view?usp=drive_link

After extracting, your project directory should look like:

```
ASMA/
├── dataset/
├── pretrained/
├── videos/
├── ros_ws/
├── scripts/
├── build.sh
├── README.md
└── other utility files...
```


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

