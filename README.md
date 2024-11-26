# IST Seminar - GPS Spoofing Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

## Dataset Selection

- drive-me-not: CRI-Lab, “cri-lab-hbku/gps-spoofing-detection-cellular.” Dec. 17, 2023. Accessed: May 01, 2024. [Online].
- Available: https://github.com/cri-lab-hbku/gps-spoofing-detection-cellular

## Existing Methods

- **PCA + One-class Classfier**
  - J. Whelan, T. Sangarapillai, O. Minawi, A. Almehmadi, and K. El-Khatib, “Novelty-based Intrusion Detection of Sensor 
  - Attacks on Unmanned Aerial Vehicles,” in Proceedings of the 16th ACM Symposium on QoS and Security for Wireless and Mobile Networks, in Q2SWinet ’20. New York, NY, USA: Association for Computing Machinery, Nov. 2020, pp. 23–28. doi: 10.1145/3416013.3426446.
  - G. Oligeri, S. Sciancalepore, O. A. Ibrahim, and R. Di Pietro, “Drive me not: GPS spoofing detection via cellular 
  - network: (architectures, models, and experiments),” in Proceedings of the 12th Conference on Security and Privacy in Wireless and Mobile Networks, Miami Florida: ACM, May 2019, pp. 12–22. doi: 10.1145/3317549.3319719.
- **Cumulation of Error**
  - I. Y. Garrett and R. M. Gerdes, “On the Efficacy of Model-Based Attack Detectors for Unmanned Aerial Systems,” 
in Proceedings of the Second ACM Workshop on Automotive and Aerial Vehicle Security, New Orleans LA USA: ACM, Mar. 
2020, pp. 11–14. doi: 10.1145/3375706.3380555.

## Project Structure

- `data/` is dataset directory.
- `models/` contains separated code of classes and function to implement detection models we covered in this repo.
- `notebooks/` includes notebooks for demonstrating the detection algorithms.
- `outputs/` stores plotted assets.
- `save_model/` is where saved detection model is located.
- `slides/` is where the slides for presentation are stored.
- `src/` Notebooks are converted to executable .py in src/
- `utils/` contains basic utilities for data processing, visualization and training of models.
