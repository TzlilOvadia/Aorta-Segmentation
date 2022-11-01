
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]




<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#methods">Methods</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
Segmentation, is an essential task in medical image processing. In this project, the main purpose of the project is to get familiarize with medical CT scans and with software environments for CT scans as a self-study and preparation for my final project. I explored and used thresholding techniques for segmentation of the skeleton and the aorta.
In medical image processing, perhaps in contrast to other fields of research you familiar with, it is often the case that we do not have a single correct answer. The main goal of the project is to try to achieve the best segmentation possible, but keep in mind that there is always observer variability, and therefore you should not expect a perfect match (Dice coefficient of 1) with the published solution.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
1. Get the files needed for the project from this link  [get_your_files_here](https://drive.google.com/drive/folders/1ocZ-PZsZMVWFpy5JOjJhQ_zFtov1mASR?usp=sharing).
2. Make sure you install the packages from the requirements.txt file.
3. In order to observe 3D Medical Images (CTs and MRI), you should get access to Medical Images Application.
   1. Windows:
      1. if you're a windows user, I'd recommend you to download ITKSnap.
   2. Mac OS:
      1. If you have installed Rosseta, feel free to go with ITKSnap app, due to the functionality advantages.
      2. Otherwise, I'd recommend you to install MRIcro, which is available on App Store.
   both apps allow you to open 3D images, add extra layers and observe the scan from different angles (front, saggital 
      and coronal).


## Usage
Here's an example for the algorithm performance, using Case_1_CT, Case_1_Aorta and Case_1_L1:
The image below is the one of the test-case used in order to evaluate the algorithm's performance'
the image constructed from a full body CT scan, and two other layers, used as the ground truth for our evaluation later.
in yellow, you can see the ground truth for the patient's Aorta, and in the blue you can see the L1 vertebrate, which is 
the Aorta's desired ROI (region of interest):
![Alt text](compare.png?raw=true "Title")

The next set of images shows the result of the algorithm, colored in orange, as an overlapping volumes, compared to the 
ground truth I mentioned above.
![Alt text](test_case.png?raw=true "Title")



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- METHODS -->
## Methods

In order to implement this algorithm, I used a pipeline model, dividing the problem into small parts, solving them one after another.
### Step #1 - Thresholding and Noise reduction

### Step #2 - Find boundary slices using L1 layer

### Step #3 - Find the Aorta's first ROI circle

### Step #4 - Moving on to next slices, searching for circles nearby the previuosly found ROI.

Denote that using this model, we had to make few basic assumptions regarding to the:
  1. shape of the aorta's ROI (we assumed we are looking for circles)
  2. given a previously found ROU, we assumed ROI on the next slice should have to be nearby the last one.
  Those assumptions are legitimate due to the nature of the given problem - we know the shape and posture of the Aorta, due to the knowledge gathered over the history, putting aside extreme anomalies, mutations etc.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

This project is given as one of the exercises in Professor Leo Joskowic's course (MSc) - Medical Image Processing @ Hebrew University of Jerusalem.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Tzlil Ovadia - [My LinkedIn](https://www.linkedin.com/in/tzlil-ovadia/)

Email - Ovadia.Tzlil@gmail.com

My repositories - [GitHub](https://github.com/TslilOvadia?tab=repositories)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
This project is given as one of the exercises in Professor Leo Joskowic's course (MSc) - Medical Image Processing @ Hebrew University of Jerusalem.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/tzlil-ovadia/
