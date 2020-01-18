### Denoising Documents
#### thresh base
(to see step by step algorithm results, read [description pdf](https://github.com/ZeinabTaghavi/Denoising_Printed_Documents_Threshbase/blob/master/description.pdf)
metho = contours, pixel density

avg of time = 3.108232495

main processes:
1 - by contours, detect and remove small noises.

<img src="https://github.com/ZeinabTaghavi/resolution_enhancement/blob/master/Final_Solution/hierarchy_results/image1.bmp-1-denoise_by_contours.jpg?raw=true" width="30%" height="30%">

2 - slice image to squares and calculate sum of white images(image is inverse).
by pixel density, detect high density areas -> they may be main part of texts in image.
by accirding to high density areas, decide if area is main or not. if not it would be black (image is inverse -> white in real image )
and new image with source.

<img src="https://github.com/ZeinabTaghavi/resolution_enhancement/blob/master/Final_Solution/hierarchy_results/image1.bmp-2-removed_wasted_round_area_pattern.jpg?raw=true" width="30%" height="30%">


3 - if needed repeat spteps 1 and 2

note : amid the code, based on usage, there may be erosion or dilation or other threshs.
