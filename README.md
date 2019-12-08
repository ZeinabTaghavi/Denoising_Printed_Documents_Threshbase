### Denoising Documents
#### thresh base
metho = contours, pixel density

avg of time = 3.108232495

main processes:
1 - by contours, detect and remove small noises.
<img src="https://github.com/ZeinabTaghavi/resolution_enhancement/blob/master/Final_Solution/hierarchy_results/image1.bmp-1-denoise_by_contours.jpg?raw=true" width="10%" height="10%">

2 - slice image to squares and calculate sum of white images(image is inverse).
by pixel density, detect high density areas -> they may be main part of texts in image.
by accirding to high density areas, decide if area is main or not. if not it would be black (image is inverse -> white in real image )
and new image with source.

<img src="https://github.com/ZeinabTaghavi/resolution_enhancement/blob/master/Final_Solution/hierarchy_results/image1.bmp-1-denoise_by_contours.jpg?raw=true" width="10%" height="10%">


3 - if needed repeat spteps 1 and 2

note : amid the code, based on usage, there may be erosion or dilation or other threshs.
