### Denoising Documents
#### thresh base
metho = contours, pixel density

avg of time = 3.108232495

main process:
1 - by contours, detect and remove small noises.

2 - slice image to squares and calculate sum of white images(image is inverse).
by pixel density, detect high density areas -> they may be main part of texts in image.
by accirding to high density areas, decide if area is main or not. if not it would be black (image is inverse -> white in real image )
and new image with source.

3 - if needed repeat spteps 1 and 2

note : amid the code, based on usage, there may be erosion or dilation or other threshs.
