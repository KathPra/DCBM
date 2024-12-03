SUN Attribute Database

The files that make up this dataset are:

	-attributes.mat: A cell array containing all of the 102 attribute words 

	-images.mat: A cell array containing all 14340 image names. 
			The image name strings are formatted in exactly the same way as the
			SUN database. In order to access the jpg for a given image, use the 
			following commands in Matlab:
				>> data_path = '{The path where you have save the SUN Images}';
				>> imshow(strcat(data_path, images{#img}));

	-attributeLabels_continuous.mat: This file contains a 14340x102 matrix. 
					Each row is an attribute occurance vector. 
					Attribute occurance vectors are vectors of real-valued labels
					ranging from 0-1, which correspond to how often a given
					attribute was voted as being present in a given image
					by Mechanical Turk Workers. These continuous values are 
					calculated from 3 votes given by the AMT workers for each
					image.
					The indicies of this matrix correspond with the cell arrays
					in attributes.mat and images.mat, e.g. the first row
					in 'attribute_labels' is the first image in 'images', and
					the first column in 'attribute_labels' is the first attribute
					in 'attributes'.

Please send and questions to gen@cs.brown.edu
					
					
