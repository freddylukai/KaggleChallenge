for name in images_009.zip images_010.zip images_011.zip images_012.zip; do
		  mkdir images &&
		  	    gsutil cp gs://nih-chest-xrays/data/$name images &&
			    	        unzip images/$name &> /dev/null &&
							  gsutil -m cp -r images/*.png gs://nih-chest-xrays/data/un/ &&
							  		    rm -r images
									    	      done

