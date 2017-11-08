
# ./resize_images.sh /path/to/input/folder/ /path/to/output/folder/ quality resize
# Example of usage:
# ./resize_images.sh ./dataset/original/jpg/ ./dataset/down_2x/  100 50
USAGE="USAGE: ./resize_images.sh /path/to/input/folder/ /path/to/output/folder/ compression_rate width height"
EXAMPLE="EXAMPLE: ./resize_images.sh ./dataset/original/jpg/ ./dataset/compress/  0 300 300"

if [ $# -lt 5 ]
	then
		echo "Missing arguments!!!"
		echo
		echo ${USAGE}
		echo ${EXAMPLE}
		exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2
COMPRESSION_RATE=$3
WIDTH=$4
HEIGHT=$5

echo "Converting images from "${INPUT_DIR} " to " ${OUTPUT_DIR} " with " ${COMPRESSION_RATE}"% compression rate and " ${RESIZE}"% resize..."

#find ${INPUT_DIR} -name '*.jpg' -exec basename {} \; | parallel convert -quality ${QUALITY}% -resize ${RESIZE}% ${INPUT_DIR}/{} ${OUTPUT_DIR}/{}_res_${RESIZE}.jpg
find ${INPUT_DIR} -name '*.png'  -exec   python resize_and_compress.py  --input_path ${INPUT_DIR}/{} --output_path ${OUTPUT_DIR}/{}_w${WIDTH}_h${HEIGHT}_cr${COMPRESSION_RATE}.jpg  --cr ${COMPRESSION_RATE} --width ${WIDTH} --height ${HEIGHT} \;
