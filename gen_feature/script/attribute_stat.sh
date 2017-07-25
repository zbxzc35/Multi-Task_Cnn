INPUT=$1
OUTPUT=$2

cat $INPUT | python attribute_stat.py > $OUTPUT
