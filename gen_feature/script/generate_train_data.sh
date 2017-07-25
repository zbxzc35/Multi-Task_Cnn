function generate_train_data {
    NAME="generate_train_data"
    
    INPUT=$1
    OUTPUT=$2
    JOB_NAME=$3

    MAPPER="${NAME}.mapper.py"
    MAPPER_PATH="script/$MAPPER"
    
    REDUCER="${NAME}.reducer.py"
    REDUCER_PATH="script/$REDUCER"
    
    CATEGORY_ATTRIBUTE="category_attribute"
    CATEGORY_ATTRIBUTE_PATH="script/$CATEGORY_ATTRIBUTE"
 
    LABEL_MAPPING="lable_mapping"
    LABEL_MAPPING_PATH="script/$LABEL_MAPPING"

    hadoop fs -test -e $OUTPUT
    if [ $? -eq 0 ]
    then
        hadoop fs -rmr $OUTPUT
    fi
    
    hadoop jar /software/servers/hadoop-2.2.0/share/hadoop/tools/lib/hadoop-streaming-2.2.0.jar  \
               -D mapred.job.priority=HIGH \
               -D mapred.job.name="$JOB_NAME" \
               -D mapreduce.map.memory.mb=8000 \
               -D mapred.reduce.tasks=0 \
               -cacheArchive "/user/jd_ad/wangjincheng/archive/python2.7.tar.gz#python27"  \
               -input "${INPUT}" \
               -output "${OUTPUT}" \
               -mapper "python27/bin/python2.7 ${MAPPER} --category_attribute=$CATEGORY_ATTRIBUTE --lable_mapping=$LABEL_MAPPING" -file "${MAPPER_PATH}" -file "$CATEGORY_ATTRIBUTE_PATH" -file "$LABEL_MAPPING_PATH"
}
