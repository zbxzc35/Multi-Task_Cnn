function generate_pair_image {
    DAY=$1
    INPUT_PATH=$2
    SAVE_PATH=$3
    JOB_NAME=$4

    hadoop fs -test -e ${SAVE_PATH}
    if [ $? -eq 0 ]
    then
        hadoop fs -rmr ${SAVE_PATH}
    fi

    pig -useHCatalog \
        -p day="${DAY}" \
        -p input_path="${INPUT_PATH}" \
        -p save_path="${SAVE_PATH}" \
        -p job_name="${JOB_NAME}" \
        script/generate_pair_image.pig
}

#generate_pair_image "2017-08-04" "/user/jd_ad/ads_reco/xuzhexuan/multi-task_cnn/pairwise_data" "/user/jd_ad/ads_reco/xuzhexuan/multi-task_cnn/pairwise_data_image" "generate_pair_image"
